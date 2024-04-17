import math
import torch
import tetgen
import pyacvd
import numpy as np
import pyvista as pv
from typing import Tuple
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import config as cfg

class XPBDSoftbody:
    def __init__(self) -> None:
        # vertices list
        self.V_list = []
        # tet element list
        self.T_list = []
        self.E_dict = {}
        # mesh list
        self.mesh_list = []
        # offset list (Allows to indentify elements of tet in continuous state list)
        self.offset_list = []
        # split list
        self.split_list = []
        # empty vertex state tensors
        self.V = None
        self.V_velocity = None
        self.V_force = None
        self.V_mass = None
        self.V_w = None
        # distance constraint
        self.C_dist_list = []
        self.C_init_d_list = []
        # volume constraint
        self.C_vol_list = []
        self.C_init_vol_list = []
        # shape matching constraint
        self.C_shape_list = []
        self.C_init_shape_list = []
        # tendon constraint
        self.C_dist_td_list = []
        self.C_init_d_td_list = []
        # integrity flag
        self.init = False
        self.point_fixed = False
        # virtual spring boundary constraint
        self.C_boundary_list = []
        self.C_init_boundary_d_list = []
        

    @staticmethod
    def remesh(mesh: pv.PolyData, n_surf: int=300, save: bool = False) -> pv.PolyData:
        # remesh through pyacvd
        clustered = pyacvd.Clustering(mesh)
        clustered.subdivide(6)
        clustered.cluster(n_surf)
        mesh_remesh = clustered.create_mesh()
        return mesh_remesh

    @classmethod
    def tetrahedralize(cls,
                       mesh: pv.PolyData,
                       n_surf: int = 600) -> Tuple[any, pv.UnstructuredGrid]:
        '''
        This function tetrahedralizes the mesh after converting it into a uniform triangle mesh.

        Args:
            mesh (pv.PolyData): input surface mesh
            n_surf (int): number of faces in output mesh

        Returns:
            (np.array, pv.UnstructuredGrid): tet elements, mesh vertices
        '''
        # remesh
        mesh_remesh = cls.remesh(mesh, n_surf)
        # tetrahedralization
        mesh_tg = tetgen.TetGen(mesh_remesh)
        _, T = mesh_tg.tetrahedralize(
            nobisect=True, order=1, minratio=1.1, mindihedral=20)
        mesh_tet = mesh_tg.grid
        # return tet index array and mesh after tetrahedralization
        return T, mesh_tet
        

    def add_thinshell(self, mesh: pv.PolyData, n_surf = 300) -> pv.PolyData:
        assert(self.init == False)
        # remesh
        mesh_remesh = self.remesh(mesh, n_surf)
        # thinshell T would just be face connections
        V = mesh_remesh.points
        T = mesh_remesh.faces.reshape((-1, 4))[:,1:]
        # store data
        self.V_list.append(V)
        self.T_list.append(T)
        self.mesh_list.append(mesh_remesh)
        self.offset_list.append(sum([V.shape[0] for V in self.V_list[:-1]]))
        # return mesh after tetrahedralization
        return mesh_remesh

    def init_states(self) -> None:
        '''
        Initialize state data structures.
        '''
        assert(self.init == False)
        self.init = True
        # process V_list
        self.V_list = [torch.tensor(V) for V in self.V_list]
        self.split_list = [V.shape[0] for V in self.V_list]
        # vertices
        self.V = torch.cat(self.V_list, axis=0).float().to(cfg.device)
        # velocity
        self.V_velocity = torch.zeros_like(self.V).float().to(cfg.device)
        # force
        self.V_force = torch.zeros_like(self.V).float().to(cfg.device)
        # mass
        self.V_mass = torch.zeros((self.V.shape[0], 1)).float().to(cfg.device)
        # density
        rho = 1000
        for Tidx in range(len(self.T_list)):
            if self.T_list[Tidx] is not None:
                # distribute mass to tet vertex by volume and density
                if self.T_list[Tidx].shape[1] == 4:
                    for tet in self.T_list[Tidx]:
                        v0, v1, v2, v3 = self.V[tet + self.offset_list[Tidx]]
                        volume = torch.dot(torch.cross(v1 - v0, v2 - v0), v3 - v0) / 6
                        assert(volume > 0)
                        mass = volume * rho
                        self.V_mass[tet + self.offset_list[Tidx]] += mass / 4
                # distribute face mass to face vertex
                else:
                    for face in self.T_list[Tidx]:
                        v0, v1, v2 = self.V[face + self.offset_list[Tidx]]
                        area = torch.norm(torch.cross(v1 - v0, v2 - v0), p=2) / 2
                        assert(area > 0)
                        mass = area
                        self.V_mass[face + self.offset_list[Tidx]] += mass / 3
            # default mass for tendon vertex
            else:
                for i in range(self.V_list[Tidx].shape[0]):
                    self.V_mass[i + self.offset_list[Tidx]] = 1
        # inverse mass
        self.V_w = 1 / self.V_mass

        self.V_mass_no_inf = self.V_mass.clone()

        self.init_x_min, self.init_x_max, self.init_y_min, self.init_y_max = self.V[:, 0].min(), self.V[:, 0].max(), self.V[:, 1].min(), self.V[:, 1].max()

    def init_spring_boundary(self):
        '''
        simply double all the data structure
        '''
        assert(self.init == True)
        self.V = torch.cat([self.V, self.V], 0)
        self.V_velocity = torch.cat([self.V_velocity, self.V_velocity], 0)
        self.V_force = torch.cat([self.V_force, self.V_force], 0)
        self.V_mass = torch.cat([self.V_mass, self.V_mass], 0)
        self.V_w = torch.cat([self.V_w, self.V_w], 0)

    def set_gravity(self, gravity_acc: torch.Tensor) -> None:
        '''
        Set the gravitational force on the vertices

        Args:
            gravity_acc (torch.Tensor): 1x3 array containing constant gravitational acceleration
        '''
        assert(self.init == True)
        assert(self.point_fixed == False)
        for i in range(self.V.shape[0]):
            self.V_force[i] = self.V_mass[i, 0] * gravity_acc

    def fix_point(self, idx: int, point_idx: int) -> None:
        assert(self.init == True)
        self.point_fixed = True
        self.V_mass[self.offset_list[idx] + point_idx] = torch.inf
        self.V_w = 1 / self.V_mass

    def fix_virtual_boundary(self) -> None:
        assert(self.init == True)
        N = self.V.shape[0] // 2
        self.V_mass[N:] = torch.inf
        self.V_w = 1 / self.V_mass

    def fix_larger_than(self, idx: int, value: float, axis: int) -> None:
        '''
        Fix all vertices larger than value on axis in space of geometry with index idx

        Args:
            idx (int): Geometry index
            value (float): Lower bound
            axis (int): Axis of minimum value
        '''
        assert(self.init == True)
        self.point_fixed = True
        for i in range(self.V_list[idx].shape[0]):
            if self.V_list[idx][i, axis] > value:
                self.V_mass[self.offset_list[idx] + i] = torch.inf
        self.V_w = 1 / self.V_mass

    def fix_less_than(self, idx: int, value: float, axis: int) -> None:
        '''
        Fix all vertices smaller than value on axis in space of geometry with index idx

        Args:
            idx (int): Geometry index
            value (float): Upper bound
            axis (int): Axis of minimum value
        '''
        assert(self.init == True)
        self.point_fixed = True
        for i in range(self.V_list[idx].shape[0]):
            if self.V_list[idx][i, axis] < value:
                self.V_mass[self.offset_list[idx] + i] = torch.inf
        self.V_w = 1 / self.V_mass

    def fix_indice(self, idx: int, indices: list) -> None:
        assert(self.init == True)
        self.point_fixed = True
        for i in indices:
            self.V_mass[self.offset_list[idx] + i] = torch.inf
        self.V_w = 1 / self.V_mass

    def fix_picked_points(self, idx: int, picked_points: torch.Tensor, threshold: float) -> None:
        assert(self.init == True)
        self.point_fixed = True
        for i in range(self.V_list[idx].shape[0]):
            for pp in picked_points:
                if torch.norm(self.V_list[idx][i] - pp) < threshold:
                    self.V_mass[self.offset_list[idx] + i] = torch.inf
        self.V_w = 1 / self.V_mass

    def find_points_indice(self, idx: int, picked_points: torch.Tensor, threshold: float) -> None:
        assert(self.init == True)
        indices = []
        for i in range(self.V_list[idx].shape[0]):
            for pp in picked_points:
                if torch.norm(self.V_list[idx][i] - pp) < threshold:
                    indices.append(self.offset_list[idx] + i)
        return indices


    def find_point(self, idx: int, point: torch.Tensor, threshold: float) -> int:
        assert(self.init == True)
        for i in range(self.V_list[idx].shape[0]):
            if torch.norm(self.V_list[idx][i] - point) < threshold:
                return i

    def init_boundary_constraints(self) -> None:
        assert(self.init == True)
        N = self.V.shape[0] // 2 # N is the number of real particles
        self.C_boundary_list = [torch.stack([torch.tensor([i, i+N]).long().to(cfg.device) for i in range(N)])]
        self.C_init_boundary_d_list = [torch.zeros(N, 1).to(cfg.device)]
        
    
    def init_grasp_constraints(self, loc, radius=None, k=None):
        assert(self.init == True)
        self.grasp_point = loc.to(cfg.device)
        self.C_grasp_list, self.C_grasp_d_list = [], []
        dist = torch.norm(self.V[:self.V.shape[0]//2] - self.grasp_point, dim=-1, keepdim=True)

        if k is not None:
            
            smallest_k = dist.topk(k, dim=0, largest=False)[1].squeeze()
            self.C_grasp_list.append(smallest_k)
            self.C_grasp_d_list.append(dist[smallest_k])
        else:
            picked = dist < radius
            self.C_grasp_list.append(torch.nonzero(picked)[:, 0].long())
            self.C_grasp_d_list.append(dist[torch.nonzero(picked)[:, 0].tolist()])

        print(f'control point connects to {self.C_grasp_list[0].shape[0]} vertices')

    def init_target_area(self, loc, radius=None, k=None):
        assert(self.init == True)
        self.target_point = loc.to(cfg.device)
        self.target_list = []
        dist = torch.norm(self.V[:self.V.shape[0]//2] - self.target_point, dim=-1, keepdim=True)

        if k is not None:
            
            smallest_k = dist.topk(k, dim=0, largest=False)[1].squeeze()
            self.target_list.append(smallest_k)
        else:
            picked = dist < radius
            self.target_list.append(torch.nonzero(picked)[:, 0].long())

        print(f'target point connects to {self.target_list[0].shape[0]} vertices')

    def init_control_area(self, loc, radius=None, k=None):
        assert(self.init == True)
        self.control_point = loc.to(cfg.device)
        self.control_list = []
        dist = torch.norm(self.V[:self.V.shape[0]//2] - self.control_point, dim=-1, keepdim=True)

        if k is not None:
            smallest_k = dist.topk(k, dim=0, largest=False)[1].squeeze()
            self.control_list.append(smallest_k)
        else:
            picked = dist < radius
            self.control_list.append(torch.nonzero(picked)[:, 0].long())

        print(f'target point connects to {self.control_list[0].shape[0]} vertices')
        

    def init_dist_constraints(self) -> None:
        '''
        Initialize the distance and tendon distance constraints
        self.C_dist_list contains vertex indices
        self.C_init_d_list contains initial constraint values
        '''
        assert(self.init == True)
        # distance constraints independent set
        C_set_list = []
        for Tidx in range(len(self.T_list)):
            # tet distance constraint
            if self.T_list[Tidx] is not None:
                assert(self.T_list[Tidx].shape[1] == 3)
                self.__init_dist_face_constraints(Tidx, C_set_list)
        # update inverse mass
        self.V_w = 1 / self.V_mass
        # convert each set to torch tensor
        self.C_dist_list = [torch.tensor(C_dist).long().to(cfg.device) for
                            C_dist in self.C_dist_list]
        self.C_init_d_list = [torch.tensor(C_init_d).reshape((-1, 1)).to(cfg.device) for
                              C_init_d in self.C_init_d_list]
        # self.C_dist_td_list = [torch.tensor(C_dist_td).to(cfg.device) for
        #                        C_dist_td in self.C_dist_td_list]
        # self.C_init_d_td_list = [torch.tensor(C_init_d_td).reshape((-1, 1)).to(cfg.device) for
        #                          C_init_d_td in self.C_init_d_td_list]


    def __init_dist_face_constraints(self, idx: int, C_set_list: list) -> None:
        # record triangle edge already added and its opposite vertex
        edge_vertex_map = dict()
        # distance constraint for each edge each tet
        for face in self.T_list[idx]:
            i0, i1, i2 = face + self.offset_list[idx]
            # all 3 edge-vertex pair in face
            edge_vertex_list = [((i0, i1), i2), ((i0, i2), i1), ((i1, i2), i0)]
            for edge, ops_vertex in edge_vertex_list:
                # no need to add current edge, but add diagonal with adjacent triangle
                if edge in edge_vertex_map.keys():
                    adj_ops_vertex = edge_vertex_map[edge]
                    edge = (ops_vertex, adj_ops_vertex)
                # no need to add current edge, but add diagonal with adjacent triangle
                elif edge[::-1] in edge_vertex_map.keys():
                    adj_ops_vertex = edge_vertex_map[edge[::-1]]
                    edge = (ops_vertex, adj_ops_vertex)
                # only add constraint for new edge
                elif edge not in edge_vertex_map.keys() and \
                    edge[::-1] not in edge_vertex_map.keys():
                    edge_vertex_map[edge] = ops_vertex
                    edge_vertex_map[edge[::-1]] = ops_vertex
                # record compatible independent set
                min_idx = -1
                min_size = math.inf
                create_new = True
                # find compatible independent set with minimum size
                for i in range(len(C_set_list)):
                    if edge[0] not in C_set_list[i] and \
                    edge[1] not in C_set_list[i]:
                        create_new = False
                        if len(C_set_list[i]) < min_size:
                            min_idx = i
                            min_size = len(C_set_list[i])
                # if no compatible set, create a new set, idx would just be -1
                if create_new:
                    C_set_list.append(set())
                    self.C_dist_list.append([])
                    self.C_init_d_list.append([])
                # add to set
                C_set_list[min_idx].add(edge[0])
                C_set_list[min_idx].add(edge[1])
                # add distance constraint
                self.C_dist_list[min_idx].append(list(edge))
                # compute initial distance
                v0, v1 = self.V[list(edge)]
                dist = torch.norm(v1 - v0, p=2)
                self.C_init_d_list[min_idx].append(dist)

        # adjacency list for each vertex
        for key in edge_vertex_map.keys():
            v1, v2 = key
            if v1 not in self.E_dict:
                self.E_dict[v1] = []
            else:
                self.E_dict[v1].append(v2)
            if v2 not in self.E_dict:
                self.E_dict[v2] = []
            else:
                self.E_dict[v2].append(v1)
        # remove duplicants
        for key in self.E_dict.keys():
            self.E_dict[key] = list(set(self.E_dict[key]))

    def init_shape_constraints_thinshell(self) -> None:
        '''
        Initializes shape matching constraints
        '''
        assert (self.init == True)
        # volume constraints independent set
        C_set_list = []
        for Tidx in range(len(self.T_list)):
            if self.T_list[Tidx] is not None and self.T_list[Tidx].shape[1] == 3:
                # volume constraint for each tet
                for face in self.T_list[Tidx]:
                    i0, i1, i2 = face + self.offset_list[Tidx]
                    # record compatible independent set
                    min_idx = -1
                    min_size = math.inf
                    create_new = True
                    # find compatible independent set (not all four vertices in set) with minimum size (prefer smaller sets)
                    for i in range(len(C_set_list)):
                        if i0 not in C_set_list[i] and i1 not in C_set_list[i] and i2 not in C_set_list[i]:
                            create_new = False
                            if len(C_set_list[i]) < min_size:
                                min_idx = i
                                min_size = len(C_set_list[i])

                    # if no compatible set, create a new set, idx would just be -1
                    if create_new:
                        C_set_list.append(set())
                        self.C_shape_list.append([])
                        self.C_init_shape_list.append([])
                    # add to set
                    C_set_list[min_idx].add(i0)
                    C_set_list[min_idx].add(i1)
                    C_set_list[min_idx].add(i2)

                    # add shape matching constraint
                    self.C_shape_list[min_idx].append(torch.tensor([i0, i1, i2]).long())

                    # compute initial center of mass
                    weighted_pos = torch.mul(self.V_mass[[i0, i1, i2]], self.V[[i0, i1, i2]])
                    rest_com = torch.sum(weighted_pos, 0) / torch.sum(self.V_mass[[i0, i1, i2]])

                    rest_local_vect = self.V[[i0, i1, i2]] - rest_com

                    self.C_init_shape_list[min_idx].append(rest_local_vect)
                    
        # convert each set to torch tensor
        self.C_shape_list = [torch.stack(C_shape) for C_shape in self.C_shape_list]
        self.C_init_shape_list = [torch.stack(C_init_shape) for C_init_shape in self.C_init_shape_list]


    def update_mesh(self) -> None:
        assert(self.init == True)
        V_new_list = torch.split(self.V, self.split_list)
        for i in range(len(V_new_list)):
            self.mesh_list[i].points[:] = V_new_list[i].detach().numpy()[:]

    def convert_to_normalize_coordinate(self, coors):
        pad_x_min = self.init_x_min - 0.01
        pad_x_max = self.init_x_max + 0.01
        pad_y_min = self.init_y_min - 0.01
        pad_y_max = self.init_y_max + 0.01
        new_coors = coors.clone()
        new_coors[..., 0] -= pad_x_min
        new_coors[..., 0] /= pad_x_max - pad_x_min
        new_coors[..., 1] -= pad_y_min
        new_coors[..., 1] /= pad_y_max - pad_y_min
        return new_coors
    
    def define_contact_field(self, bottom_points):
        grid_x, grid_y = np.mgrid[0:1:700j, 0:1:700j]
        norm_coors = self.convert_to_normalize_coordinate(bottom_points)
        points, value = norm_coors[:, :-1], norm_coors[:, -1:]
        grid = griddata(points.cpu().numpy(), value.cpu().numpy(), (grid_x, grid_y), method='nearest')
        grid = ndimage.gaussian_filter(grid, sigma=(7, 7, 0), order=0)
        self.contact_field = torch.from_numpy(grid).squeeze().float().to(cfg.device)
        
    
        