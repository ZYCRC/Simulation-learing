# %%
from xpbd_softbody import XPBDSoftbody
import pyvista as pv
import config as cfg
import matplotlib.pyplot as plt
import torch
from xpbd_softbody_layer import XPBDStep
import numpy as np
from sklearn.neighbors import KDTree
from xpbd_softbody_layer import get_energy_boundary
from scipy.interpolate import interp1d
from cubic_bezier import *
from tqdm import trange
from torchviz import make_dot

# %%
N_skin = 400
N_bandage = 400
control_trajectory = np.array([[-0.4677,  0.0367,  0.1300],
                               [-0.4,  0.0367,  0.2300],
                               [-0.3,  0.0367,  0.33],
                               [-0.2,  0.0367,  0.43],
                               [-0.0,  0.0367,  0.53],
                               [0.2,  0.0367,  0.63],
                               [0.4,  0.0367,  0.73],
                               [0.6,  0.0367,  0.83]]) / 100
# control_trajectory = np.array([[-0.4677,  0.0367,  0.1300],
#                                [-0.4677,  0.0367,  0.40],
#                                [-0.4677,  0.0367,  0.60],
#                                [-0.4677,  0.0367,  0.99]])
control_trajectory[:, 2] += 0.0024

# %%
# interpolate trajectory
x = np.arange(control_trajectory.shape[0])
xnew = np.linspace(x.min(), x.max(), 10 * control_trajectory.shape[0])  # 10 times denser
f = interp1d(x, control_trajectory, axis=0, kind='cubic')
control_trajectory = f(xnew)
control_trajectory = torch.from_numpy(control_trajectory).to(cfg.device)

# %%
# load data
softbody = XPBDSoftbody()
skin_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0), x_length=0.01, y_length=0.008, z_length=0.0024), n_surf=N_skin)
bandage_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0.0024), x_length=0.01, y_length=0.008, z_length=0.0024), n_surf=N_bandage)
softbody.init_states()
softbody.init_dist_constraints()
softbody.add_multi_boundary_constrain(1, 0, 0.001, range(198, 296))
softbody.fix_less_than(0, 0, 2)
softbody.fix_point(1, 306)

# %%
softbody.V[400 + 306]

# %%
# color = np.zeros(400)
# color[306] = 1

# %%
# softbody.fix_point(1, 1)

# %%
cloth_dist_stiffness = 1
V_boundary_stiffness_ref = 0.1
V_dist_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device) * cloth_dist_stiffness
V_boundary_stiffness = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device) * V_boundary_stiffness_ref
V_shape_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)*3

# %%
filename = 'tissue2tissue_stiffness_' + str(V_boundary_stiffness_ref) + '.gif'
stiffness_text = 'Boundary stiffness = ' + str(V_boundary_stiffness_ref)

# %%
pl = pv.Plotter()
pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2')
pl.add_lines(control_trajectory.cpu().numpy(), connected=True, color='r')
pl.add_text(stiffness_text)
# pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
pl.add_legend()
pl.camera_position = 'iso'
pl.show()
print(pl.camera_position)

# %%
softbody.V[306]

# %%
pl.open_gif(filename)
for i in range(control_trajectory.shape[0]):
    softbody.V[softbody.offset_list[1] + 306] = control_trajectory[i]
    step_ref = XPBDStep(softbody,
                V_dist_stiffness=V_dist_stiffness, 
                V_shape_stiffness=V_shape_stiffness,
                V_boundary_stiffness=V_boundary_stiffness, 
                dt=cfg.dt,
                substep=cfg.substep,
                iteration=cfg.iteration,
                quasi_static=cfg.quasi_static,
                plane_height=cfg.ground_plane_height, 
                use_shape_matching=True,
                use_spring_boundary=True,
                use_dist=True) #cfg.use_spring_boundary
    V_ref, V_velocity_ref = step_ref.forward(softbody.V, softbody.V_velocity)
    softbody.V = V_ref.clone()
    softbody.V_velocity = V_velocity_ref.clone()
    energy = get_energy_boundary(softbody, softbody.V, V_boundary_stiffness)
    # V_boundary_stiffness[energy.squeeze() > 0.5] = 1e-8

    V_boundary_stiffness = V_boundary_stiffness * torch.sigmoid(5e9 * (1e-8 - energy)) + 1e-8 * torch.sigmoid(5e9 * (energy - 1e-8))
    # print(energy)
    skin_mesh.points = softbody.V[:N_skin].cpu().numpy()
    bandage_mesh.points = softbody.V[N_skin:N_bandage+N_skin].cpu().numpy()
    # pl.remove_actor(bandage_actor)
    # bandage_actor = pl.add_points(softbody.V[N_skin:N_skin+N_bandage].cpu().numpy(), color='r')
    pl.write_frame()
pl.close()


