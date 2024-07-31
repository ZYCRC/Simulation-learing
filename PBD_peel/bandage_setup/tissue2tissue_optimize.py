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
from torchmin import minimize_constr

picked_list = []

def find_idx(pos):
    pos_torch = torch.tensor(pos)
    result = torch.sum(softbody.V_list[1] == pos_torch, 1)
    idx = int(torch.where(result == 3)[0][0])
    picked_list.append(idx)
# %%
N_skin = 400
N_bandage = 400

# %%
# create cubic bezier spline control
u = torch.linspace(0, 1, 3).to(cfg.device)
us = torch.linspace(0, 1, 50).to(cfg.device)
start_point = np.array([[0.0099, 0.0014, 0.0013]])
start_point = torch.from_numpy(start_point).to(cfg.device)
## Parallel peeling control
# spline_control = np.array([ [0,  0.0367,  0.57],
#                             [1,  0.0367,  0.67],
#                             [2,  0.0367,  0.77]])
# Straight up peeling control
spline_control = np.array([[0.0099, 0.0014, 0.0023],
                           [0.0099, 0.0014, 0.0033],
                           [0.0099, 0.0014, 0.0043],])
spline_control = torch.from_numpy(spline_control).to(cfg.device)

# %%
contact_sur = torch.tensor([400, 401, 402, 403, 404, 406, 407, 408, 409, 416, 417, 418, 419, 420,
         421, 422, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 444,
         445, 447, 448, 450, 451, 452, 467, 468, 469, 470, 471, 472, 473, 474,
         475, 476, 478, 479, 492, 493, 494, 495, 496, 497, 506, 507, 508, 509,
         510, 511, 512, 513, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528,
         529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542,
         543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556,
         557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570,
         571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
         585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
         599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612,
         613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626,
         627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640,
         641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654,
         655, 656, 657, 658, 659, 660, 661]) - 400

# %%
# load data
softbody = XPBDSoftbody()
skin_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0), x_length=0.02, y_length=0.02, z_length=0.002), n_surf=N_skin)
bandage_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0.002), x_length=0.02, y_length=0.02, z_length=0.002), n_surf=N_bandage)
softbody.init_states()
softbody.init_dist_constraints()
softbody.init_shape_constraints_thinshell([1])
softbody.add_multi_boundary_constrain(1, 0, 0.0019, contact_sur)
# softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
softbody.fix_less_than(0, 0, 2)
softbody.fix_point(1, 50)
# softbody.fix_less_than(1, 0.002, 2)

# %%
norm_vec = np.array([-1, -1, 0])
center = (0, 0, 0)
sig_plane = pv.Plane(center=center, direction=norm_vec, i_size=0.01, j_size=0.01)

# %%
pl = pv.Plotter()
# pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2')
# pl.add_mesh(sig_plane)
pl.add_points(softbody.V[softbody.offset_list[1] + 50].cpu().numpy(), color='r')
# pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
pl.add_legend()
pl.camera_position = 'iso'
pl.enable_point_picking(show_message='Pick a point', callback=find_idx)
pl.show()
print(picked_list)

# %%
# V_origin = softbody.V.clone()
# V_velocity_origin = softbody.V_velocity.clone()

# # %%
# softbody.C_boundary_list[0].shape

# # %%
# # color = np.zeros(400)
# # color[306] = 1

# # %%
# cloth_dist_stiffness = 1
# V_boundary_stiffness_ref = 0.1
# V_dist_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device) * cloth_dist_stiffness
# V_boundary_stiffness = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device) * V_boundary_stiffness_ref
# V_boundary_stiffness = V_boundary_stiffness.type(torch.DoubleTensor)
# V_shape_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)*0.0005
# energy_threshold = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device).type(torch.DoubleTensor) * 1e-8
# energy_max = 1e-7
# energy_min = 1e-7

# # %%
# center = torch.tensor([0, 0, 0])
# N_norm = torch.norm(softbody.V_list[1][contact_sur] - center, dim=1)
# stiffness_color = []
# for i in range(contact_sur.shape[0]):
#     energy_threshold[softbody.C_boundary_lut_0[0][i]] = N_norm[i]*(energy_max - energy_min) / (N_norm.max() - N_norm.min()) + energy_max - N_norm.max()*(energy_max - energy_min) / (N_norm.max() - N_norm.min())
#     stiffness_color.append(N_norm[i]*(energy_max - energy_min) / (N_norm.max() - N_norm.min()) + energy_max - N_norm.max()*(energy_max - energy_min) / (N_norm.max() - N_norm.min()))

# # %%
# V_boundary_stiffness_origin = V_boundary_stiffness.clone()
# energy_coff = 15 / energy_threshold

# # %%
# pl = pv.Plotter()
# pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
# pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2', opacity=0.5)
# # pl.add_mesh(sig_plane)
# pl.add_points(softbody.V_list[1][contact_sur].cpu().numpy(), scalars=stiffness_color, label='stiffness')
# # pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
# pl.add_legend()
# pl.camera_position = 'iso'
# pl.show()

# # %%
# filename = 'tissue2tissue_stiffness_' + str(V_boundary_stiffness_ref) + '.gif'
# stiffness_text = 'Boundary stiffness = ' + str(V_boundary_stiffness_ref)

# # %%
# select_points = []
# for i in range(softbody.V_list[1].shape[0]):
#     if (norm_vec @ softbody.V_list[1][i].detach().cpu().numpy()) < 0:
#         select_points.append(i + softbody.offset_list[1])

# # %%
# reveal_points = []
# for i in softbody.C_boundary_V_0[0]:
#     if i in select_points:
#         reveal_points.append(i)

# # %%
# pl = pv.Plotter()
# pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
# pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2', opacity=0.5)
# # pl.add_mesh(sig_plane)
# pl.add_points(softbody.V[reveal_points].cpu().numpy(), color='r', label='Attachment need to be revealed')
# # pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
# pl.add_legend()
# pl.camera_position = 'iso'
# pl.show()

# # %%
# boundary_mask = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device) * 0.1
# for i in reveal_points:
#     idx = torch.where(softbody.C_boundary_V_0[0] == i)[0]
#     # print(idx)
#     boundary_idx = softbody.C_boundary_lut_0[0][idx]
#     boundary_mask[boundary_idx] = 1e-8

# # %%
# softbody.V[softbody.offset_list[1] + 50]

# # %%
# # Set grad parameter
# spline_control.requires_grad_(True)

# # %%
# optimizer = torch.optim.Adam([spline_control], lr=100)

# # %%
# # define loss function
# # target = torch.ones_like(V_boundary_stiffness) * 1e-5

# def loss_fn(predict, target, penalty, alpha=1e5):
#     return torch.norm(target - predict) + penalty * alpha

# # %%
# ## evaluation function
# def eval(predict, target):
#     predict_revealed = predict < 1e-2
#     predict_unrevealed = predict > 1e-2
#     # print(torch.sum(predict_revealed))
#     target_revealed = target < 1e-2
#     target_unrevealed = target > 1e-2

#     # calculate arrcuary reveal
#     reveal_acc = torch.sum(torch.logical_and(predict_revealed, target_revealed)) / torch.sum(target_revealed)
#     unreveal_acc = torch.sum(torch.logical_and(predict_unrevealed, target_unrevealed)) / torch.sum(target_unrevealed)

#     return reveal_acc, unreveal_acc

# # %%
# softbody.V[softbody.offset_list[1] + 306]

# # %%
# # torch.autograd.set_detect_anomaly(False)
# softbody.C_boundary_list[0].shape

# # %%
# spline_list = []

# # %%
# torch.sigmoid(torch.tensor(10))

# # %%

# # pl.open_gif(filename)
# for t in range(60):
#     print('itr', t)
#     # get cubic bezier spline control after step
#     x_con = torch.cat((start_point[:, 0], spline_control[:, 0]))
#     y_con = torch.cat((start_point[:, 1], spline_control[:, 1]))
#     z_con = torch.cat((start_point[:, 2], spline_control[:, 2]))
#     spline = cubic_bezier_arc_3D(x_con, y_con, z_con, 0.0005)
#     spline_x = spline[0]
#     spline_y = spline[1]
#     spline_z = spline[2]
#     spline_trajectory = torch.transpose(torch.vstack((spline_x, spline_y, spline_z)), 0, 1)
#     # print(spline_trajectory)
#     # restore original vertex and velocity
#     softbody.V = V_origin.clone()
#     softbody.V_velocity = V_velocity_origin.clone()

#     # restore stiffness
#     cloth_dist_stiffness = 1
#     V_boundary_stiffness_ref = 0.1
#     V_dist_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)
#     V_boundary_stiffness = V_boundary_stiffness_origin.clone()
#     V_shape_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)*0.0005
    
#     for i in trange(spline_trajectory.shape[0]):
#     # for i in range(1):
#         softbody.V[softbody.offset_list[1] + 50] = spline_trajectory[i]
        
#         step_ref = XPBDStep(softbody,
#                     V_dist_stiffness=V_dist_stiffness, 
#                     V_shape_stiffness=V_shape_stiffness,
#                     V_boundary_stiffness=V_boundary_stiffness, 
#                     dt=cfg.dt,
#                     substep=cfg.substep,
#                     iteration=cfg.iteration,
#                     quasi_static=cfg.quasi_static,
#                     plane_height=cfg.ground_plane_height, 
#                     use_shape_matching=True,
#                     use_spring_boundary=True,
#                     use_dist=True)
#         V_ref, V_velocity_ref = step_ref.forward(softbody.V, softbody.V_velocity)
#         softbody.V = V_ref.clone()
#         softbody.V_velocity = V_velocity_ref.clone()
#         # print((softbody.V == torch.inf).any())
#         ref_V_boundary_stiffness = V_boundary_stiffness.clone()
#         energy = get_energy_boundary(softbody, softbody.V, ref_V_boundary_stiffness)
#         V_boundary_stiffness = V_boundary_stiffness * torch.sigmoid(energy_coff * (energy_threshold - energy)) + 1e-8 * torch.sigmoid(energy_coff * (energy - energy_threshold))
#         # energy_level = torch.abs(energy_threshold - energy) / (energy_threshold - energy)
#         # V_boundary_stiffness = (V_boundary_stiffness * energy_level + V_boundary_stiffness) / 2 + (1e-8 * (-energy_level) + 1e-8) / 2
#         # V_boundary_stiffness = V_boundary_stiffness * torch.sigmoid(9 * energy_level) + 1e-8 * torch.sigmoid(9 * energy_level)
#         skin_mesh.points = softbody.V[:N_skin].detach().cpu().numpy()
#         bandage_mesh.points = softbody.V[N_skin:N_bandage+N_skin].detach().cpu().numpy()

#     # loss = torch.norm(V_boundary_stiffness)
#     constrain = torch.sigmoid(1e5 * (start_point[:, 2] - spline_trajectory[1:, 2]))
#     constrain = torch.sum(constrain)
#     loss = loss_fn(V_boundary_stiffness, boundary_mask, constrain, alpha=1)
#     loss.backward()
#     print('loss:', loss)
#     print('Accuracy:', eval(V_boundary_stiffness, boundary_mask))
#     print('constrain', constrain)
#     print(spline_control.grad)
#     spline_list.append(spline_trajectory)
#     if t % 5 == 0:
#         pl = pv.Plotter()
#         pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
#         pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2')
#         pl.add_lines(spline_trajectory.detach().cpu().numpy(), connected=True, color='r')
#         pl.add_text(stiffness_text)
#         # pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
#         pl.add_legend()
#         pl.camera_position = 'iso'
#         pl.show()
        
#     optimizer.step()
#     optimizer.zero_grad()

# # pl.close()

# # %%
# pl = pv.Plotter()
# pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 1')
# pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object 2')
# pl.add_text(stiffness_text)
# # pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
# pl.add_legend()
# pl.camera_position = 'iso'

# # %%
# # pl.open_gif('tissue2tissue_opt_dist+boundary+shape_various_stiffness_46_itrs.gif')
# for spline_trajectory in spline_list[0:50:2]:
#     softbody.V = V_origin.clone()
#     softbody.V_velocity = V_velocity_origin.clone()

#     # restore stiffness
#     cloth_dist_stiffness = 1
#     V_boundary_stiffness_ref = 0.1
#     V_dist_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)
#     V_boundary_stiffness = torch.ones((softbody.C_boundary_list[0].shape[0], 1)).to(cfg.device) * V_boundary_stiffness_ref
#     V_shape_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device)*0.0005
    
#     spline_actor = pl.add_lines(spline_trajectory.detach().cpu().numpy(), connected=True, color='r')
#     for i in trange(spline_trajectory.shape[0]):
#         softbody.V[softbody.offset_list[1] + 50] = spline_trajectory[i]
        
#         step_ref = XPBDStep(softbody,
#                     V_dist_stiffness=V_dist_stiffness, 
#                     V_shape_stiffness=V_shape_stiffness,
#                     V_boundary_stiffness=V_boundary_stiffness, 
#                     dt=cfg.dt,
#                     substep=cfg.substep,
#                     iteration=cfg.iteration,
#                     quasi_static=cfg.quasi_static,
#                     plane_height=cfg.ground_plane_height, 
#                     use_shape_matching=True,
#                     use_spring_boundary=True,
#                     use_dist=True)
#         V_ref, V_velocity_ref = step_ref.forward(softbody.V, softbody.V_velocity)
#         softbody.V = V_ref.clone()
#         softbody.V_velocity = V_velocity_ref.clone()
#         # print((softbody.V == torch.inf).any())
#         ref_V_boundary_stiffness = V_boundary_stiffness.clone()
#         energy = get_energy_boundary(softbody, softbody.V, ref_V_boundary_stiffness)
#         V_boundary_stiffness = V_boundary_stiffness * torch.sigmoid(1e9 * (1e-8 - energy)) + 1e-8 * torch.sigmoid(1e9 * (energy - 1e-8))
#         skin_mesh.points = softbody.V[:N_skin].detach().cpu().numpy()
#         bandage_mesh.points = softbody.V[N_skin:N_bandage+N_skin].detach().cpu().numpy()
#         print(V_boundary_stiffness)
#         # pl.write_frame()
#     pl.remove_actor(spline_actor)
#     # loss = torch.norm(V_boundary_stiffness)

# # pl.close()


