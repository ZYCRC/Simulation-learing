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
                               [0.6,  0.0367,  0.83]])
control_trajectory[:, 2] += 0.24

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
skin_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0), x_length=1, y_length=0.8, z_length=0.24), n_surf=N_skin)
bandage_mesh = softbody.add_thinshell(pv.Cube(center=(0, 0, 0.24), x_length=1, y_length=0.8, z_length=0.24), n_surf=N_bandage)
softbody.init_states()
softbody.init_dist_constraints()
# softbody.init_rigid_constraints(1, 0.3)
# softbody.init_shape_constraints_thinshell([1])
# softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
softbody.fix_less_than(0, 0, 2)
softbody.fix_point(1, 306)
softbody.add_multi_boundary_constrain(1, 0, 0.1, range(198, 296))

# %%
softbody.V[400 + 306]

# %%
# color = np.zeros(400)
# color[306] = 1

# %%
while True:
    pl = pv.Plotter()
    pl.add_mesh(skin_mesh, color='#ffdbac', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Deformable object')
    pl.add_mesh(bandage_mesh, color='#D5A97D', show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
    pl.add_lines(control_trajectory.cpu().numpy(), connected=True, color='r')
    # pl.add_mesh(bandage_mesh, scalars=color, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', label='Bandage')
    pl.add_legend()

    pl.show()
    print(pl.camera_position)