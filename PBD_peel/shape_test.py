# %%
from xpbd_softbody import XPBDSoftbody
import pyvista as pv
import config as cfg
import matplotlib.pyplot as plt
import torch
from xpbd_softbody_layer import XPBDStep

# %%
#load a shape
softbody = XPBDSoftbody()
mesh = softbody.add_thinshell(pv.Plane(), n_surf=10)
softbody.init_states()
softbody.set_gravity(torch.tensor([0, 0, -9.8]).to(cfg.device))
softbody.init_dist_constraints()
softbody.fix_point(0, 1)
softbody.init_shape_constraints_thinshell()

# %%
statics = torch.zeros(10)
for entry in softbody.C_shape_list:
    for item in entry:
        statics[item] += 1

# %%
pl = pv.Plotter()
pl.add_mesh(mesh, scalars=statics, show_edges=True, edge_color='#b37164ff',  lighting=False,style='surface', cmap='jet')
pl.show()

# %%
V_shape_stiffness = torch.ones_like(softbody.V_mass).to(cfg.device) * 0.1

# %%
for i in range(100):
    print(i)
    step_ref = XPBDStep(softbody,
                V_dist_stiffness=None, 
                V_shape_stiffness=V_shape_stiffness,
                V_boundary_stiffness=None, 
                dt=cfg.dt,
                substep=cfg.substep,
                iteration=cfg.iteration,
                quasi_static=cfg.quasi_static,
                plane_height=cfg.ground_plane_height, 
                use_shape_matching=True,
                use_spring_boundary=False,
                use_dist=False) #cfg.use_spring_boundary
    V_ref, V_velocity_ref = step_ref.forward(softbody.V, softbody.V_velocity)
    softbody.V = V_ref.clone()
    softbody.V_velocity = V_velocity_ref.clone()
    mesh.points = softbody.V.cpu().numpy()
    pl.show(interactive_update=True)


