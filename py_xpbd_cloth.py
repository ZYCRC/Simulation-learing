import numpy as np
import open3d as o3d

# side vertex count
n = 5
# create grid
x, z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))

# transform grid to V matrix
x = x.reshape((-1, 1))
y = np.ones_like(x)
z = z.reshape((-1, 1))
# stack together, V of shape (n^2, 3)
V = np.hstack((x, y, z))

# predict
V_predict = np.array(V)
# velocity
V_velocity = np.zeros_like(V)
# force
V_force = np.zeros_like(V)
for i in range(V_force.shape[0]):
    V_force[i] = np.array([0, -9.8, 0])
# mass
V_mass = np.ones((V.shape[0], 1))
V_mass[0, 0] = np.inf
V_mass[n-1, 0] = np.inf
# inverse mass
V_w = 1 / V_mass

# face index matrix
F = np.zeros((2 * (n-1)**2, 3), dtype=int)
for row in range(n-1):
    for col in range(n-1):
        curr_v_idx = n * row + col
        curr_f_idx = 2 * ((n-1) * row + col)
        F[curr_f_idx, 0] = curr_v_idx
        F[curr_f_idx, 1] = curr_v_idx + 1
        F[curr_f_idx, 2] = curr_v_idx + n + 1
        F[curr_f_idx + 1, 0] = curr_v_idx
        F[curr_f_idx + 1, 1] = curr_v_idx + n + 1
        F[curr_f_idx + 1, 2] = curr_v_idx + n

# even-odd decomposition
even_size = int(np.ceil((n-1) / 2))
odd_size = int(np.floor((n-1) / 2))
# E hori
E0 = np.zeros(((n-1)*n, 2), dtype=int)
for row in range(n):
    for col in range(n-1):
        curr_v_idx = n * row + col
        curr_e_idx = (n-1) * row + col
        E0[curr_e_idx, 0] = curr_v_idx
        E0[curr_e_idx, 1] = curr_v_idx + 1
E_init_d0 = np.linalg.norm(V[E0[:, 0]] - V[E0[:, 1]], ord=2, axis=1).reshape((-1, 1))
V_sum_w0 = V_w[E0[:, 0]] + V_w[E0[:, 1]]
V_sum_w0[V_sum_w0 == 0] = np.inf

# triangle vertical edge even
E2 = np.zeros((n * (n-1), 2), dtype=int)
for row in range(n-1):
    for col in range(n):
        curr_v_idx = n * row + col
        curr_e_idx = n * row + col
        E2[curr_e_idx, 0] = curr_v_idx
        E2[curr_e_idx, 1] = curr_v_idx + n
E_init_d2 = np.linalg.norm(V[E2[:, 0]] - V[E2[:, 1]], ord=2, axis=1).reshape((-1, 1))
V_sum_w2 = V_w[E2[:, 0]] + V_w[E2[:, 1]]
V_sum_w2[V_sum_w2 == 0] = np.inf

# triangle diagonal edge even
E3 = np.zeros(((n-1) **2, 2), dtype=int)
for row in range(n-1):
    for col in range(n-1):
        curr_v_idx = n * row + col
        curr_e_idx = (n-1) * row + col
        E3[curr_e_idx, 0] = curr_v_idx
        E3[curr_e_idx, 1] = curr_v_idx + n + 1
E_init_d3 = np.linalg.norm(V[E3[:, 0]] - V[E3[:, 1]], ord=2, axis=1).reshape((-1, 1))
V_sum_w3 = V_w[E3[:, 0]] + V_w[E3[:, 1]]
V_sum_w3[V_sum_w3 == 0] = np.inf
# triangle horizontal edge even
# E0 = np.zeros((n * even_size, 2), dtype=int)
# for row in range(n):
#     for col in range(even_size):
#         curr_v_idx = n * row + (2 * col)
#         curr_e_idx = even_size * row + col
#         E0[curr_e_idx, 0] = curr_v_idx
#         E0[curr_e_idx, 1] = curr_v_idx + 1
# E_init_d0 = np.linalg.norm(V[E0[:, 0]] - V[E0[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w0 = V_w[E0[:, 0]] + V_w[E0[:, 1]]
# V_sum_w0[V_sum_w0 == 0] = np.inf
# # triangle horizontal edge odd
# E1 = np.zeros((n * odd_size, 2), dtype=int)
# for row in range(n):
#     for col in range(odd_size):
#         curr_v_idx = n * row + (2 * col + 1)
#         curr_e_idx = odd_size * row + col
#         E1[curr_e_idx, 0] = curr_v_idx
#         E1[curr_e_idx, 1] = curr_v_idx + 1
# E_init_d1 = np.linalg.norm(V[E1[:, 0]] - V[E1[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w1 = V_w[E1[:, 0]] + V_w[E1[:, 1]]
# V_sum_w1[V_sum_w1 == 0] = np.inf
# triangle vertical edge even
# E2 = np.zeros((n * even_size, 2), dtype=int)
# for row in range(even_size):
#     for col in range(n):
#         curr_v_idx = n * (2 * row) + col
#         curr_e_idx = n * row + col
#         E2[curr_e_idx, 0] = curr_v_idx
#         E2[curr_e_idx, 1] = curr_v_idx + n
# E_init_d2 = np.linalg.norm(V[E2[:, 0]] - V[E2[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w2 = V_w[E2[:, 0]] + V_w[E2[:, 1]]
# V_sum_w2[V_sum_w2 == 0] = np.inf
# # triangle vertical edge odd
# E3 = np.zeros((n * odd_size, 2), dtype=int)
# for row in range(odd_size):
#     for col in range(n):
#         curr_v_idx = n * (2 * row + 1) + col
#         curr_e_idx = n * row + col
#         E3[curr_e_idx, 0] = curr_v_idx
#         E3[curr_e_idx, 1] = curr_v_idx + n
# E_init_d3 = np.linalg.norm(V[E3[:, 0]] - V[E3[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w3 = V_w[E3[:, 0]] + V_w[E3[:, 1]]
# V_sum_w3[V_sum_w3 == 0] = np.inf
# triangle diagonal edge even
# E4 = np.zeros(((n-1) * even_size, 2), dtype=int)
# for row in range(even_size):
#     for col in range(n-1):
#         curr_v_idx = (n-1) * (2 * row) + col
#         curr_e_idx = (n-1) * row + col
#         E4[curr_e_idx, 0] = curr_v_idx
#         E4[curr_e_idx, 1] = curr_v_idx + n + 1
# E_init_d4 = np.linalg.norm(V[E4[:, 0]] - V[E4[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w4 = V_w[E4[:, 0]] + V_w[E4[:, 1]]
# V_sum_w4[V_sum_w4 == 0] = np.inf
# # triangle diagonal edge odd
# E5 = np.zeros(((n-1) * odd_size, 2), dtype=int)
# for row in range(odd_size):
#     for col in range(n-1):
#         curr_v_idx = (n-1) * (2 * row + 1) + col
#         curr_e_idx = (n-1) * row + col
#         E5[curr_e_idx, 0] = curr_v_idx
#         E5[curr_e_idx, 1] = curr_v_idx + n + 1
# E_init_d5 = np.linalg.norm(V[E5[:, 0]] - V[E5[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w5 = V_w[E5[:, 0]] + V_w[E5[:, 1]]
# V_sum_w5[V_sum_w5 == 0] = np.inf
# bending diagonal edge even
# E6 = np.zeros(((n-1) * even_size, 2), dtype=int)
# for row in range(even_size):
#     for col in range(n-1):
#         curr_v_idx = (n-1) * (2 * row) + col
#         curr_e_idx = (n-1) * row + col
#         E6[curr_e_idx, 0] = curr_v_idx + 1
#         E6[curr_e_idx, 1] = curr_v_idx + n
# E_init_d6 = np.linalg.norm(V[E6[:, 0]] - V[E6[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w6 = V_w[E6[:, 0]] + V_w[E6[:, 1]]
# V_sum_w6[V_sum_w6 == 0] = np.inf
# # bending diagonal edge odd
# E7 = np.zeros(((n-1) * odd_size, 2), dtype=int)
# for row in range(odd_size):
#     for col in range(n-1):
#         curr_v_idx = (n-1) * (2 * row + 1) + col
#         curr_e_idx = (n-1) * row + col
#         E7[curr_e_idx, 0] = curr_v_idx + 1
#         E7[curr_e_idx, 1] = curr_v_idx + n
# E_init_d7 = np.linalg.norm(V[E7[:, 0]] - V[E7[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w7 = V_w[E7[:, 0]] + V_w[E7[:, 1]]
# V_sum_w7[V_sum_w7 == 0] = np.inf
# # bending horizontal diagonal edges even
# E8 = np.zeros(((n-2) * even_size, 2), dtype=int)
# for row in range(even_size):
#     for col in range(n-2):
#         curr_v_idx = n * (2 * row) + col
#         curr_e_idx = (n-2) * row + col
#         E8[curr_e_idx, 0] = curr_v_idx
#         E8[curr_e_idx, 1] = curr_v_idx + n + 2
# E_init_d8 = np.linalg.norm(V[E8[:, 0]] - V[E8[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w8 = V_w[E8[:, 0]] + V_w[E8[:, 1]]
# V_sum_w8[V_sum_w8 == 0] = np.inf
# # bending horizontal diagonal edges odd
# E9 = np.zeros(((n-2) * odd_size, 2), dtype=int)
# for row in range(odd_size):
#     for col in range(n-2):
#         curr_v_idx = n * (2 * row + 1) + col
#         curr_e_idx = (n-2) * row + col
#         E9[curr_e_idx, 0] = curr_v_idx
#         E9[curr_e_idx, 1] = curr_v_idx + n + 2
# E_init_d9 = np.linalg.norm(V[E9[:, 0]] - V[E9[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w9 = V_w[E9[:, 0]] + V_w[E9[:, 1]]
# V_sum_w9[V_sum_w9 == 0] = np.inf
# # bending vertical diagonal edges even
# E10 = np.zeros(((n-2) * even_size, 2), dtype=int)
# for row in range(n-2):
#     for col in range(even_size):
#         curr_v_idx = n * row + (2 * col)
#         curr_e_idx = even_size * row + col
#         E10[curr_e_idx, 0] = curr_v_idx
#         E10[curr_e_idx, 1] = curr_v_idx + 2 * n + 1
# E_init_d10 = np.linalg.norm(V[E10[:, 0]] - V[E10[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w10 = V_w[E10[:, 0]] + V_w[E10[:, 1]]
# V_sum_w10[V_sum_w10 == 0] = np.inf
# # bending vertical diagonal edges odd
# E11 = np.zeros(((n-2) * odd_size, 2), dtype=int)
# for row in range(n-2):
#     for col in range(odd_size):
#         curr_v_idx = n * row + (2 * col + 1)
#         curr_e_idx = odd_size * row + col
#         E11[curr_e_idx, 0] = curr_v_idx
#         E11[curr_e_idx, 1] = curr_v_idx + 2 * n + 1
# E_init_d11 = np.linalg.norm(V[E11[:, 0]] - V[E11[:, 1]], ord=2, axis=1).reshape((-1, 1))
# V_sum_w11 = V_w[E11[:, 0]] + V_w[E11[:, 1]]
# V_sum_w11[V_sum_w11 == 0] = np.inf

# solver parameters
dt = 0.01
iteration = 10
# material parameters
stretch_stiffness = 5e4
stretch_compliance = 1 / (stretch_stiffness * dt**2)
# project an independent set of constraints
def project_E(E, E_init_d, V_sum_w, L):
    # position difference vectors
    N = V_predict[E[:, 0]] - V_predict[E[:, 1]]
    # distance
    D = np.linalg.norm(N, ord=2,axis=1).reshape((-1, 1))
    # constarint values
    C = D - E_init_d
    # normalized difference vectors
    N /= D
    # delta lagrange
    L_delta = (-C - stretch_compliance * L) / (V_sum_w + stretch_compliance)
    # update lagrange
    L += L_delta
    # update for 0 vertex in constraint
    V_predict[E[:, 0]] += V_w[E[:, 0]] * L_delta * N
    # update for 1 vertex in constraint
    V_predict[E[:, 1]] -= V_w[E[:, 1]] * L_delta * N

# o3d visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().mesh_show_back_face = True
# create mesh and wireframe
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(V)
mesh.triangles = o3d.utility.Vector3iVector(F)
wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
vis.add_geometry(mesh)
vis.add_geometry(wireframe)
# create coordinate frame
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
    origin=[0, 0, -1]
)
vis.add_geometry(coordinate)

# main loop
for _ in range(10000):
    # update velocity
    V_velocity += dt * V_force / V_mass
    # update predict
    V_predict += dt * V_velocity

    # set lagrange to 0
    L0 = np.zeros_like(E_init_d0)
    # L1 = np.zeros_like(E_init_d1)
    L2 = np.zeros_like(E_init_d2)
    L3 = np.zeros_like(E_init_d3)
    # L4 = np.zeros_like(E_init_d4)
    # L5 = np.zeros_like(E_init_d5)
    # L6 = np.zeros_like(E_init_d6)
    # L7 = np.zeros_like(E_init_d7)
    # L8 = np.zeros_like(E_init_d8)
    # L9 = np.zeros_like(E_init_d9)
    # L10 = np.zeros_like(E_init_d10)
    # L11 = np.zeros_like(E_init_d11)
    # solver iteration
    for _ in range(iteration):
        project_E(E0, E_init_d0, V_sum_w0, L0)
        # project_E(E1, E_init_d1, V_sum_w1, L1)
        project_E(E2, E_init_d2, V_sum_w2, L2)
        project_E(E3, E_init_d3, V_sum_w3, L3)
        # project_E(E4, E_init_d4, V_sum_w4, L4)
        # project_E(E5, E_init_d5, V_sum_w5, L5)
        # project_E(E6, E_init_d6, V_sum_w6, L6)
        # project_E(E7, E_init_d7, V_sum_w7, L7)
        # project_E(E8, E_init_d8, V_sum_w8, L8)
        # project_E(E9, E_init_d9, V_sum_w9, L9)
        # project_E(E10, E_init_d10, V_sum_w10, L10)
        # project_E(E11, E_init_d11, V_sum_w11, L11)

    # update actual velocity and position
    V_velocity = (V_predict - V) / dt
    V = np.array(V_predict)

    # update visualizer
    mesh.vertices = o3d.utility.Vector3dVector(V)
    wireframe.points = mesh.vertices
    vis.update_geometry(mesh)
    vis.update_geometry(wireframe)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
