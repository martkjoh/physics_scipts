"""Helper module providing a readable interactive isosurface UI.

Usage in notebook:
from NRCH.interactive_isosurface import create_isosurface_ui
create_isosurface_ui(bu, ba, bb, simulate=simulate, get_init=get_init)
"""

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import ipywidgets as widgets
from IPython.display import display


def _compute_grid(u_res, a_res, b_res, ar, br, ur):
    if br is None:
        br = ar * 2
    u_vals = np.linspace(0.01, ur, u_res)
    a_vals = np.linspace(-ar, br * 0.8, a_res)
    b_vals = np.linspace(0, br, b_res)
    Ug, Ag, Bg = np.meshgrid(u_vals, a_vals, b_vals, indexing='ij')
    return u_vals, a_vals, b_vals, Ug, Ag, Bg


def _compute_isosurface(V, u_vals, a_vals, b_vals):
    du = u_vals[1] - u_vals[0] if len(u_vals) > 1 else 1.0
    da = a_vals[1] - a_vals[0] if len(a_vals) > 1 else 1.0
    db = b_vals[1] - b_vals[0] if len(b_vals) > 1 else 1.0
    verts, faces, normals, values = measure.marching_cubes(V, level=0.0, spacing=(du, da, db))
    verts_u = u_vals[0] + verts[:, 0]
    verts_a = a_vals[0] + verts[:, 1]
    verts_b = b_vals[0] + verts[:, 2]
    mesh_faces = [list(zip(verts_a[f], verts_b[f], verts_u[f])) for f in faces]
    return mesh_faces


def _plot_mesh_and_trajectories(mesh_faces, u_vals, a_vals, b_vals, ntraj, dt, nsteps, ar, br, ur, bu, ba, bb, simulate=None, get_init=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    poly = Poly3DCollection(mesh_faces, alpha=0.8, linewidths=0)
    poly.set_facecolor(cm.viridis(0.6))
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    ax.set_xlim(a_vals.min(), a_vals.max())
    ax.set_ylim(b_vals.min(), b_vals.max())
    ax.set_zlim(u_vals.min(), u_vals.max())

    ax.set_xlabel("$\\alpha_1$")
    ax.set_ylabel("b")
    ax.set_zlabel("u")

    if get_init is not None and simulate is not None:
        try:
            initials = get_init(ntraj, ar, br, ur)
        except TypeError:
            try:
                initials = get_init(ntraj)
            except Exception:
                initials = None
        if initials is not None:
            for u0, a0, b0 in initials:
                v = lambda u, a, b: -np.array([bu(u, a, b, 0.0), ba(u, a, b, 0.0), bb(u, a, b, 0.0)])
                traj = simulate(v, u0, a0, b0, dt=dt, nsteps=nsteps)
                ax.plot(traj[:, 1], traj[:, 2], traj[:, 0], color='k', linewidth=0.8, alpha=0.9)
                ax.scatter([traj[0, 1]], [traj[0, 2]], [traj[0, 0]], color='red', s=20)
                ax.scatter([traj[-1, 1]], [traj[-1, 2]], [traj[-1, 0]], color='blue', s=20)

    plt.show()


def make_interactive_isosurface(bu, ba, bb, simulate=None, get_init=None, *, u_res=40, a_res=60, b_res=60, ar=0.7, br=None, ur=0.2, ntraj=8, dt=0.01, nsteps=1000):
    u_vals, a_vals, b_vals, Ug, Ag, Bg = _compute_grid(u_res, a_res, b_res, ar, br, ur)
    V = bu(Ug, Ag, Bg, 0.0)
    mesh_faces = _compute_isosurface(V, u_vals, a_vals, b_vals)
    _plot_mesh_and_trajectories(mesh_faces, u_vals, a_vals, b_vals, ntraj, dt, nsteps, ar, br, ur, bu, ba, bb, simulate=simulate, get_init=get_init)


def create_isosurface_ui(bu, ba, bb, simulate=None, get_init=None):
    u_res_w = widgets.IntSlider(value=40, min=10, max=120, step=5, description='u_res')
    a_res_w = widgets.IntSlider(value=60, min=10, max=120, step=5, description='a_res')
    b_res_w = widgets.IntSlider(value=60, min=10, max=120, step=5, description='b_res')

    ar_w = widgets.FloatSlider(value=0.7, min=0.01, max=2.0, step=0.01, description='ar')
    br_w = widgets.FloatSlider(value=1.4, min=0.01, max=4.0, step=0.01, description='br')
    ur_w = widgets.FloatSlider(value=0.2, min=0.01, max=1.0, step=0.01, description='ur')

    ntraj_w = widgets.IntSlider(value=8, min=0, max=30, step=1, description='ntraj')
    dt_w = widgets.FloatLogSlider(value=0.01, base=10, min=-3, max=-1, step=0.1, description='dt')
    nsteps_w = widgets.IntSlider(value=1000, min=100, max=3000, step=100, description='nsteps')

    ui = widgets.VBox([
        widgets.HBox([u_res_w, a_res_w, b_res_w]),
        widgets.HBox([ar_w, br_w, ur_w]),
        widgets.HBox([ntraj_w, dt_w, nsteps_w])
    ])
    out = widgets.Output()

    def _update(change=None):
        with out:
            out.clear_output(wait=True)
            make_interactive_isosurface(bu, ba, bb, simulate=simulate, get_init=get_init,
                                         u_res=u_res_w.value, a_res=a_res_w.value, b_res=b_res_w.value,
                                         ar=ar_w.value, br=br_w.value, ur=ur_w.value,
                                         ntraj=ntraj_w.value, dt=float(dt_w.value), nsteps=nsteps_w.value)

    for w in [u_res_w, a_res_w, b_res_w, ar_w, br_w, ur_w, ntraj_w, dt_w, nsteps_w]:
        w.observe(_update, names='value')

    display(ui, out)
    _update()

    return ui, out
