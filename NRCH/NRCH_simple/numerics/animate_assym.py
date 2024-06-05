from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi, sqrt

import os

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

param_names = ["u, -r", "a", "b", "phi1", "phi2", "N", "dt"]
param_title = ["u, -r", "\\alpha", "\\beta", "{\\varphi_1}", "\\varphi_2", "N", "\\Delta t"]

folder = "data/assym/"
folder_vid = "numerics/vid/assym/"

L = 10.
frames = 1000


def filename_from_param(param):
    return ''.join(param_names[i] + '=' + str(param[i]) + '_' for i in range(len(param_names)))[:-1]

def title_from_param(param):
    return "$" + ''.join(param_title[i] + '=' + "%.3f" % param[i] + '\\quad' for i in range(len(param_names))) + "$"

def get_all_filenames_in_folder(folder_path):
    """
    Returns a list of all filenames of all files in a folder.
    """
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames

def param_dict_from_filename(s):
    """
    Extracts parameters from a string in the format 'value1=xxx_value2=yyy...' and
    returns a dictionary in the format {'value1': xxx, 'value2': yyy, ...}.
    """
    params = {}
    for param in s.split('_'):
        key, value = param.split('=')
        params[key] = float(value)
    return params

def param_from_dict(param_dict):
    return [param_dict[key] for key in param_names]

def param_from_filename(filename):
    return param_from_dict(param_dict_from_filename(filename))

def load_file(filename):
    file = folder+filename+'.txt'
    param = param_from_filename(filename)

    u, a, b, phi1, phi2, N, dt = param
    N = int(N)
    param = u, a, b, phi1, phi2, N, dt

    phit = np.loadtxt(file)
    frames = len(phit)
    phit = phit.reshape((frames, 2, N))
    x = np.linspace(0, L, N)
    return x, phit, param

def add_phase(ax, param):
    u, a, b, phi1, phi2, N, dt = param
    alpha = a / u
    L = 1.5

    x = lambda u, v : sqrt(u**2 + v**2)
    y = lambda u, v : sqrt(np.abs(u**2 - v**2))

    g1 = lambda x, y, a :  ((x**2 - 1) - sqrt(y**4 - a**2 + 0j)).real
    g2 = lambda x, y, a :  ((x**2 - 1) + sqrt(y**4 - a**2 + 0j)).real

    f1 = lambda u, v, a : g1(x(u, v), y(u, v), a)
    f2 = lambda u, v, a : g2(x(u, v), y(u, v), a) 

    f = [f1, f2]

    u, v = np.linspace(-L, L, N), np.linspace(-L, L, N) 
    u, v = np.meshgrid(u, v)

    for i in range(2):
        color = 'blue' if i == 1 else 'black'
        ls = '--' if i==1 else "-"
        ax.contour(u, v, f[i](u, v, alpha), levels=[0], colors=color, linestyles=ls)

    ax.plot(phi2, phi1, 'ro')

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)

def plot_error(phit, param):
    u, a, b, phi1, phi2, N, dt = param
    fig, ax = plt.subplots()
    pt = np.einsum('tix->ti', phit)
    dpt = (pt[1:] - pt[:-1])/dt
    frames = len(phit)
    t = np.linspace(0, frames*dt, frames-1)
    ax.plot(t, dpt[:,0], label="$\\frac{\\mathrm{d} \\bar \\varphi_1}{\\mathrm{d} t}$")
    ax.plot(t, dpt[:,1], label="$\\frac{\\mathrm{d} \\bar \\varphi_2}{\\mathrm{d} t}$")
    ax.legend()

def make_anim(filename):
    filename = filename[:-4]

    x, phit, param = load_file(filename)
    u, a, b, phi1, phi2, N, dt = param
    # plot_error(phit, param)

    fig = plt.figure(layout="constrained", figsize=(12, 6.5))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[:, 1:3])
    ax3 = fig.add_subplot(gs[1, 0]) 
    ax = [ax1, ax2, ax3]
    fig.suptitle(title_from_param(param))

    l1, = ax[0].plot([], [], 'r-', label='$\\varphi_1$')
    l2, = ax[0].plot([], [], 'k-', label='$\\varphi_2$')
    ax[0].plot([0, L], [phi1, phi1], 'r--')
    ax[0].plot([0, L], [phi2, phi2], 'k--')

    prange = 1.5

    ax[0].set_xlim(0, L) 
    ax[0].set_ylim(-prange, prange)
    ax[0].legend(loc=1)
    l5 = ax[0].text(L/10, 1, 'progress:')

    t = np.linspace(0, 2*pi)
    m1, = ax[1].plot([], [], 'r-..')
    ax[1].plot(phi2, phi1, 'ro')
    ax[1].plot(np.cos(t), np.sin(t), 'k--') 
    ax[1].set_xlim(-prange, prange)
    ax[1].set_ylim(-prange, prange)

    add_phase(ax[2], param)
    
    n = 4

    def animate(m):
        m = m*n
        n2 = frames//10
        txt = 'progress:' + (m+1)//n2*'|'
        l5.set_text(txt)

        p = phit[m]
        l1.set_data(x, p[0])
        l2.set_data(x, p[1])
        p2 = np.sqrt( p[:, 0]**2 + p[:, 1]**2 ) 

        m1.set_data([*p[1], p[1, 0]], [*p[0], p[0, 0]])

        return l1, l2, m1

    anim = animation.FuncAnimation(fig, animate,  interval=10, frames=frames//n)
    plt.show()
    # anim.save(folder_vid+filename+".mp4", fps=15)


fnames = get_all_filenames_in_folder(folder)

# make_anim(fnames[1])

from multiprocessing import Pool
with Pool(12) as pool:
    pool.map(make_anim, fnames)
    plt.close()
