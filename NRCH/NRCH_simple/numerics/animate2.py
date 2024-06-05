from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi

import os

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

param_names = ["u, -r", "phibar", "a", "b"]

folder = "data/"
folder_vid = "numerics/sym/"

# folder = "data/sol2/"
# folder_vid = "numerics/vid/sol2/"

folder = "data/sol1/"
folder_vid = "numerics/vid/sol1/"


N = 200
L = 10.
dx = L/N
dt = .1 * (dx)**4
frames = 1000


def filename_from_param(param):
    return ''.join(param_names[i] + '=' + str(param[i]) + '_' for i in range(len(param_names)))[:-1]

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
    u, phibar, a, b  = param
    phit = np.loadtxt(file)
    frames = len(phit)
    phit = phit.reshape((frames, 2, N))
    x = np.linspace(0, L, N)
    return x, phit, param

def add_phase(ax, phibar, alpha):
    from numpy import pi, sin, cos
    from matplotlib import cm, ticker, colors as c

    rgba_to_hex = lambda rgba : '#'+''.join([f'{int(v*255):02x}' for v in rgba])
    color = rgba_to_hex(cm.viridis(.25))

    x0 = 1/np.sqrt(3)
    x1 = 1/np.sqrt(2)
    xx1 = np.linspace(x0, x1, 1000)
    xx2 = np.linspace(x1, 1, 1000)
    xx = np.linspace(x0, 1, 1000)
    xx3 = np.linspace(-x1, -x0, 1000)
    xx4 = np.linspace(-x0, x0, 1000)
    x = np.linspace(-1, 1)

    a = lambda x : np.sqrt(-3*x**4 + 4*x**2 - 1)

    ax.plot(-xx1, a(xx1), 'b-.')
    ax.plot(-xx2, a(xx2), 'k-')
    ax.plot([-x1, -x1], [a(x1), 1], 'g-')

    ax.fill_between(-xx, a(xx),np.zeros_like(xx), color=color, alpha=0.3, linewidth=0.0)
    ax.fill_between(xx3, a(xx3), 1, color=color, alpha=0.6, hatch='///', linewidth=0.0, edgecolor='#00000000')
    ax.fill_between(-xx3, a(xx3), 1, color=color, alpha=0.6, hatch='///', linewidth=0.0, edgecolor='#00000000')
    ax.fill_between(xx4, 0, 1, color=color, alpha=0.6, hatch='///', linewidth=0.0, edgecolor='#00000000')
    
    ax.plot(x, x**2, '--', color='purple', label='EL')
    ax.plot(phibar, alpha, 'ro')

    ax.set_ylabel("$\\alpha/|r|$")
    ax.set_xlabel("$\\sqrt{u} \\bar \\varphi/|r|$")

    ax.set_ylim(0, .8)
    ax.set_xlim(-1, .1)


def plot_error(phit):
    fig, ax = plt.subplots()
    pt = np.einsum('tix->ti', phit)
    dpt = (pt[1:] - pt[:-1])/dt
    frames = len(phit)
    t = np.linspace(0, frames*dt, frames-1)
    ax.plot(t, dpt[:,0], label="$\\frac{\\mathrm{d} \\bar \\varphi_1}{\\mathrm{d} t}$")
    ax.plot(t, dpt[:,1], label="$\\frac{\\mathrm{d} \\bar \\varphi_2}{\\mathrm{d} t}$")
    ax.legend()


def plot_sol2(ax, param):
    u, phibar, a, b = param

    tt = np.linspace(0, L, 1000)
    ax.plot(
        tt, 
        (1 + phibar)*np.cos(2*tt/L*2*np.pi) + phibar, 
        ":r",
        # label="$(1 + \\bar \\varphi) \\cos 2\\varphi + \\bar\\varphi$",
        label="$A\\cos2\\phi + c$"
        )
    ax.plot(
        tt, 
        2*np.sqrt(-phibar-phibar**2)*np.cos(tt/L*2*np.pi),
        ":k",
        # label="$2 \\sqrt{ \\bar\\varphi(1 - \\bar\\varphi) } \\cos^2\\varphi$",
        label="$B\\cos^2\phi$"
        )


def make_anim(filename):
    filename = filename[:-4]

    x, phit, param = load_file(filename)
    u, phibar, a, b = param
    # plot_error(phit)

    fig = plt.figure(layout="constrained", figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[1, 0]) 
    ax = [ax1, ax2, ax3]
    fig.suptitle(", ".join(filename_from_param(param).split('_')))

    # plot_sol2(ax[0], param)

    l1, = ax[0].plot([], [], 'r-', label='$\\varphi_1$')
    l2, = ax[0].plot([], [], 'k-', label='$\\varphi_2$')
    ax[0].plot([0, L], [phibar, phibar], 'r--')
    ax[0].plot([0, L], [0, 0], 'k--')

    A = np.sqrt((u - (2 * np.pi / L * 2)**2)/u)
    ax[0].plot([0, L], [A, A], 'g--', label="$\\sqrt{(-r - (4\\pi/L)^2)/2}$")
    A = np.sqrt((u - (2 * np.pi / L )**2)/u)
    ax[0].plot([0, L], [A, A], 'b--', label="$\\sqrt{(-r - (2\\pi/L)^2)/2}$")


    ax[0].set_xlim(0, L) 
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].legend(loc=1)
    l5 = ax[0].text(L/10, 1, 'progress:')
    

    t = np.linspace(0, 2*pi)
    prange = 1.2
    m1, = ax[1].plot([], [], 'r-..')
    ax[1].plot(0, phibar, 'ro')
    ax[1].plot(np.cos(t), np.sin(t), 'k--') 
    ax[1].set_xlim(-prange, prange)
    ax[1].set_ylim(-prange, prange)


    add_phase(ax[2], phibar, a/u)
    
    n = 1

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
    # plt.show()
    anim.save(folder_vid+filename+".mp4", fps=30)
    # FFwriter = animation.FFMpegWriter(fps=30) 
    # anim.save("done/fig/vid/"+filename+".mp4", writer=FFwriter)


fnames = get_all_filenames_in_folder(folder)

# make_anim(fnames[0])

from multiprocessing import Pool
with Pool(4) as pool:
    pool.map(make_anim, fnames)

# [make_anim(fname) for fname in fnames]
