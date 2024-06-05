from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi, sqrt


from loadfiles import *

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)



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
    ax.plot(-np.abs(phibar), alpha, 'ro')

    ax.set_ylabel("$\\alpha/|r|$")
    ax.set_xlabel("$\\sqrt{u} \\bar \\varphi/|r|$")

    ax.set_ylim(0, .8)
    ax.set_xlim(-1, .1)


def plot_error(phit, dt):
    fig, ax = plt.subplots()
    pt = np.einsum('tix->ti', phit)
    dpt = (pt[1:] - pt[:-1])/dt
    frames = len(phit)
    t = np.linspace(0, frames*dt, frames-1)
    ax.plot(t, dpt[:,0], label="$\\frac{\\mathrm{d} \\bar \\varphi_1}{\\mathrm{d} t}$")
    ax.plot(t, dpt[:,1], label="$\\frac{\\mathrm{d} \\bar \\varphi_2}{\\mathrm{d} t}$")
    ax.legend()


def plot_sol2(ax, param):
    u, a, b, phibar, N, dt = param
    tt = np.linspace(0, L, 1000)
    ax.plot(tt, (1 + phibar)*np.cos(2*tt/L*2*np.pi) + phibar,":r",label="$A\\cos2\\phi + c$")
    ax.plot(tt, 2*np.sqrt(-phibar-phibar**2)*np.cos(tt/L*2*np.pi),":k",label="$B\\cos^2\phi$")

def plot_sol1(ax, param):
    u, a, b, phibar, N, dt = param
    A = np.sqrt((u - (2 * np.pi / L * 2)**2)/u)
    ax.plot([0, L], [A, A], 'g--', label="$\\sqrt{(-r - (4\\pi/L)^2)/u}$")
    A = np.sqrt((u - (2 * np.pi / L )**2)/u)
    ax.plot([0, L], [A, A], 'b--', label="$\\sqrt{(-r - (2\\pi/L)^2)/u}$")


def make_anim(folder, filename):
    filename = filename[:-4]

    phit, xit, param = load_file(folder, filename)
    u, a, b, phibar, N, dt = param
    L = 10
    dx = L / N
    x = np.linspace(0, L, N)
    D2 = lambda J : ( np.roll(J, 1, axis=-1) + np.roll(J, -1, axis=-1) - 2 * J ) / (dx)**2 
    D = lambda J : (np.roll(J, 1, axis=-1) - np.roll(J, -1, axis=-1) ) / (2 * dx)
    plot_error(phit, dt)

    fig = plt.figure(layout="constrained", figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[1, 0]) 

    ax = [ax1, ax2, ax3, ax4]
    fig.suptitle(", ".join(filename_from_param(param).split('_')))

    # plot_sol1(ax[0], param)
    # plot_sol2(ax[0], param)

    l6, = ax[3].plot([], [], 'r-')
    axx = ax[3].twinx()
    l7, = axx.plot([], [], 'k-')
    l8, = ax[3].plot([], [], 'g--') 
    l9, = ax[3].plot([], [], 'r-.')

    jj = 10
    ax[3].set_ylim(-jj, jj)
    ax[3].set_xlim(0, L)
    axx.set_ylim(-1.5, 1.5)

    l1, = ax[0].plot([], [], 'r.-', label='$\\varphi_1$')
    l2, = ax[0].plot([], [], 'k.-', label='$\\varphi_2$')
    ax[0].plot([0, L], [phibar, phibar], 'r--')
    ax[0].plot([0, L], [0, 0], 'k--')

    ax[0].set_xlim(0, L) 
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].legend(loc=1)
    l5 = ax[0].text(L/10, 1, 'progress:')
    

    t = np.linspace(0, 2*pi)
    prange = 1.2
    m1, = ax[1].plot([], [], 'r--.')
    ax[1].plot(0, phibar, 'ro')
    ax[1].plot(np.cos(t), np.sin(t), 'k--') 
    ax[1].set_xlim(-prange, prange)
    ax[1].set_ylim(-prange, prange)

    add_phase(ax[2], phibar, a/u)

    p2 =  np.sum(phit[0]**2, axis=0)
    F0 = np.zeros(len(phit))
    F = u * (- p2 / 2 + p2**2 / 4 ) - p2 * D2(p2) / 2
    F0[0] = np.sum(F) *dx
    frames = len(phit)

    n = 10
    def animate(m, F0):
        m = m*n
        n2 = frames//10
        txt = 'progress:' + (m+1)//n2*'|'
        l5.set_text(txt)

        p = phit[m]
        l1.set_data(x, p[0])
        l2.set_data(x, p[1])
        p2 = np.sqrt( p[:, 0]**2 + p[:, 1]**2 ) 

        m1.set_data([*p[1], p[1, 0]], [*p[0], p[0, 0]])

        p2 =  np.sum(p**2, axis=0)
        dF = u * (-1 + p2 ) * p - D2(p)
        ap = a * np.array([p[1], -p[0]])
        mu = dF #+ ap

        fdot = np.sum(D2(mu) * dF, axis=0)
        Fdot = np.sum(fdot) * dx
        if m>0:
            F0[m] = F0[m-1] + Fdot * dt

        F1 = u * (- p2 / 2 + p2**2 / 4 ) - p2 * D2(p2) / 2
        F1 = np.sum(F1) * dx

        # l6.set_data(x, fdot)
        # l7.set_data([0, L], F0[m] * np.ones(2))
        # l8.set_data([0, L], F1 * np.ones(2))
        # l9.set_data([0, L], Fdot * np.ones(2))

        l6.set_data(x, dF[0])
        l7.set_data(x, p[0]) 
 

    anim = animation.FuncAnimation(fig, animate,  interval=50, frames=frames//n, repeat=True, fargs=[F0,])
    plt.tight_layout()
    plt.show()
    # anim.save(folder_vid+filename+".mp4", fps=30)


folder = "data/sym/"
folder_vid = "vid/"

fnames = get_all_filenames_in_folder(folder)

[make_anim(folder, fname) for fname in fnames]


# from multiprocessing import Pool
# with Pool(4) as pool:
 #     pool.map(make_anim, fnames)
