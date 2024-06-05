from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
from matplotlib import colors as c
from matplotlib import cm
from matplotlib import ticker


plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("font", family="serif", size=16)

# Blue, green, puple surface, red, black, blue line
zo = iter([1, 4, 5, -1, 2])
ang = (-78, 14)
# ang = 

# zo = iter([4, 6, 5, 7, 5, 2])
# ang = (85, 10)

fig, ax = plt.subplots(subplot_kw={"projection": "3d", 'computed_zorder' : False}, figsize=(10,15))


N = 200
th = 1
k = sqrt(th)

def spin(ax):
    x0 = np.linspace(-k, k, N)
    r0 = np.linspace(-th, 0.001, N)
    x, r = np.meshgrid(x0, r0)

    a = np.sqrt(x**2**2 - (r + 2*x**2)**2 + 0j).real
    a[0:2]=0
    edge = np.zeros_like(a, dtype=bool)
    for i in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (0, -2), (2, 0), (-2, 0)):
        edge = edge | np.roll(a!=0, i, (0, 1))

    a[edge==0]=None
    a[0:2]=None

    ax.plot_surface(x, r, a,cmap=cm.Blues, antialiased=False, rstride=2, cstride=1, vmin=-.4, lw=1, zorder=next(zo))

def stab(ax):
    z = np.linspace(0, th/2 + 0.2, N)
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    z, x = np.meshgrid(z, x)
    y = - 2 * x**2

    m = z<x**2
    x[m]=None
    y[m]=None
    z[m]=None
    ax.plot_surface(x, y, z, cmap=cm.viridis, alpha=1, zorder=next(zo),  vmin=-1, vmax=2)

def exc(ax):
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    y = np.linspace(0, -th, N)
    x, y = np.meshgrid(x, y)
    z = x**2

    ax.plot_surface(x, y, z, cmap=cm.Purples, alpha=1, zorder=next(zo), vmin=-.5, vmax=2)


spin(ax)
stab(ax)
# exc(ax)


rgba_to_hex = lambda rgba : '#'+''.join([f'{int(v*255):02x}' for v in rgba])
color = rgba_to_hex(cm.viridis(.25))


y = k / np.sqrt( 2)

t = np.linspace(y, -y, 1000)
ax.plot(t, -2*t**2, t**2, "r--", alpha=.8, lw=1.2, zorder=next(zo))


x1 =  np.sqrt(th / 3)
X1 = np.linspace(-x1, x1, N)
X2 = np.linspace(-sqrt(th), -x1, N)
x = np.concatenate([X2, X1, -X2[::-1]])
r1 = - x**2
r2 = np.concatenate([-th * np.ones_like(X2), -3*X1**2, -th * np.ones_like(X2)])

ax.add_collection3d(plt.fill_between(x, r1, r2, alpha=0.3, color=color),0)
ax.add_collection3d(plt.fill_between(X1, - 3 * X1**2, -th, color=color, alpha=0.6, label="spinodal", hatch='\\\\\\', edgecolor='#00000000'),0)

x = np.linspace(-k,k, N)
oh = np.zeros_like(x)
ax.plot(x, -x**2, oh, "k-", label='$\\xi_+ = 0$', zorder=next(zo))

x = np.linspace(-x1,x1, N)
oh = np.zeros_like(x)
ax.plot(x, -3*x**2, oh, "b-.", label='$\\xi_- = 0$', zorder=next(zo))


### Fixes

ax.set_xlabel("\n$\\sqrt{u}\\bar\\varphi$", linespacing=2)
ax.set_ylabel("$\\theta$")
ax.set_zlabel("$\\alpha$")
ax.zaxis.set_tick_params(labelsize=10)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


# ax.set_zticks([0, 0.4, 0.8, 1.2])
# ax.set_yticks([-2, -1.5, -1., -0.5, 0.])

ax.set_box_aspect(aspect = (2,2,.5))

plt.subplots_adjust(top=1, bottom=0, right=0.8, left=0, hspace=0, wspace=1)
save_opt = dict(
    bbox_inches='tight',
    pad_inches = 0, 
    transparent=True, 
    dpi=300
)

ax.azim=ang[0]
ax.elev=ang[1]

# fig.savefig("done/fig/surf3.svg", **save_opt)
plt.show()
