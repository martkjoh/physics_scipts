import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

f1 = lambda x : (1 + np.sqrt(1+0j + x**2)) / 2
f2 = lambda x : (1 - np.sqrt(1+0j + x**2)) / 2

n = 400
a = 1.5

n = 101
r = np.linspace(0, a, 20)
theta = np.linspace(0, 2*np.pi, n)
r, theta = np.meshgrid(r, theta)
x, y = r*np.cos(theta), r*np.sin(theta)
x, y = x.flatten(), y.flatten()

# x = np.linspace(-a, a, n)
# y = np.linspace(-a, a, n)
# x, y = np.meshgrid(x, y)

z = x + 1j*y
w1 = f1(z)
w2 = f2(z)


def norm(x):
    a, b = np.min(x), np.max(x)
    print(a, b)
    return (x - b)/(a - b)

def map_colors(p3dc, func, cmap='viridis'): 
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    Source: https://stackoverflow.com/questions/63298864/plot-trisurface-with-custom-color-array
    """
    
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s],y[s],z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T
    
    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


a1 = ax.plot_trisurf(x, y, w1.real)
a2 = ax.plot_trisurf(x, y, w2.real)
 
# a1.set_fc()
map_colors(a1, lambda x, y, z: f1(x+1j*y).imag)
map_colors(a2, lambda x, y, z: f2(x+1j*y).imag)


plt.axis('off')

plt.grid(visible=None)


plt.show()
ax.view_init(10, 40)

save_opt = dict(
    bbox_inches='tight',
    pad_inches = -0.4,
    transparent = True, 
    dpi=300
)

# plt.savefig('fig/reimann.pdf', **save_opt)
