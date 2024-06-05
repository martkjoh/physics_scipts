import numpy as np
from numpy import sqrt
import plotly.offline as ply
import plotly.graph_objs as go


th = 2
k = sqrt(th)

N = 1000
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



fig = go.Figure()
fig.add_surface(
    x=x, y=r, z=a,
    surfacecolor=a,
    showscale=False, 
    colorscale="Blues",
    contours = {
        # "x": {"show": True, "start": -k, "end": k, "size": 0.1},
        "y": {"show": True, "start": -th, "end": 0, "size": 0.1},
        "z": {"show": True, "start":  0, "end": k, "size": 0.1}
    }
)


z = np.linspace(0, th/2 + 0.2, N)
x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
z, x = np.meshgrid(z, x)
y = - 2 * x**2

m = z<x**2
x[m]=None
y[m]=None
z[m]=None

fig.add_surface(
    x=x, y=y, z=z,
    showscale=False, 
    surfacecolor=y,
    colorscale='greens',
    contours = {
        # "x": {"show": True, "start": -k, "end": k, "size": 0.1},
        "y": {"show": True, "start": -th, "end": 0, "size": 0.1},
        "z": {"show": True, "start":  0, "end": k, "size": 0.1}
    }
)


N = 1000
x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
y = np.linspace(0, -th, N)
x, y = np.meshgrid(x, y)
z = x**2

fig.add_surface(
    x=x, y=y, z=z,
    showscale=False, 
    surfacecolor=z,
    colorscale='purples',
    contours = {
        # "x": {"show": True, "start": -k, "end": k, "size": 0.1},
        "y": {"show": True, "start": -th, "end": 0, "size": 0.1},
        "z": {"show": True, "start":  0, "end": k, "size": 0.1}
    }
)




fig.update_layout(
    width=2000, 
    height=1000,
    font = {"family" : "Droid Serif"},
    scene=dict(
        # camera=dict(eye=dict(x=2, y=3, z=1)),
        aspectratio={
            'x' : 3, 'y' : 2, 'z' : 1.
        },
        aspectmode="manual",
        xaxis_title="$√u𝜑$",
        yaxis_title="$r$",
        zaxis_title="$𝛼$",
    )
)


fig.show()

