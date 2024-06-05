import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

import plotly.offline as ply
import plotly.graph_objs as go


plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)


xf1 = lambda a, r: np.sqrt(1/4*(3*r + np.sqrt(r**2 - 8*a**2) + np.sqrt(8*a**2 + 2*r*(r - 2*np.sqrt(1 - 8*a**2) ))))
xf2 = lambda a, r: np.sqrt(1/4*(3*r + np.sqrt(r**2 - 8*a**2) - np.sqrt(8*a**2 + 2*r*(r - 2*np.sqrt(1 - 8*a**2) ))))
xf3 = lambda a, r: np.sqrt(1/4*(3*r - np.sqrt(r**2 - 8*a**2) + np.sqrt(8*a**2 + 2*r*(r + 2*np.sqrt(1 - 8*a**2) ))))
xf4 = lambda a, r: np.sqrt(1/4*(3*r - np.sqrt(r**2 - 8*a**2) - np.sqrt(8*a**2 + 2*r*(r + 2*np.sqrt(1 - 8*a**2) ))))

xf = [xf1, xf2, xf3, xf4]


a = 1.2
n=500

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)
x, y = np.meshgrid(x, y)

n = 500
a = 1.2

n = 301
r = np.linspace(0, a, 20)
theta = np.linspace(0, 2*np.pi, n)
r, theta = np.meshgrid(r, theta)
x, y = r*np.cos(theta), r*np.sin(theta)

z = x + 1j*y
r = 1
ws1 = [f(z, r) for f in xf]
ws2 = [-w for w in ws1]
ws = ws1 + ws2

data = [
    go.Surface(
        x=x, y=y, z=w.real,
        surfacecolor=w.imag, 
        showscale=False, 
        colorscale="Viridis"
    )
    for w in ws
]


def makeLayout():
    ax = dict(
        showgrid = False,
        zeroline = False,
        showline = False,
        showticklabels = False,
        ticks = "",
    )

    return go.Layout(
        scene = dict(
            xaxis = ax,
            yaxis = ax,
            zaxis = ax,     
            aspectmode = "data",
        ),
        width=1000, 
        height=1000 
    )

fig = go.Figure(
    data=data, 
    layout=makeLayout()
)
ply.plot(fig)


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# [ax.plot_surface(x, y, w.real, 
#     facecolors=cm.coolwarm(w.imag)
#     ) for w in ws]

# plt.show()
