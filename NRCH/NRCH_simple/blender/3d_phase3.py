import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, sqrt
from matplotlib import colors as c
from matplotlib import cm
from matplotlib import ticker


plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("font", family="serif", size=16)


N = 100

fig, ax = plt.subplots(subplot_kw={"projection": "3d", 'computed_zorder' : False}, figsize=(10,15))


u = np.linspace(0, 1/sqrt(2), N)
v = np.linspace(1/sqrt(2), 1, N)
u, v = np.meshgrid(u, v)
a = sqrt((u**2 - v**2)**2 + (u**2 + v**2 - 1)**2)

ax.plot_surface(v, u, a)

u = np.linspace(0, 1, N)
a = np.linspace(0, 1, N)
a, u = np.meshgrid(a, u)
e=0
th = e + (1/2 * np.arccos(a)  - e)* u

A = (1 - cos(2*th)**2)
C = 1 + a**2

r1 = sqrt( (1 - sqrt(1 - A*C) ) / A )
u1, v1 = r1 * cos(th), r1 * sin(th)

th = 1/2 * np.arccos(a) + ( pi/4 - np.arccos(a)/2 )* u
u3, v3 = cos(th), sin(th)


ax.plot_surface(u1, v1, a)
ax.plot_surface(u3, v3, a)



ax.set_xlim(0, 1)
plt.show()