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


N = 200

fig, ax = plt.subplots(subplot_kw={"projection": "3d", 'computed_zorder' : False}, figsize=(10,15))


u0 = np.linspace(0, +1, N)
v0 = np.linspace(0, -1, N)

u, v = np.meshgrid(u0, v0)

r = v
x = sqrt(- (1 + 2*u)/3 * r)
a = sqrt(x**4 - (r + 2*x**2)**2 + 0j).real


ax.plot_surface(x, r, a, cmap=cm.Blues, antialiased=False)
ax.plot_surface(-x, r, a, cmap=cm.Blues, antialiased=False)


plt.show()