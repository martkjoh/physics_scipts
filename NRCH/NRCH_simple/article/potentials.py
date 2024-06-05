from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos


plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("font", family="serif", size=16)



rho = np.linspace(0, 1.5)
th = np.linspace(0, 2*pi)

rho, th = np.meshgrid(rho, th)
x, y = rho*cos(th), rho*sin(th)


r = -1.
u = 1.

f1 = lambda x, y : r/2 * (x**2 + y**2) + u/4 * (x**2 + y**2)**2
f2 = lambda x, y : r/2 * (x**2 + y**2) + 2 * u/4 * (x**4 + y**4)


fig, ax = plt.subplots(subplot_kw={"projection": "3d", 'computed_zorder' : False}, figsize=(10,15))
# ax.plot_surface(x, y, f1(x, y),  antialiased=False)
ax.plot_surface(x, y, f2(x, y),  antialiased=False)

ax.set_zlim(-.2, .4)

plt.show()
