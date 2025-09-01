import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.animation import FuncAnimation as FA

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)


fig, ax = plt.subplots()

plt.subplots_adjust(wspace=0, hspace=0, bottom=-1)
x = np.linspace(0, 10)
y = x**2

plt.show()
