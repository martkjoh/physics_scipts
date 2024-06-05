#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

#%%

# rp = lambda a: 2 / (3*a**2 - 1) * (2*a + np.sqrt(a**2 + 1))
rm = lambda a: 2 / (3*a**2 - 1) * (2*a - np.sqrt(a**2 + 1))

n = 100
a = np.linspace(0, 1, n)

fig, ax = plt.subplots()
# ax.plot(a, rp(a))
ax.plot(a, rm(a))
ax.set_ylim(-.1, 2.1)
# %%
