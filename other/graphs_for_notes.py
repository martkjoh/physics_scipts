#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
arrow = patches.FancyArrowPatch
# import matplotlib.patches.FancyArrowPatch as arrow

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)

#%%

a, b = 20, 1
f = lambda x: b*2*(1/ (1 + np.exp(-a*x)) - 1/2) 


n = 500
x = np.linspace(-1, 1, n)

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = plt.twinx()
l1 = ax.plot(x, f(x), 'r', label="$f(x)$")
l2 = ax2.plot(x[1::], (f(x[1::]) - f(x[:-1:]))*(2*n), '--k', label="$f'(x)$")

l = l1 + l2
plt.legend(l, [a.get_label() for a in l], loc=2)

ax.arrow(-.2, -.6, .4, 0, width=.01, head_width=.05)
ax.arrow(.2, -.6, -.4, 0, width=.01, head_width=.05)
ax.text(-.03, -.75, '$\\xi$')
# ax.tick_params(labelleft=False, labelright=False)
# ax2.tick_params(labelleft=False, labelright=False)
plt.show()


# %%
a = 1.6
x = np.linspace(-a, a, 200)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, (1-x**2)**2, label='$\\mathcal{V}(\\varphi)$')
# ax.tick_params(labelleft=False, labelright=False)
ax.legend(loc=9)
# %%
a = 1.6
b = -.14
x = np.linspace(-a-b, a-b, 200)
fig, ax = plt.subplots(figsize=(10, 6))

a1, a2 = -1, -.02
ax.plot(x, (1-x**2)**2 + a1*x, label='$\\mathcal{V}(\\varphi)$')
ax.plot(x, a1*x+a2, 'k--')

a = arrow((-1, .9), (-1, -1), arrowstyle='<|-|>', lw=2, mutation_scale=10, zorder=4)
ax.add_artist(a)
a = arrow((-1, -1), (.9, -1), arrowstyle='<|-|>', lw=2, mutation_scale=10, zorder=4)
ax.add_artist(a)

ax.text(-1.2, -.1, '$\\Delta \\mathcal{V}$', zorder=4)
ax.text(-.1, -1.2, '$\\Delta \\varphi$', zorder=4)

plt.show()
# %%
a, b = 20, 1
f = lambda x: b*2*(1/ (1 + np.exp(-a*x)) - 1/2) 


n = 500
x = np.linspace(-1, 1, n)

fig, ax = plt.subplots(figsize=(10, 6))
l1 = ax.plot(x, f(x), 'r', label="$\\varphi(x, t)$")
l1 = ax.plot(x, f(x+.2), 'k--', label="$\\varphi(x, t + \\mathrm{d} t)$")

plt.legend()
a = [arrow((-.2, 0), (0, 0), arrowstyle='<|-|>', lw=2, mutation_scale=10, zorder=4), ]
ax.text(-.15, -.2, '$\\mathrm{d}x$')
a += [arrow((.2, .9), (.2, -.9), arrowstyle='<|-|>', lw=2, mutation_scale=10, zorder=4), ]
ax.text(.25, -.1, '$\\Delta\\varphi$', zorder=4)

a += [arrow((-.8, 0), (-.4, 0), arrowstyle='-|>', lw=4, mutation_scale=10, zorder=4), ]
a += [arrow((.4, 0), (.8, 0), arrowstyle='-|>', lw=4, mutation_scale=10, zorder=4), ]
ax.text(-.7, .1, '$J(R^-)$', zorder=4)
ax.text(+.5, .1, '$J(R^+)$', zorder=4)

[ax.add_artist(b) for b in a]
# ax.tick_params(labelleft=False, labelright=False)
# ax2.tick_params(labelleft=False, labelright=False)
plt.show()
# %%
