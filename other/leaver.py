#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import cm, colors, collections

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

#%%
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

phibar = np.linspace(-1, 1, 10)
s = np.linspace(-1, 1, 100)

V = lambda s, p: 1/2 * (p**2 - 1) / (p*s - 1)
V2 = lambda s, p: 1 - V(s, p)

for p in phibar:
    c = (p + 1)/2
    ax[0].plot(s, V(s, p), color=cm.viridis(c))
    ax[1].plot(s, V2(s, p), color=cm.viridis(c))

ax[0].set_xlabel("$\\sigma$")
ax[1].set_xlabel("$\\sigma$")
ax[0].set_ylabel("$V$")
ax[1].set_ylabel("$1 - V$")

ax[0].set_ylim(-.1, 1.1)
ax[1].set_ylim(-.1, 1.1)

sp = lambda s, p: (p - V(s, p) * s) / V2(s, p)

for p in phibar:
    c = (p + 1)/2
    ax[2].plot(s, sp(s,p), color=cm.viridis(c))

ax[2].set_ylim(-1.1, 1.1)
ax[2].set_xlabel("$\\sigma$")
ax[2].set_ylabel("$\\sigma'$")

norm = colors.Normalize(vmin=-1, vmax=1)
cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
cmap.set_array([])
cb = fig.colorbar(cmap, ax=ax, location="right")
cb.set_label( label="$\\bar \\varphi$", labelpad=25, rotation=270)

fig.savefig('fig/vol.svg')

#%%
fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

pi = lambda s, p: np.sqrt(1 - s**2)
pip = lambda s, p: - V(s, p) / V2(s, p) * pi(s, p)

sigma = np.linspace(-1, 1, 40)
ps = np.linspace(0, -.9, 3)

for i, p in enumerate(ps):
    ind = i
    for s in sigma[:-1]:
        col = cm.coolwarm((s + 1)/2)

        ax[ind].plot(pi(s, p), s, 'x', color=col)
        ax[ind].plot(pip(s, p), sp(s, p), 'o', color=col)
        ax[ind].plot([pi(s, p), pip(s, p)], (s, sp(s, p)), color=col, alpha=.6)


    ax[ind].set_xlim(-1.1, 1.1)
    ax[ind].set_ylim(-1.1, 1.1)

    th = np.linspace(0, 2*np.pi, 100)
    ax[ind].plot(np.cos(th), np.sin(th), 'k--', zorder=10, alpha=.3)

    ax[ind].plot(0, p, 'kx')

ax[0].set_xlabel("$\\sigma$")
ax[1].set_xlabel("$\\sigma$")
ax[2].set_xlabel("$\\sigma$")
ax[0].set_ylabel("$\\pi$")

norm = colors.Normalize(vmin=-1, vmax=1)
cmap = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
cmap.set_array([])
cb = fig.colorbar(cmap, ax=ax, location="right")
cb.set_label( label="$\\sigma$", labelpad=25, rotation=270)
plt.show()
fig.savefig('fig/tie.svg')

# %%
