#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

import sympy as sp
from sympy import *
from IPython.display import display, Latex


D = lambda f, x : (np.array(diff(f(x), Matrix(x))).T)[0]
def pr(T):
    if len(np.shape(T))==1:
        return display(Latex("$$" + sp.latex(Matrix(T)) +"$$"))
    elif len(np.shape(T))==2:
        return display(Latex("$$" + sp.latex(Matrix(T)) +"$$"))
    else:
        return display(Latex("$$" + sp.latex(T) +"$$"))

def smp(A, f=simplify):
    n, m = np.shape(A)
    B = np.empty_like(A)
    for i in range(n):
        for j in range(m):
            B[i, j] = f(A[i,j])
    return B

def smp2(A, f=simplify):
    n  = np.shape(A)[0]
    B = np.empty_like(A)
    for i in range(n):
        B[i] = f(A[i])
    return B

# %%
one = eye(2)
e = np.array([[0, 1], [-1, 0]])
t = symbols('\\theta')
A = np.array([[cos(t), sin(t)], [sin(t), -cos(t)]])
# %%
pr(A)
pr(A.T)
pr(smp(A@A))
pr(A@e)
# %%
p1, p2 = symbols('\\varphi_1, \\varphi_2')
phi = np.array([p1, p2])
pr(phi)
phi@A@phi
psi = A@phi
pr(psi)

#%%
g3 = psi @ e @ A @ psi
g3 = simplify(g3)
pr(g3)
dg = np.array([diff(g3, p1), diff(g3, p2)])

pr(smp2(A@dg))
# pr(A @ D(g, phi))
# %%
r, u, a = symbols('r, u, \\alpha')
f = lambda phi : r/2*phi@phi + u/4*(phi@phi)**2

mu1 = D(f, phi)
mu2 = a*e@phi
pr(mu1)
pr(mu1+mu2)
#%%
pr(A@mu1)

# %%
i = integrate((A@mu1)[0], p1) + integrate((A@mu1)[1], p2)
i = simplify(i)
pr(i)
#%%
pr(simplify(a/2*phi@(A@e)@phi))
# %%
g = i + a/2*phi@(A@e)@phi
g = factor(simplify(g))
pr(g)

#%%
M = np.array([simplify(diff(g, p1)), simplify(diff(g, p2))])
pr(M)
M = simplify(A@M)
pr(M)
#%%
pr(simplify(g.subs(t,0)))
pr(simplify(g.subs(t,pi/2)))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20,20))


t0, r0, u0, a0 = pi/4, -1, 1, 1/2

g = simplify(g.subs(t,t0))
g = simplify(g.subs(r,r0))
g = simplify(g.subs(u,u0))
g = simplify(g.subs(a,a0))
gl = lambdify(phi, g)


k = 1.5
N = 200
n = 50
r = np.linspace(0, k, N)
th = np.linspace(0, 2*np.pi, n)
r, th = np.meshgrid(r, th)
x, y = r * (np.cos(th), np.sin(th))
z = gl(x, y)

ax.plot_surface(x, y, z)
plt.show()

# %%
pr(phi@A@phi)
# %%
