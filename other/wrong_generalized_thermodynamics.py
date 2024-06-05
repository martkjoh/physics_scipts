#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)

import sympy as sp
from sympy import *
from IPython.display import display, Latex

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

# %%
one = eye(2)
t = symbols('\\theta')
A = np.array([[cos(t), sin(t)], [sin(t), -cos(t)]])
pr(A)
e = np.array([[0, 1], [-1, 0]])
pr(e)
# %%
pr(A)
pr(A.T)
pr(smp(A@A))
pr(e@A)
# %%
p1, p2 = symbols('\\varphi_1, \\varphi_2')
phi = np.array([p1, p2])
pr(phi)
# %%
phi@A@phi
# %%
r, u, a = symbols('r, u, \\alpha')

g = r/2 * phi@phi + u/4 * (phi@phi)**2 + a/2 * phi@(A@e)@phi
g = simplify(g)
pr(g)

# %%
dg = np.array([diff(g, p1), diff(g, p2)])
dg = simplify(dg)
pr(dg)
# %%
solve([dg[0], dg[1]], (p1, p2))
# %%
pr(Matrix(A).eigenvects())
# %%
k = Matrix([[2, 0], [0, 1]])
pr(k.eigenvects())
