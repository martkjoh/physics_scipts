import numpy as np

N = 11
D = np.zeros((N, N))
i = np.linspace(0, N-1, N, dtype=int)

D[(i+1)%N, i] = -8
D[i, (i+1)%N] = +8
D[(i+2)%N, i] = +1
D[i, (i+2)%N] = -1
print(D)
print(D@D)