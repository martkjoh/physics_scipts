import numpy as np
from numpy import sin, cos, pi

n = 500
x, y = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
x, y = np.meshgrid(x, y)
z = 0.5 * sin(pi*x) * sin(pi*y)



data = []
for i in range(n):
    for j in range(n):
        data.append((x[i, j], y[i, j], z[i, j]))

np.savetxt('data.csv', data, delimiter=',')

indices = np.arange(len(data)).reshape(n, n)
np.savetxt('indices.csv', indices, delimiter=',', fmt='%i')


mask = z > 0.2

faces = []
for i in range(n - 1):
    for j in range(n - 1):
        if mask[i, j]:
            continue
        v1 = indices[i][j]
        v2 = indices[i][j+1]
        v3 = indices[i+1][j+1]
        v4 = indices[i+1][j]
        faces.append([v1, v2, v3, v4])



np.savetxt('faces.csv', faces, delimiter=',', fmt='%i')

