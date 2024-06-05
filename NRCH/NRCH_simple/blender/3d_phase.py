import numpy as np
from numpy import sqrt

N = 50
m = 50
th = 1
k = sqrt(th)

def list_to_data(x, y, z):
    n = np.shape(x)[0]
    data = []
    for i in range(n):
        for j in range(n):
            data.append((x[i, j], y[i, j], z[i, j]))

    return data


def faces_from_data(data, n, mask):
    indices = np.arange(len(data)).reshape(n, n)
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

    return faces


def save(data, edges, faces, name):
    np.savetxt('blender/data'+name+'.csv', data, delimiter=',')
    np.savetxt('blender/edges'+name+'.csv', edges, delimiter=',', fmt='%i')
    np.savetxt('blender/faces'+name+'.csv', faces, delimiter=',', fmt='%i')
    

def spin():
    name='spinodal'

    u0 = np.linspace(0, +.25, m)
    v0 = np.linspace(0, -1, m)

    u, v = np.meshgrid(u0, v0)

    r = v
    x = sqrt(- (1 + 2*u)/3 * r)
    a = sqrt(x**4 - (r + 2*x**2)**2 + 0j).real

    mask = np.zeros_like(a)

    data = list_to_data(x, r, a)
    faces = faces_from_data(data, m, mask)
    save(data, [], faces, name+"1")

    data = list_to_data(-x, r, a)
    faces = faces_from_data(data, m, mask)
    save(data, [], faces, name+"2")

def spin2():
    name='spinodal2'

    u0 = np.linspace(.25, 1, m)
    v0 = np.linspace(0, -1, m)

    u, v = np.meshgrid(u0, v0)

    r = v
    x = sqrt(- (1 + 2*u)/3 * r)
    a = sqrt(x**4 - (r + 2*x**2)**2 + 0j).real

    mask = np.zeros_like(a)

    data = list_to_data(x, r, a)
    faces = faces_from_data(data, m, mask)
    save(data, [], faces, name+"1")

    data = list_to_data(-x, r, a)
    faces = faces_from_data(data, m, mask)
    save(data, [], faces, name+"2")


def stab():
    name = 'stability'
    z = np.linspace(0, th/2 + 0.1, N)
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    z, x = np.meshgrid(z, x)
    y = - 2 * x**2

    mask = z<x**2

    data = list_to_data(x, y, z)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)

def stab():
    name = 'stability'
    u0, v0 = np.linspace(-1, 1, N), np.linspace(0, 1, N)
    u, v = np.meshgrid(u0, v0)

    x = k/sqrt(2) * u
    a = th/2 + 0.1
    z = a * v + u**2 *0.5 * (1 - v)
    y = - 2 * x**2

    mask = np.zeros_like(x)

    data = list_to_data(x, y, z)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)

def exc():
    name = 'exceptional'
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    y = np.linspace(0, -th, N)
    x, y = np.meshgrid(x, y)
    z = x**2
    mask = np.zeros_like(x)

    data = list_to_data(x, y, z)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)

def crit_exc():
    name = 'CEL'
    t = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    x, y, z = t, -2*t**2, t**2
    data = [(x[i], y[i], z[i]) for i in range(N)]
    edges = [(i, i+1) for i in range(N-1)]

    save(data, edges, [], name)


spin()
spin2()
stab()
exc()
crit_exc()