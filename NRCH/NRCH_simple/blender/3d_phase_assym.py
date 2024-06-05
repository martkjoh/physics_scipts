import numpy as np
from numpy import sqrt, pi, cos, sin

N = 100
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

amax = 1.5
    
## Old version
def assym0():
    name = "assym0"

    u = np.linspace(0, 1, N)
    a = np.linspace(0, 1, N)
    a, u = np.meshgrid(a, u)
    e=0
    th = e + (1/2 * np.arccos(a)  - e)* u

    A = (1 - cos(2*th)**2)
    C = 1 + a**2

    r = sqrt( (1 - sqrt(1 - A*C) ) / A )
    u, v = r * cos(th), r * sin(th)

    mask = np.zeros_like(u)

    data = list_to_data(u, v, a)
    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

def assym1():
    name = "assym1"

    u = np.linspace(0, 1, N)
    a = np.linspace(0, 1, N)
    u, a = np.meshgrid(u, a)

    th = 1/2 * np.arccos(a) + ( pi/4 - np.arccos(a)/2 )* u
    u, v = cos(th), sin(th)

    mask = np.zeros_like(u)

    data = list_to_data(u, v, a)
    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

def assym0():
    name = "assym0"

    u = np.linspace(0, 1/sqrt(2), N)
    v = np.linspace(1/sqrt(2), 1.2, N)
    u, v = np.meshgrid(u, v)
    a = sqrt((u**2 - v**2)**2 - (u**2 + v**2 - 1)**2 +0j).real

    mask = (v**2 - u**2)**2 - a**2 >= v**2 + u**2 - 1

    data = list_to_data(v, u, a)
    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

def assym4():
    name = "assym4"

    u = np.linspace(0, 1/sqrt(2), N)
    v = np.linspace(1/sqrt(2), 1.2, N)
    u, v = np.meshgrid(u, v)
    a = sqrt((u**2 - v**2)**2 - (u**2 + v**2 - 1)**2 +0j).real

    mask = (v**2 - u**2)**2 - a**2 < v**2 + u**2 - 1

    data = list_to_data(v, u, a)
    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

def assym2():
    name = "assym2"

    th = np.linspace(0, pi/4, N)
    a = np.linspace(1, 1.4, N)
    a, th = np.meshgrid(a, th)

    mask = np.zeros_like(a)

    data = list_to_data(cos(th), sin(th), a)
    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

def assym3():
    name = "assym3"

    a = np.linspace(0, 1.4, N)
    v = np.linspace(0, 1.4, N)
    a, v = np.meshgrid(a, v)
    u = sqrt(v**2 + a)
    mask = np.zeros_like(a)

    data = list_to_data(u, v, a)

    faces = faces_from_data(data, N, mask)
    save(data, [], faces, name)

    
def CEP():
    name = 'CEP'
    a = np.linspace(0, 1, N)
    u, v = sqrt( (1 - a) / 2 ), sqrt((1 + a)/ 2)
    data = [(u[i], v[i], a[i]) for i in range(N)]
    edges = [(i, i+1) for i in range(N-1)]

    save(data, edges, [], name)

assym0()
assym1()
assym2()
assym3()
assym4()
CEP()
 