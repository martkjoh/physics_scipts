import bpy
import numpy as np
import os

print(os.getcwd())
os.chdir('/home/mjohnsrud/repos/NRCH/blender')

name = 'from_data'

# if bpy.context.object.mode == 'EDIT':
#     bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')


names = ["assym0", "assym1", "assym2", "assym3", "assym4"]

def get_vs(vertices):
    u, v, a = vertices.T
    return [
        np.array([+u, +v, a]).T,
        np.array([+v, +u, a]).T,
        np.array([-u, +v, a]).T,
        np.array([-v, +u, a]).T,
        np.array([+u, -v, a]).T,
        np.array([+v, -u, a]).T,
        np.array([-u, -v, a]).T,
        np.array([-v, -u, a]).T
    ]

def add_obj(name, vertices, faces, edges):
    vs = get_vs(vertices)

    for vertices in vs:
        new_mesh = bpy.data.meshes.new(name)
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()

        theObj = bpy.data.objects.new(name, new_mesh)
        bpy.context.collection.objects.link(theObj)

def delete(name):
    objs = [obj for obj in bpy.data.objects if obj.name[:len(name)]==name]
    bpy.ops.object.delete({'selected_objects': objs})

for name in names:
    delete(name)

    vertices = np.loadtxt('data'+name+'.csv', delimiter=',')
    faces = np.loadtxt('faces'+name+'.csv', delimiter=',', dtype=int)
    edges = []

    add_obj(name, vertices, faces, edges) 

name = 'CEP'

vertices = np.loadtxt('data'+name+'.csv', delimiter=',')


delete(name)
faces = []
edges = np.loadtxt('edges'+name+'.csv', delimiter=',', dtype=int)
add_obj(name, vertices, faces, edges)

