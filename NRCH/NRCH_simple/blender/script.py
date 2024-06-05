import bpy
import math
import numpy as np

import os

print(os.getcwd())
os.chdir('/home/mjohnsrud/repos/NRCH/blender')

name = 'from_data'

bpy.ops.object.select_all(action='DESELECT')


names = ['spinodal1', "spinodal2", 'stability', 'exceptional']


def add_obj(name, vertices, faces, edges):
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

name = 'CEL'
delete(name)

vertices = np.loadtxt('data'+name+'.csv', delimiter=',')
faces = []
edges = np.loadtxt('edges'+name+'.csv', delimiter=',', dtype=int)
add_obj(name, vertices, faces, edges)

