import yaml
import bpy
import os
import random

filepath = bpy.data.filepath
directory = os.path.dirname(filepath)
fileDir = directory+'/data/facelandmarks_ke.yml'
print(fileDir)


print("reading yaml file")

vertices = []
uv = []

with open(fileDir) as file:
    obj = yaml.safe_load(file)
    
    zero_x = (obj['Point3f']['data'][21*3] + obj['Point3f']['data'][22*3]) / 2.0
    zero_y = (obj['Point3f']['data'][21*3+1] + obj['Point3f']['data'][22*3+1]) / 2.0
    zero_z = (obj['Point3f']['data'][21*3+2] + obj['Point3f']['data'][22*3+2]) / 2.0
    
    
    for i in range(0, 77, 1):
        x = obj['Point3f']['data'][i*3] - zero_x
        y = obj['Point3f']['data'][i*3+1] - zero_y
        z = obj['Point3f']['data'][i*3+2] - zero_z
        
        vertices.append([x*0.4,y*0.4,z*0.4])
        
        u = obj['Point2f']['data'][i*2]
        v = obj['Point2f']['data'][i*2+1]
        
        uv.append([u,v])
        
print("vertices")
print(vertices)

faces = [[11,12,44],[11,44,45],[11,45,46],
        [11,46,48],[48,46,47],[48,47,40],
        [12,25,44],[25,43,44],[25,26,43],
        [25,24,26],[43,26,28],[43,28,42],
        [26,24,23],[26,23,27],[28,26,27],
        [41,28,27],[41,42,28],[40,41,22],
        [22,41,27],[22,27,23],[13,23,24],
        [14,23,13],[14,22,23],
        [0,1,34],[1,35,34],[1,36,35],
        [1,50,36],[50,37,36],[50,30,37],
        [0,34,18],[18,34,33],[18,33,19],
        [18,19,17],[33,32,29],[33,29,19],
        [19,16,17],[19,20,16],[19,29,20],
        [32,31,29],[29,31,20],[31,30,21],
        [20,31,21],[16,20,21],[17,16,15],
        [15,16,14],[14,16,21],[14,21,22],
        [21,49,22],[21,30,50],[21,50,49],
        [22,49,48],[22,48,40],[50,51,49],
        [51,52,49],[49,52,53],[49,53,48],
        [51,53,52],[50,58,51],[58,57,51],
        [51,57,56],[51,56,53],[53,56,55],
        [53,55,54],[48,53,54],[1,2,58],
        [1,58,50],[2,3,58],[3,57,58],
        [48,54,11],[11,54,10],[10,54,9],
        [54,55,9],[56,61,62],[56,62,63],
        [57,61,56],[57,60,61],[57,59,60],
        [3,59,57],[3,4,59],[56,62,63],
        [56,63,55],[55,63,64],[55,64,65],
        [55,65,9],[9,65,8],[59,4,76],
        [4,5,76],[5,75,76],[5,6,75],
        [6,74,75],[74,6,73],[6,7,73],
        [7,72,73],[7,8,72],[8,65,72],
        [59,68,60],[60,68,61],[68,67,61],
        [61,67,62],[62,67,63],[63,67,66],
        [63,66,64],[64,66,65],[59,76,69],
        [76,75,69],[75,70,69],[75,74,70],
        [70,74,73],[70,73,71],[71,73,72],
        [71,72,65],
        #Right Eye
        [34,35,38],[35,36,38],[36,37,38],
        [37,30,38],[30,31,38],[31,32,38],
        [32,33,38],[33,34,38],
        #Left Eye
        [40,47,39],[47,46,39],[46,45,39],
        [45,44,39],[44,43,39],[43,42,39],
        [42,41,39],[41,40,39]
        ]

mesh = bpy.data.meshes.new("faceMesh")
mesh.from_pydata(vertices,[],faces)

uvtex = mesh.uv_textures.new()
uvtex.name = "UV Name"
for ly in mesh.uv_textures:
    i = 0
    for idx, dat in enumerate(mesh.uv_layers[ly.name].data):
        vertIdx = faces[(int)(i/3)][i%3]
        dat.uv = [uv[vertIdx][0] / 4096, -uv[vertIdx][1] / 2048]
        i += 1

mesh.update()
obj = bpy.data.objects.new("face",mesh)
scene = bpy.context.scene
scene.objects.link(obj)

        
fileDir = directory+'/data/facelandmarks_kris.yml'
print(fileDir)


print("reading yaml file")

vertices = []
uv = []

with open(fileDir) as file:
    obj = yaml.safe_load(file)
    
    zero_x = (obj['Point3f']['data'][21*3] + obj['Point3f']['data'][22*3]) / 2.0
    zero_y = (obj['Point3f']['data'][21*3+1] + obj['Point3f']['data'][22*3+1]) / 2.0
    zero_z = (obj['Point3f']['data'][21*3+2] + obj['Point3f']['data'][22*3+2]) / 2.0
    
    for i in range(0, 77, 1):
        x = obj['Point3f']['data'][i*3] - zero_x
        y = obj['Point3f']['data'][i*3+1] - zero_y
        z = obj['Point3f']['data'][i*3+2] - zero_z
        
        vertices.append([x*0.4,y*0.4,z*0.4])
        
        u = obj['Point2f']['data'][i*2]
        v = obj['Point2f']['data'][i*2+1]
        
        uv.append([u,v])
        
        
print("vertices")
print(vertices)

mesh = bpy.data.meshes.new("faceMesh")
mesh.from_pydata(vertices,[],faces)

uvtex = mesh.uv_textures.new()
uvtex.name = "UV Name"
for ly in mesh.uv_textures:
    i = 0
    for idx, dat in enumerate(mesh.uv_layers[ly.name].data):
        vertIdx = faces[(int)(i/3)][i%3]
        dat.uv = [uv[vertIdx][0] / 4096, -uv[vertIdx][1] / 2048]
        i += 1


mesh.update()
obj = bpy.data.objects.new("face2",mesh)
scene = bpy.context.scene
scene.objects.link(obj)