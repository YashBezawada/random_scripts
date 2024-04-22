# for creating a responsive plot
# %matplotlib qt

import uproot
import numpy as np
from scipy import stats as st

 
# importing required libraries
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import math

file = uproot.open("/home/yash/ArtieSim/build/outputs/ana_test_0.root")

Tree = file["Analysis"]
Branches = Tree.arrays()

x_pos = np.array(Branches["x_pos"])
y_pos = np.array(Branches["y_pos"])
z_pos = np.array(Branches["z_pos"])
Vx = np.array(Branches["N_x"])
Vy = np.array(Branches["N_y"])
Vz = np.array(Branches["N_z"])
x_tracking = np.array(Branches["x_tracking"])
y_tracking = np.array(Branches["y_tracking"])
z_tracking = np.array(Branches["z_tracking"])
# nE = np.array(Branches["n_energy"])
# nE_unique = np.unique(nE)

# plt_x = x_pos[(nE == nE_unique[0])]
# plt_y = y_pos[(nE == nE_unique[0])]
# plt_z = z_pos[(nE == nE_unique[0])]

def cuboid_data(o, size=(0.5,0.5,5)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)

    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(0.5,0.5,5)]*len(positions)
    # print(colors)
    # print(sizes)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
        # print(cuboid_data(p, size=s))
    
    return Poly3DCollection(np.concatenate(g), facecolors="w", alpha=0.5, **kwargs)

positions = []

for i in range(len(Vx)):
    x_vox = Vx[i]*0.5 # math.floor(x_tracking[i]*2)
    y_vox = Vy[i]*0.5 # math.floor(y_tracking[i]*2)
    z_vox = Vz[i]*5 # math.floor(z_tracking[i]*0.2)
    positions.append([x_vox,y_vox,z_vox])

positions = np.unique(np.array(positions), axis=0)

print(positions)

print("----------------------------")

traj_pos = []
for i in range(len(x_pos)):
    traj_pos.append([x_pos[i], y_pos[i], z_pos[i]])
traj_pos = np.unique(np.array(traj_pos), axis=0)
print(traj_pos)

# creating figure
fig = plt.figure()
ax = plt.axes(projection='3d')

pc = plotCubeAt(positions, colors="w",edgecolor="k")
ax.add_collection3d(pc)

traj_plot = ax.plot3D(x_pos, y_pos, z_pos, color="blue")

tracking_plot = ax.scatter3D(x_tracking, y_tracking, z_tracking, color="green")

ax.set_xlabel("X") 
ax.set_ylabel("Y") 
ax.set_zlabel("Z")

plt.show()