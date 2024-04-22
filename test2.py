import matplotlib.pyplot as plt
import numpy as np
import uproot
import ROOT

myFile = ROOT.TFile.Open("gausProfile_sig47mm.root", "RECREATE")
rand3 = ROOT.TRandom3()
bins = 100
profileHist2D = ROOT.TH2D("ProfileHist2D", "Gaussian 2D Profile", bins, -12.0, 12.0, bins, -12.0, 12.0)
# sig = 1.7 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
sig = 4.7
for i in range(1000000):
    profileHist2D.Fill(rand3.Gaus(0, sig), rand3.Gaus(0, sig))
c1 = ROOT.TCanvas()
ROOT.gStyle.SetPalette(57)
profileHist2D.Draw("colz")
myFile.WriteObject(profileHist2D, "ProfileHist2D")

# file = ROOT.TFile.Open("/home/yash/ArtieSim/data/evalflux.root", "READ")

# hist = file.Get("hEval_Abs")
# num_bins = hist.GetNbinsX()
# print("Min Energy = ", hist.GetXaxis().GetBinLowEdge(1))
# print("Max Energy = ", hist.GetXaxis().GetBinUpEdge(num_bins))
# c1 = ROOT.TCanvas()
# hist.Draw()
# c1.Print("test2.png")
# c1 = file.Get("Canvas_1")
# # c1 = ROOT.TCanvas()
# # c1 = file["Canvas_1"]
# # c1.Print("test2.png")
# h1 = c1.GetPrimitive("histfluka")
# c2 = ROOT.TCanvas()
# c2.SetLogx()``
# c2.SetLogy()
# h1.Draw()
# c2.Print("test2.png")


# Tree = file["Analysis"]
# Branches = Tree.arrays()

# x_pos = np.array(Branches["x_pos"])
# y_pos = np.array(Branches["y_pos"])
# z_pos = np.array(Branches["z_pos"])
# Vx = np.array(Branches["N_x"])
# Vy = np.array(Branches["N_y"])
# Vz = np.array(Branches["N_z"])

# # Create a numpy array representing the voxels
# voxel_size = (0.5, 0.5, 5)

# positions = np.c_[Vx,Vy,Vz]
# print(positions)

# # and plot everything
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # Make grid
# test2 = np.zeros((9, 9, 5))
# # Activating Voxels
# for pos in positions:
#     test2[pos[0], pos[1], pos[2]] = True
#     # print(i)
# print(test2)
# # ax.voxels(voxel, facecolors="w", edgecolor='k')

# # plt.show()


# voxel = [[[0.,  0.5, 1. ],
#         [0.,  0.,  1. ],
#         [0.5, 0.,  1. ],
#         [0.5, 0.5, 1. ]]

#         [[0.,  0.,  1. ],
#         [0.,  0.,  5. ],
#         [0.5, 0.,  5. ],
#         [0.5, 0.,  1. ]]

#         [[0.5, 0.,  5. ],
#         [0.5, 0.,  1. ],
#         [0.5, 0.5, 1. ],
#         [0.5, 0.5, 5. ]]

#         [[0.,  0.,  5. ],
#         [0.,  0.,  1. ],
#         [0.,  0.5, 1. ],
#         [0.,  0.5, 5. ]]

#         [[0.,  0.5, 1. ],
#         [0.,  0.5, 5. ],
#         [0.5, 0.5, 5. ],
#         [0.5, 0.5, 1. ]]

#         [[0.,  0.5, 5. ],
#         [0.,  0.,  5. ],
#         [0.5, 0.,  5. ],
#         [0.5, 0.5, 5. ]]]
