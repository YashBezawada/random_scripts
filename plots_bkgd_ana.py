import uproot
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import csv
import ROOT

def EnergyToTOF(e):
    KE_M = e/939.56542052
    denominator = 1.0 - 1.0/((KE_M + 1.0)*(KE_M + 1.0))
    correction_factor = np.sqrt(1.0 / denominator)
    return (188/299792458) * correction_factor

def TOFToEnergy(t):
    denom_term = (188)/(299792458 * t)
    denominator = 1.0 - (denom_term * denom_term)
    factor = np.sqrt(1.0 / denominator) - 1
    return 939.56542052 * factor

# energy = []
# transmission = []

# with open('sig_n_al27_1_100_kev.txt') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=' ')
#     line_count = 0
#     d = 5.08 #cm
#     b_to_cm2 = 10e-24
#     rho_eff = 2.699 #g/cm3
#     m_Ar = 6.6335209e-23 #g
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#         else:
#             energy.append(float(row[1]))
#             for i in range(6,12):
#                 if (row[i] != ''):
#                     t_e = np.exp(- rho_eff * d * float(row[i]) * b_to_cm2 / m_Ar)
#                     transmission.append(t_e)
#                     break

# energy = np.array(energy)
# transmission = np.array(transmission)
# plt.scatter(energy, transmission)
# plt.yscale("log")
# plt.title("transmission plot for Al-27 (thickness = 5.08 cm)")
# plt.xlabel("Energy in eV")
# plt.ylabel("transmission")
# plt.show()

# bins = 99
# min = 1000
# max = 100000
# bin_width = (max - min)/99.0

# e_high =  min + bin_width# 2000
# transmission_bin_count = []
# avg_trans_sum = 0
# counter = 0
# for i in range(len(energy)):
#     if energy[i] < e_high:
#         avg_trans_sum += transmission[i]
#         counter += 1
#         if i == len(energy) - 1:
#             transmission_bin_count.append(avg_trans_sum/counter)
#         elif energy[i+1] >= e_high:
#             transmission_bin_count.append(avg_trans_sum/counter)
#             e_high += bin_width
#             counter = 0
#             avg_trans_sum = 0

# print(len(transmission_bin_count))

# c1 = ROOT.TCanvas()
# c1.SetLogy()
    
# trans_hist = ROOT.TH1D("trans_hist","transmission hist",bins,min,max)
# for i in range(bins):
#     trans_hist.SetBinContent(i+1, transmission_bin_count[i])

# trans_hist.GetXaxis().SetTitle("Energy in eV")
# trans_hist.GetYaxis().SetTitle("Transmission")
# trans_hist.SetTitle("Transmission hist for Al-27 (thickness = 5.08 cm)")
# trans_hist.Draw()
# c1.Print("Plots/Transmission_Hist.png")

# bins = 99
# min = EnergyToTOF(trans_hist.GetXaxis().GetBinUpEdge(99) * 1e-6) * 1e9
# max = EnergyToTOF(trans_hist.GetXaxis().GetBinLowEdge(1) * 1e-6) * 1e9
# bin_width = (max - min)/99.0

# print("Min TOF = ", min)
# print("Max TOF = ", max)
# print("Bin width TOF = ", bin_width)

# # e high, tof low
# tof = []
# for e in energy:
#     tof.append(EnergyToTOF(e * 1e-6) * 1e9)
# tof = np.array(tof)
# # tof_inv is low to high
# tof_inv = np.flip(tof)
# transmission_inv = np.flip(transmission)

# print("first tof_inv element = ", tof_inv[0])
# print("last tof_inv element = ", tof_inv[-1])

# is_sorted = lambda a: np.all(a[:-1] <= a[1:])
# print(is_sorted(tof_inv))

# tof_high = min + bin_width
# tof_low = min
# transmission_tof_bin_count = []
# avg_trans_tof_sum = 0
# counter = 0
# iter_counter = 0
# for i in range(len(tof_inv)):
#     if tof_inv[i] >= tof_high:
#         transmission_tof_bin_count.append(0)
#     if tof_inv[i] < tof_high:
#         avg_trans_tof_sum += transmission_inv[i]
#         counter += 1
#         if i == len(tof_inv) - 1:
#             transmission_tof_bin_count.append(avg_trans_tof_sum/counter)
#         elif tof_inv[i+1] >= tof_high:
#             transmission_tof_bin_count.append(avg_trans_tof_sum/counter)
#             tof_high += bin_width
#             counter = 0
#             avg_trans_tof_sum = 0

# while iter_counter < len(tof_inv):
#     if tof_inv[iter_counter] >= tof_high:
#         transmission_tof_bin_count.append(0)
#         tof_high += bin_width
#         continue
#     if tof_inv[iter_counter] < tof_high:
#         avg_trans_tof_sum += transmission_inv[iter_counter]
#         counter += 1
#         if iter_counter == len(tof_inv) - 1:
#             transmission_tof_bin_count.append(avg_trans_tof_sum/counter)
#         elif tof_inv[iter_counter+1] >= tof_high:
#             transmission_tof_bin_count.append(avg_trans_tof_sum/counter)
#             tof_high += bin_width
#             counter = 0
#             avg_trans_tof_sum = 0
#     iter_counter += 1

# for i in range(len(tof_inv)):
#     if tof_inv[i] < tof_high and tof_inv[i] >= tof_low:
#         avg_trans_tof_sum += transmission_inv[i]
#         counter += 1
#         if tof_inv[i+1] >= tof_high:
#             transmission_tof_bin_count.append(avg_trans_tof_sum/counter)
#             tof_high += bin_width
#             tof_low += bin_width
#             counter = 0
#             avg_trans_tof_sum = 0
#     else:
#         transmission_tof_bin_count.append(0)
#         tof_high += bin_width
#         tof_low += bin_width

# print(len(transmission_tof_bin_count))

trans_tof_hist = ROOT.TH1D("trans_tof_hist","Transmission Hist (ToF x-axis)",bins,min,max)
for i in range(bins):
    trans_tof_hist.SetBinContent(i+1, transmission_tof_bin_count[i])

# trans_tof_hist.GetXaxis().SetTitle("ToF in ns")
# trans_tof_hist.GetYaxis().SetTitle("Transmission")
# trans_tof_hist.SetTitle("Transmission Al-27 (Flight Path = 188m, thickness = 5.08 cm)")
# trans_tof_hist.SetStats(0)
# trans_tof_hist.Draw()
# c1.Print("Plots/Transmission_ToF_Hist.png")

#########################################################################################################################

file_in = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_ideal_short_filter_in_0.root")
Tree_in = file_in["bkgdAnalysis"]
Branches_in = Tree_in.arrays()
n_energy = np.array(Branches_in["n_energy"])
n_energy = n_energy[n_energy != 0]
n_detected_in = np.array(Branches_in["n_energy_detected"])
n_detected_in = np.array( n_detected_in[n_detected_in != 0] )
bkgd_events_in = np.array(Branches_in["n_tof_bkgd"])
bkgd_events_in = bkgd_events_in[bkgd_events_in != 0]

file_out = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_ideal_short_filter_out_0.root")
Tree_out = file_out["bkgdAnalysis"]
Branches_out = Tree_out.arrays()
n_detected_out = np.array(Branches_out["n_energy_detected"])
n_detected_out = np.array( n_detected_out[n_detected_out != 0] )

c1 = ROOT.TCanvas()
c1.SetLogy()

n_tof = []
for e in n_energy:
    n_tof.append(EnergyToTOF(e) * 1e9)
n_tof = np.array(n_tof)

# print("min = ", np.min(bkgd_events_in))
# print("max = ", np.max(bkgd_events_in))

bins = 99
min = EnergyToTOF(0.1) * 1e9
max = EnergyToTOF(0.001) * 1e9

n_tof_hist = ROOT.TH1D("n_tof_hist","Generated Neutron ToF",bins,min,max)
for t in n_tof:
    n_tof_hist.Fill(t)
n_tof_hist.GetXaxis().SetTitle("ToF in ns")
n_tof_hist.GetYaxis().SetTitle("Neutron Count")
n_tof_hist.Draw()
c1.Print("Plots/n_tof_Hist.png")

bkgd_tof_hist = ROOT.TH1D("bkgd_tof_hist","Generated Background Events",bins,min,max)
for t in bkgd_events_in:
    bkgd_tof_hist.Fill(t)
# bkgd_tof_hist.GetXaxis().SetTitle("ToF in ns")
# bkgd_tof_hist.GetYaxis().SetTitle("Neutron Count")
# bkgd_tof_hist.Draw()
# bkgd_tof_hist.GetYaxis().SetRange(0, 200)
# c1.Print("Plots/bkgd_tof_Hist.png")

n_detected_tof = []
for e in n_detected_in:
    n_detected_tof.append(EnergyToTOF(e) * 1e9)
n_detected_tof = np.array(n_detected_tof)

n_detected_tof_hist = ROOT.TH1D("n_detected_tof_hist","Detected Neutron ToF",bins,min,max)
for t in n_detected_tof:
    n_detected_tof_hist.Fill(t)
# n_detected_tof_hist.GetXaxis().SetTitle("ToF in ns")
# n_detected_tof_hist.GetYaxis().SetTitle("Neutron Count")
# n_detected_tof_hist.Draw()
# c1.Print("Plots/n_detected_tof_Hist.png")

total_detected_hist_in = ROOT.TH1D("total_detected_hist_in","Detected Neutron Events + Bkgd Events",bins,min,max)
for t_d in n_detected_tof:
    total_detected_hist_in.Fill(t_d)
    
for t_b in bkgd_events_in:
    total_detected_hist_in.Fill(t_b)
    # new_bin_value = n_detected_tof_hist.GetBinContent(i+1) + bkgd_tof_hist.GetBinContent(i+1)
    # total_detected_hist_in.SetBinContent(i+1, new_bin_value)
# total_detected_hist_in.GetXaxis().SetTitle("ToF in ns")
# total_detected_hist_in.GetYaxis().SetTitle("Neutron Count")
# total_detected_hist_in.Draw()
# c1.Print("Plots/total_detected_tof_Hist_v2.png")

# bin_list = []
# for i in range(bins):
#     if trans_tof_hist.GetBinContent(i + 1) < 1e-7 and trans_tof_hist.GetBinContent(i + 1) != 0:
#         bin_list.append(i + 1)

# N_filter_in_sum = 0
# for bin in bin_list:
#     N_filter_in_sum += total_detected_hist_in.GetBinContent(bin)
# AVG_background = N_filter_in_sum / len(bin_list)

# print("Calculated backgrond per bin = ", AVG_background)
# print("Simulated background per bin = ", 9996/99)

# bins = 99
# min = 0.001 #1e-4
# max = 0.1 #5e-5
# n_energy_hist = ROOT.TH1D("n_energy_hist","Generated Neutron Energy",bins,min,max)
# for e in n_energy:
#     n_energy_hist.Fill(e)
# n_energy_hist.GetXaxis().SetTitle("Energy in MeV")
# n_energy_hist.GetYaxis().SetTitle("Neutron Count")
# n_energy_hist.Draw()
# c1.Print("Plots/n_energy_Hist.png")

# bkgd_hist = ROOT.TH1D("bkgd_hist","Generated Background Events",bins,min,max)
