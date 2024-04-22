import uproot
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import csv
import ROOT
from array import array

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

def FindBinNum(x, bin_edges):
    for i in range(len(bin_edges)):
        if bin_edges[i] > x:
            return i-1
    if x == bin_edges[-1]:
        return  len(bin_edges) - 1

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

######################################################### Extracting ENDF data

energy = []
transmission = []

with open('sig_n_al27_0_20_MeV.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    d = 5.08 #cm
    b_to_cm2 = 10e-24
    rho_eff = 2.699 #g/cm3
    m_Ar = 6.6335209e-23 #g
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            energy.append(float(row[1]))
            for i in range(3,12):
                if (row[i] != ''):
                    t_e = np.exp(- rho_eff * d * float(row[i]) * b_to_cm2 / m_Ar)
                    transmission.append(t_e)
                    break

energy = np.array(energy)
transmission = np.array(transmission)

############ converting ENDF data into a histogram
bins = 99
min_edge = 1000
max_edge = 100000
bin_width = (max_edge - min_edge)/bins

e_high =  min_edge + bin_width# 2000
transmission_bin_content = []
avg_trans_sum = 0
counter = 0
for i in range(len(energy)):
    if energy[i] < e_high:
        avg_trans_sum += transmission[i]
        counter += 1
        if i == len(energy) - 1:
            transmission_bin_content.append(avg_trans_sum/counter)
        elif energy[i+1] >= e_high:
            transmission_bin_content.append(avg_trans_sum/counter)
            e_high += bin_width
            counter = 0
            avg_trans_sum = 0
    
trans_hist = ROOT.TH1D("trans_hist","transmission hist",bins,min_edge,max_edge)
for i in range(bins):
    trans_hist.SetBinContent(i+1, transmission_bin_content[i])

############ converting ENDF energy hist into tof hist (non-uniform bin width)

bin_edges_e = []
for i in range(bins):
    if i == bins - 1:
        bin_edges_e.append(trans_hist.GetXaxis().GetBinLowEdge(bins))
        bin_edges_e.append(trans_hist.GetXaxis().GetBinUpEdge(bins))
    else:
        bin_edges_e.append(trans_hist.GetXaxis().GetBinLowEdge(i+1))

bin_edges_tof = []
for i in range(len(bin_edges_e)):
    bin_edges_tof.append(EnergyToTOF(bin_edges_e[i] * 1e-6) * 1e9)
bin_edges_tof.reverse()

bin_content = []
for i in range(bins):
    bin_content.append(trans_hist.GetBinContent(i+1))
bin_content.reverse()

### Making bin edge list of bins
threshold = 1e-7
bin_edge_list_low = []
bin_edge_list_up = []
region_start = False
for i in range(bins):
    if region_start == False:
        if bin_content[i] <= threshold and bin_content[i] != 0:
            bin_edge_list_low.append(bin_edges_tof[i])
            region_start = True
            if i == bins -1:
                bin_edge_list_up.append(bin_edges_tof[i+1])
    if region_start == True:
        if bin_content[i] > threshold:
            bin_edge_list_up.append(bin_edges_tof[i])
            region_start = False
        elif i == bins - 1:
            bin_edge_list_up.append(bin_edges_tof[i+1])

trans_tof_hist = ROOT.TH1D("trans_tof_hist","Transmission Hist (ToF x-axis)",bins,array('d', bin_edges_tof))
for i in range(bins):
    trans_tof_hist.SetBinContent(i+1, trans_hist.GetBinContent(bins - i))

### Making list of bins below the threshold 
bin_list = []
for i in range(bins):
    if trans_tof_hist.GetBinContent(i + 1) < threshold and trans_tof_hist.GetBinContent(i + 1) != 0:
        bin_list.append(i + 1)

# c1 = ROOT.TCanvas()
# c1.SetLogy()
# trans_tof_hist.GetXaxis().SetTitle("ToF in ns")
# trans_tof_hist.GetYaxis().SetTitle("Transmission")
# trans_tof_hist.SetTitle("Transmission Al-27 (Flight Path = 188m, thickness = 5.08 cm)")
# trans_tof_hist.SetStats(0)
# trans_tof_hist.Draw()
# c1.Print("Plots/Transmission_ToF_Hist.png")

######################################################### Extrating geant4 sim data

### Filter in
file_in = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_uniformBkgd_3milevts_5cmFilter_in_0.root")
Tree_in = file_in["bkgdAnalysis"]
Branches_in = Tree_in.arrays()
n_energy = np.array(Branches_in["n_energy"])
n_energy = n_energy[n_energy != 0]
n_detected_in = np.array(Branches_in["n_energy_detected"])
n_detected_in = np.array( n_detected_in[n_detected_in != 0] )
bkgd_events_in = np.array(Branches_in["n_tof_bkgd"])
bkgd_events_in = bkgd_events_in[bkgd_events_in != 0]

# ### Filter out
# file_out = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_ideal_short_filter_out_0.root")
# Tree_out = file_out["bkgdAnalysis"]
# Branches_out = Tree_out.arrays()
# n_detected_out = np.array(Branches_out["n_energy_detected"])
# n_detected_out = np.array( n_detected_out[n_detected_out != 0] )

c1 = ROOT.TCanvas()
c1.SetLogy()

# n_tof = []
# for e in n_energy:
#     n_tof.append(EnergyToTOF(e) * 1e9)
# n_tof = np.array(n_tof)

# print("min = ", np.min(bkgd_events_in))
# print("max = ", np.max(bkgd_events_in))

bins = 200
min_edge = EnergyToTOF(0.1) * 1e9 # min([np.min(n_energy), np.min(bkgd_events_in)]) 
max_edge = EnergyToTOF(0.001) * 1e9 # max([np.max(n_energy), np.max(bkgd_events_in)])
width_bin = (max_edge - min_edge) / bins

# n_tof_hist = ROOT.TH1D("n_tof_hist","Generated Neutron ToF",bins,min,max)
# for t in n_tof:
#     n_tof_hist.Fill(t)

# bkgd_tof_hist = ROOT.TH1D("bkgd_tof_hist","Generated Background Events",bins,min,max)
# for t in bkgd_events_in:
#     bkgd_tof_hist.Fill(t)

### Converting neutron energy to tof
n_detected_tof = []
for e in n_detected_in:
    n_detected_tof.append(EnergyToTOF(e) * 1e9)
n_detected_tof = np.array(n_detected_tof)

# n_detected_tof_hist = ROOT.TH1D("n_detected_tof_hist","Detected Neutron ToF",bins,min,max)
# for t in n_detected_tof:
#     n_detected_tof_hist.Fill(t)

### Adding Detected neutrons and the background
total_detected_hist_in = ROOT.TH1D("total_detected_hist_in","Detected Neutron Events + Bkgd Events",bins,min_edge,max_edge)
for t_d in n_detected_tof:
    total_detected_hist_in.Fill(t_d)
for t_b in bkgd_events_in:
    total_detected_hist_in.Fill(t_b)
total_detected_hist_in.GetXaxis().SetTitle("ToF in ns")
total_detected_hist_in.GetYaxis().SetTitle("Neutron Count")
total_detected_hist_in.Draw()
c1.Print("Plots/total_detected_tof_Hist.png")

### Determining the background
N_filter_in_sum = 0
bin_counter = 0
for i in range(len(bin_edge_list_low)):
    bin_num_low = total_detected_hist_in.GetXaxis().FindBin(bin_edge_list_low[i]) + 1 # FindBinNum(bin_edge_list_low[i], bin_edges_tof) + 1
    bin_num_up = total_detected_hist_in.GetXaxis().FindBin(bin_edge_list_up[i]) - 1 # FindBinNum(bin_edge_list_up[i], bin_edges_tof) - 1
    print(bin_num_low)
    print(bin_num_up)
    bin_num = bin_num_low
    while bin_num <= bin_num_up:
        N_filter_in_sum += total_detected_hist_in.GetBinContent(bin_num)
        bin_counter += 1
        bin_num += 1

AVG_background = N_filter_in_sum / bin_counter

print("Calculated backgrond per bin = ", AVG_background)
print("Simulated background per bin = ", 30000/200)
print(N_filter_in_sum)

############################################################################################

# bins = 99
# ### Rebinning sim data to non-uniform bin width
# tot_detected_in_rebin = ROOT.TH1D("tot_detected_in_rebin","Detected Neutron Events + Bkgd Events",bins,array('d', bin_edges_tof))
# for t_d in n_detected_tof:
#     tot_detected_in_rebin.Fill(t_d)    
# for t_b in bkgd_events_in:
#     tot_detected_in_rebin.Fill(t_b)

# tot_detected_in_rebin.GetXaxis().SetTitle("ToF in ns")
# tot_detected_in_rebin.GetYaxis().SetTitle("Neutron Count")
# tot_detected_in_rebin.Draw()
# c1.Print("Plots/total_detected_tof_Hist_rebin.png")

# ### Determining the background
# N_filter_in_sum = 0
# for bin in bin_list:
#     N_filter_in_sum += total_detected_hist_in.GetBinContent(bin)
# tot_tof_below_threshold = 0
# for i in range(len(bin_edge_list_low)):
#     tot_tof_below_threshold += bin_edge_list_up[i] - bin_edge_list_low[i] 

# AVG_background = N_filter_in_sum * width_bin / tot_tof_below_threshold
