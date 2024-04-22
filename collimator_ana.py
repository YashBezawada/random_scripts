import numpy as np
import uproot
import ROOT
import matplotlib.pyplot as plt

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

num_bins = 100
min_e = 0.000000001
max_e = 20.0
pin_hole_rad = 0.005 #m
pin_hole_rad_square = pin_hole_rad * pin_hole_rad
# min_tof = EnergyToTOF(1000) * 1e9
# max_tof = EnergyToTOF(0.000000001) * 1e9

# c1 = ROOT.TCanvas()
# c1.SetLogy()

################## Extracting nTOF TOF hist
file_nTOF = ROOT.TFile.Open("/home/yash/ArtieSim/data/evalflux.root", "READ")
nTOF_flux_hist_full = file_nTOF.Get("hEval_Abs")

# c1 = ROOT.TCanvas()
# c1.SetLogx()
# c1.SetLogy()
# hist_nTOF_TOF_full.Draw()

nTOF_flux_hist = ROOT.TH1D("nTOF_flux_hist","Generated Neutron Energy Weights",num_bins,min_e,max_e)

for i in range(num_bins):
    low_edge = nTOF_flux_hist.GetXaxis().GetBinLowEdge(i+1)
    high_edge = nTOF_flux_hist.GetXaxis().GetBinUpEdge(i+1)
    bin_low = nTOF_flux_hist_full.GetXaxis().FindBin( low_edge )
    binLow_low = nTOF_flux_hist_full.GetXaxis().GetBinLowEdge(bin_low)
    binLow_high = nTOF_flux_hist_full.GetXaxis().GetBinUpEdge(bin_low)
    bin_high = nTOF_flux_hist_full.GetXaxis().FindBin( high_edge )
    binHigh_low = nTOF_flux_hist_full.GetXaxis().GetBinLowEdge(bin_high)
    binHigh_high = nTOF_flux_hist_full.GetXaxis().GetBinUpEdge(bin_high)
    tot_bins = bin_high - bin_low + 1
    sum_bins = 0
    for j in range(tot_bins):
        if j == 0:
            sum_bins += (binLow_high - low_edge) * nTOF_flux_hist_full.GetBinContent(bin_low + j) / (binLow_high - binLow_low)
            continue
        if j == tot_bins - 1:
            sum_bins += (high_edge - binHigh_low) * nTOF_flux_hist_full.GetBinContent(bin_low + j) / (binHigh_high - binHigh_low)
            continue
        sum_bins += nTOF_flux_hist_full.GetBinContent(bin_low + j)
    nTOF_flux_hist.SetBinContent(i+1, sum_bins)

################## Extracting Simulated energy
myFile = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_gasAr_brassCol_20cm_5mm_uniformE_0.root")
Tree = myFile["bkgdAnalysis"]
Branches = Tree.arrays()
n_energy = np.array(Branches["n_energy"])
print("Total number of generated neutrons = ", n_energy.size)

n_detected_full = np.array(Branches["n_energy_detected"])
n_detected = np.array( n_detected_full[n_detected_full != 0] )

x_start = np.array(Branches["x_start"])
x_start_detected = np.array( x_start[n_detected_full != 0] )

y_start = np.array(Branches["y_start"])
y_start_detected = np.array( y_start[n_detected_full != 0] )

### Neutrons passing through collimator
n_total_passColl_array = []
n_detected_passColl_array = [] # This is useful while calculating weights
for i in range(n_detected_full.size):
    x_det = x_start[i]
    y_det = y_start[i]
    if ( (x_det * x_det + y_det * y_det) > pin_hole_rad_square ):
        n_total_passColl_array.append(n_energy[i])
        if(n_detected_full[i] == 0):
            n_detected_passColl_array.append(0)
            continue
        n_detected_passColl_array.append(n_detected_full[i])
        continue
    n_total_passColl_array.append(0)
    n_detected_passColl_array.append(0)
n_total_passColl_array = np.array(n_total_passColl_array)
n_detected_passColl_array = np.array(n_detected_passColl_array)
n_total_passColl = np.array( n_total_passColl_array[n_total_passColl_array != 0] )
n_detected_passColl = np.array( n_detected_passColl_array[n_detected_passColl_array != 0] )

print("Number of neutrons passing through collimator = ", n_total_passColl.size)
print("Number of neutrons detected after passing through collimator = ", n_detected_passColl.size)
##################

### Making Histograms
gen_tof_hist = ROOT.TH1D("gen_tof_hist","Generated Neutron Energy",num_bins,min_e,max_e)
for evt in n_energy:
    gen_tof_hist.Fill(evt)

total_detected_hist = ROOT.TH1D("total_detected_hist","Detected Neutron Events",num_bins,min_e,max_e)
for evt in n_detected:
    total_detected_hist.Fill(evt)

n_total_passColl_hist = ROOT.TH1D("n_total_passColl_hist","Neutrons passing through collimator",num_bins,min_e,max_e)
for evt in n_total_passColl:
    n_total_passColl_hist.Fill(evt)

n_detected_passColl_hist = ROOT.TH1D("n_detected_passColl_hist","Neutrons detected after passing through collimator",num_bins,min_e,max_e)
for evt in n_detected_passColl:
    n_detected_passColl_hist.Fill(evt)
###

### Calculating weights
total_evt_weights = []
for i in range(n_energy.size):
    bin_num = gen_tof_hist.GetXaxis().FindBin(n_energy[i])
    total_evt_weights.append(nTOF_flux_hist.GetBinContent(bin_num) / gen_tof_hist.GetBinContent(bin_num))
total_evt_weights = np.array(total_evt_weights)

detected_weights = np.array( total_evt_weights[n_detected_full != 0] )

total_passColl_weights = np.array( total_evt_weights[n_total_passColl_array != 0] )

detected_passColl_weights = np.array( total_evt_weights[n_detected_passColl_array != 0] )
###

### Calculating Bin uncertainties
weighted_total_evt_binUnc = []
for i in range(num_bins):
    weighted_total_evt_binUnc.append(nTOF_flux_hist.GetBinContent(i+1) / np.sqrt(gen_tof_hist.GetBinContent(i+1)))

weighted_det_evt_binUnc = []
for i in range(num_bins):
    weighted_det_evt_binUnc.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(total_detected_hist.GetBinContent(i+1)) / gen_tof_hist.GetBinContent(i+1) )

weighted_total_passColl_binUnc = []
for i in range(num_bins):
    weighted_total_passColl_binUnc.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(n_total_passColl_hist.GetBinContent(i+1)) / gen_tof_hist.GetBinContent(i+1) )

weighted_det_passColl_binUnc = []
for i in range(num_bins):
    weighted_det_passColl_binUnc.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(n_detected_passColl_hist.GetBinContent(i+1)) / gen_tof_hist.GetBinContent(i+1) )

total_evt_binUnc = []
for i in range(num_bins):
    total_evt_binUnc.append(np.sqrt(gen_tof_hist.GetBinContent(i+1)))

det_evt_binUnc = []
for i in range(num_bins):
    det_evt_binUnc.append(np.sqrt(total_detected_hist.GetBinContent(i+1)))

total_passColl_binUnc = []
for i in range(num_bins):
    total_passColl_binUnc.append(np.sqrt(n_total_passColl_hist.GetBinContent(i+1)))

det_passColl_binUnc = []
for i in range(num_bins):
    det_passColl_binUnc.append(np.sqrt(n_detected_passColl_hist.GetBinContent(i+1)))

###

################################# Plotting

# nTOF_flux_hist.GetXaxis().SetTitle("ToF in ns")
# nTOF_flux_hist.GetYaxis().SetTitle("Neutron Count")
# c1.cd()
# nTOF_flux_hist.Draw()
# c1.Print("Plots/nTOF_flux_hist.png")

# gen_tof_hist.GetXaxis().SetTitle("ToF in ns")
# gen_tof_hist.GetYaxis().SetTitle("Neutron Count")
# gen_tof_hist.Draw()
# c1.Print("Plots/gen_tof_hist.png")

fig0 = plt.figure(0)
counts_tot, edges_tot = np.histogram(n_energy, bins=num_bins, weights=total_evt_weights)
cbins_tot = (edges_tot[:-1] + edges_tot[1:])/2.0
plt.errorbar(cbins_tot, counts_tot, yerr=weighted_total_evt_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron Energy (Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/weighted_totalE_hist.png")
# plt.show()

fig1 = plt.figure(1)
counts_in, edges_in = np.histogram(n_detected, bins=num_bins, weights=detected_weights)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=weighted_det_evt_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron Energy (Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/weighted_detectedE_hist.png")
# plt.show()

fig2 = plt.figure(2)
counts_tot, edges_tot = np.histogram(n_energy, bins=num_bins)
cbins_tot = (edges_tot[:-1] + edges_tot[1:])/2.0
plt.errorbar(cbins_tot, counts_tot, yerr=total_evt_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron Energy")
plt.ylim(bottom=8000, top=50000)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/totalE_hist.png")
# plt.show()

fig3 = plt.figure(3)
counts_in, edges_in = np.histogram(n_detected, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=det_evt_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron Energy")
plt.ylim(bottom=90, top=1000)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/detectedE_hist.png")
# plt.show()

fig4 = plt.figure(4)
counts_in, edges_in = np.histogram(x_start, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.plot(cbins_in, counts_in, "ko", markersize=5)
plt.xlabel("x (m)")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron X Distribution")
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/gen_x_dis.png")
# plt.show()

fig5 = plt.figure(5)
counts_in, edges_in = np.histogram(y_start, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.plot(cbins_in, counts_in, "ko", markersize=5)
plt.xlabel("y (m)")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron Y Distribution")
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/gen_y_dis.png")
# plt.show()

fig6 = plt.figure(6)
counts_in, edges_in = np.histogram(x_start_detected, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.plot(cbins_in, counts_in, "ko", markersize=5)
plt.xlabel("x (m)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron X Distribution")
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/detected_x_dis.png")
# plt.show()

fig7 = plt.figure(7)
counts_in, edges_in = np.histogram(y_start_detected, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.plot(cbins_in, counts_in, "ko", markersize=5)
plt.xlabel("y (m)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron Y Distribution")
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/collAna/detected_y_dis.png")
# plt.show()

fig8 = plt.figure(8)
counts_in, edges_in = np.histogram(n_total_passColl, bins=num_bins, weights=total_passColl_weights)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=weighted_total_passColl_binUnc, fmt="ko", markersize=5, label='Total')

counts_in, edges_in = np.histogram(n_detected_passColl, bins=num_bins, weights=detected_passColl_weights)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=weighted_det_passColl_binUnc, fmt="ro", markersize=5, label='Detected')

plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Neutrons passing through collimator (r = 5 mm, Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("Plots/collAna/weighted_n_passColl_hist.png")
# plt.show()

fig9 = plt.figure(9)
counts_in, edges_in = np.histogram(n_total_passColl, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=total_passColl_binUnc, fmt="ko", markersize=5, label='Total')

counts_in, edges_in = np.histogram(n_detected_passColl, bins=num_bins)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=det_passColl_binUnc, fmt="ro", markersize=5, label='Detected')

plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Neutrons passing through collimator (r = 5 mm)")
# plt.ylim(bottom=90, top=1000)
# plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("Plots/collAna/n_passColl_hist.png")
# plt.show()