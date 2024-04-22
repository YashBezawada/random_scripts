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
min_e = 0.001
max_e = 0.1
min_tof = EnergyToTOF(0.1) * 1e9
max_tof = EnergyToTOF(0.001) * 1e9

# c1 = ROOT.TCanvas()
# c1.SetLogy()

################## Extracting nTOF TOF hist
file_nTOF_TOF = ROOT.TFile.Open("/home/yash/ArtieSim/data/Neutron_Gamma_Flux_1cmColli_188m.root", "READ")
hist_canvas = file_nTOF_TOF.Get("Canvas_1")
hist_nTOF_TOF_full = hist_canvas.GetPrimitive("histfluka")
# c1 = ROOT.TCanvas()
# c1.SetLogx()
# c1.SetLogy()
# hist_nTOF_TOF_full.Draw()

nTOF_flux_hist = ROOT.TH1D("nTOF_flux_hist","Generated Neutron Energy Weights",num_bins,min_tof,max_tof)

for i in range(num_bins):
    bin_low = hist_nTOF_TOF_full.GetXaxis().FindBin( nTOF_flux_hist.GetXaxis().GetBinLowEdge(i+1) )
    bin_high = hist_nTOF_TOF_full.GetXaxis().FindBin( nTOF_flux_hist.GetXaxis().GetBinUpEdge(i+1) )
    tot_bins = bin_high - bin_low + 1
    sum_bins = 0
    for j in range(tot_bins):
        sum_bins += hist_nTOF_TOF_full.GetBinContent(bin_low + j)
    nTOF_flux_hist.SetBinContent(i+1, sum_bins)

################## Extracting Simulated energy
### Filter in
file_in = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_uniformBkgd_3milevts_5cmFilter_in_0.root")
Tree_in = file_in["bkgdAnalysis"]
Branches_in = Tree_in.arrays()
n_energy = np.array(Branches_in["n_energy"])
n_energy = n_energy[n_energy != 0]
print("Total number of generated neutrons = ", n_energy.size)
n_detected_array_in = np.array(Branches_in["n_energy_detected"])
n_detected_in = np.array( n_detected_array_in[n_detected_array_in != 0] )
bkgd_array_in = np.array(Branches_in["n_tof_bkgd"])
bkgd_events_in = np.array( bkgd_array_in[bkgd_array_in != 0] )
print("Total number of background neutrons = ", bkgd_events_in.size)

### Filter out
file_out = uproot.open("/home/yash/ArtieSim/build/outputs/ntof_uniformBkgd_3milevts_5cmFilter_out_0.root")
Tree_out = file_out["bkgdAnalysis"]
Branches_out = Tree_out.arrays()
n_detected_array_out = np.array(Branches_out["n_energy_detected"])
n_detected_out = np.array( n_detected_array_out[n_detected_array_out != 0] )
##################

### Converting neutron energy to tof
n_tof = []
for e in n_energy:
    n_tof.append(EnergyToTOF(e) * 1e9)
n_tof = np.array(n_tof)

n_detected_tof_in = []
for e in n_detected_in:
    n_detected_tof_in.append(EnergyToTOF(e) * 1e9)
n_detected_tof_in = np.array(n_detected_tof_in)

n_detected_tof_out = []
for e in n_detected_out:
    n_detected_tof_out.append(EnergyToTOF(e) * 1e9)
n_detected_tof_out = np.array(n_detected_tof_out)
###

### Making Histograms
total_evts = np.concatenate((n_tof, bkgd_events_in))
gen_tof_hist = ROOT.TH1D("gen_tof_hist","Generated Neutron + bkgd ToF",num_bins,min_tof,max_tof)
for evt in total_evts:
    gen_tof_hist.Fill(evt)

tot_detected_in = np.concatenate((n_detected_tof_in, bkgd_events_in))
total_detected_hist_in = ROOT.TH1D("total_detected_hist_in","Detected Neutron Events + Bkgd Events (Filter In)",num_bins,min_tof,max_tof)
for evt in tot_detected_in:
    total_detected_hist_in.Fill(evt)

tot_detected_out = np.concatenate((n_detected_tof_out, bkgd_events_in))
total_detected_hist_out = ROOT.TH1D("total_detected_hist_out","Detected Neutron Events + Bkgd Events (Filter Out)",num_bins,min_tof,max_tof)
for evt in tot_detected_out:
    total_detected_hist_out.Fill(evt)
###

### Calculating weights
total_evt_weights = []
for i in range(total_evts.size):
    bin_num = gen_tof_hist.GetXaxis().FindBin(total_evts[i])
    total_evt_weights.append(nTOF_flux_hist.GetBinContent(bin_num) / gen_tof_hist.GetBinContent(bin_num))
total_evt_weights = np.array(total_evt_weights)

detected_weights_in = np.array( total_evt_weights[n_detected_array_in != 0] )
bkgd_weights = np.array( total_evt_weights[bkgd_array_in != 0] )
tot_detected_weights_in = np.concatenate((detected_weights_in, bkgd_weights))

detected_weights_out = np.array( total_evt_weights[n_detected_array_out != 0] )
tot_detected_weights_out = np.concatenate((detected_weights_out, bkgd_weights))
###

### Calculating Bin uncertainties
total_evt_bin_unc = []
for i in range(num_bins):
    total_evt_bin_unc.append(nTOF_flux_hist.GetBinContent(i+1) / np.sqrt(gen_tof_hist.GetBinContent(i+1)))

det_evt_bin_unc_in = []
for i in range(num_bins):
    det_evt_bin_unc_in.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(total_detected_hist_in.GetBinContent(i+1)) / gen_tof_hist.GetBinContent(i+1) )

det_evt_bin_unc_out = []
for i in range(num_bins):
    det_evt_bin_unc_out.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(total_detected_hist_out.GetBinContent(i+1)) / gen_tof_hist.GetBinContent(i+1) )
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
counts_tot, edges_tot = np.histogram(total_evts, bins=num_bins, weights=total_evt_weights)
cbins_tot = (edges_tot[:-1] + edges_tot[1:])/2.0
plt.errorbar(cbins_tot, counts_tot, yerr=total_evt_bin_unc, fmt="ko", markersize=5)
plt.xlabel("ToF in ns")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron ToF (Weighted)")
plt.yscale("log")
plt.savefig("Plots/weighted_tot_tof_hist.png")
# plt.show()

fig1 = plt.figure(1)
counts_in, edges_in = np.histogram(tot_detected_in, bins=num_bins, weights=tot_detected_weights_in)
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=det_evt_bin_unc_in, fmt="ko", markersize=5)
plt.xlabel("ToF in ns")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron ToF - Filter In (Weighted)")
plt.yscale("log")
plt.savefig("Plots/weighted_det_tof_filter_in.png")
# plt.show()

fig2 = plt.figure(2)
counts_out, edges_out = np.histogram(tot_detected_out, bins=num_bins, weights=tot_detected_weights_out)
cbins_out = (edges_out[:-1] + edges_out[1:])/2.0
plt.errorbar(cbins_out, counts_out, yerr=det_evt_bin_unc_out, fmt="ko", markersize=5)
plt.xlabel("ToF in ns")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron ToF - Filter Out (Weighted)")
plt.yscale("log")
plt.savefig("Plots/weighted_det_tof_filter_out.png")

### Transmision Plot
fig3 = plt.figure(3)
counts_trans = counts_in / counts_out

trans_unc = []
for i in range(counts_trans.size):
    a = det_evt_bin_unc_in[i] / counts_in[i]
    b = det_evt_bin_unc_out[i] / counts_out[i]
    trans_unc.append( counts_trans[i] * np.sqrt(a*a + b*b) )
trans_unc = np.array(trans_unc)
plt.errorbar(cbins_in, counts_trans, yerr=trans_unc, fmt="ko", markersize=5)
plt.xlabel("ToF in ns")
plt.ylabel("Neutron Count")
plt.title("Transmission Histogram (5cm Al-27 filter)")
plt.savefig("Plots/sim_transmission_hist.png")

plt.yscale("log")
plt.savefig("Plots/sim_transmission_hist_logy.png")