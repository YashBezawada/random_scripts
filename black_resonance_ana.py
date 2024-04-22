import numpy as np
import uproot
import ROOT
import matplotlib.pyplot as plt

def EnergyToTOF(e):
    KE_M = e/939.56542052
    denominator = 1.0 - 1.0/((KE_M + 1.0)*(KE_M + 1.0))
    correction_factor = np.sqrt(1.0 / denominator)
    return (184/299792458) * correction_factor

def TOFToEnergy(t):
    denom_term = (184)/(299792458 * t) #t is in seconds
    denominator = 1.0 - (denom_term * denom_term)
    factor = np.sqrt(1.0 / denominator) - 1
    return 939.56542052 * factor  #e in MeV

num_bins = 50
min_e = 0.001 # MeV
max_e = 0.1 # MeV
pin_hole_rad = 0.005 #m
pin_hole_rad_square = pin_hole_rad * pin_hole_rad
# min_tof = EnergyToTOF(1000) * 1e9
# max_tof = EnergyToTOF(0.000000001) * 1e9

# c1 = ROOT.TCanvas()
# c1.SetLogy()

################## Extracting nTOF Energy hist
file_nTOF = ROOT.TFile.Open("/home/yash/ArtieSim/data/evalflux.root", "READ")
nTOF_flux_hist_full = file_nTOF.Get("hEval_Abs") # Energy is in eV in this histogram

# c1 = ROOT.TCanvas()
# c1.SetLogx()
# c1.SetLogy()
# hist_nTOF_TOF_full.Draw()

nTOF_flux_hist = ROOT.TH1D("nTOF_flux_hist","Generated Neutron Energy Weights",num_bins,min_e,max_e)

for i in range(num_bins):
    low_edge = nTOF_flux_hist.GetXaxis().GetBinLowEdge(i+1) * 1e6 # 1e6: converting MeV to eV
    high_edge = nTOF_flux_hist.GetXaxis().GetBinUpEdge(i+1) * 1e6 # 1e6: converting MeV to eV
    bin_low = nTOF_flux_hist_full.GetXaxis().FindBin( low_edge ) 
    binLow_low = nTOF_flux_hist_full.GetXaxis().GetBinLowEdge(bin_low)
    binLow_high = nTOF_flux_hist_full.GetXaxis().GetBinUpEdge(bin_low)
    bin_high = nTOF_flux_hist_full.GetXaxis().FindBin( high_edge ) 
    binHigh_low = nTOF_flux_hist_full.GetXaxis().GetBinLowEdge(bin_high)
    binHigh_high = nTOF_flux_hist_full.GetXaxis().GetBinUpEdge(bin_high)
    tot_bins = bin_high - bin_low + 1
    sum_bins = 0
    for j in range(tot_bins):
        if tot_bins == 1:
            sum_bins += (high_edge - low_edge) * nTOF_flux_hist_full.GetBinContent(bin_low + j) / (binLow_high - binLow_low)
            break
        if j == 0:
            sum_bins += (binLow_high - low_edge) * nTOF_flux_hist_full.GetBinContent(bin_low + j) / (binLow_high - binLow_low)
            continue
        if j == tot_bins - 1:
            sum_bins += (high_edge - binHigh_low) * nTOF_flux_hist_full.GetBinContent(bin_low + j) / (binHigh_high - binHigh_low)
            continue
        sum_bins += nTOF_flux_hist_full.GetBinContent(bin_low + j)
    nTOF_flux_hist.SetBinContent(i+1, sum_bins * 1e-6 ) # 1e-6: converting eV to MeV

################## Extracting Simulated energy
myFile_in = uproot.open("/home/yash/marex_sim/build/outputs/ntof_5milevts_1_100keV_gasAr_filter_Al_8cm_0.root")
Tree_in = myFile_in["bkgdAnalysis"]
Branches_in = Tree_in.arrays()
n_energy = np.array(Branches_in["n_energy"])
print("Total number of generated neutrons = ", n_energy.size)
n_detectionStatus_in = np.array(Branches_in["detectionStatus"])
n_bkgd_tof_in = np.array(Branches_in["n_tof_bkgd"])
n_bkgd_tof_in = np.array( n_bkgd_tof_in[n_bkgd_tof_in != 0] )

myFile_out = uproot.open("/home/yash/marex_sim/build/outputs/ntof_5milevts_1_100keV_gasAr_filter_out_0.root")
Tree_out = myFile_out["bkgdAnalysis"]
Branches_out = Tree_out.arrays()
n_detectionStatus_out = np.array(Branches_out["detectionStatus"])
# n_bkgd_tof_out = np.array(Branches_out["n_tof_bkgd"])
# n_bkgd_tof_out = np.array( n_bkgd_tof_out[n_bkgd_tof_out != 0] )

### Concatenating the arrays
n_energy_gen = np.array( n_energy[n_energy != 0])

n_bkgd_in = []
for t in n_bkgd_tof_in:
    n_bkgd_in.append(TOFToEnergy(t * 1e-9))  
n_bkgd_in = np.array(n_bkgd_in)

# n_bkgd_out = []
# for t in n_bkgd_tof_out:
#     n_bkgd_out.append(TOFToEnergy(t * 1e-9))  
# n_bkgd_out = np.array(n_bkgd_out)

n_energy_tot = np.concatenate((n_energy_gen, n_bkgd_in), axis=0)

n_detected_in = np.array( n_energy_tot[n_detectionStatus_in == True] )
n_detected_out = np.array( n_energy_tot[n_detectionStatus_out == True] )

### Making Histograms
gen_e_hist = ROOT.TH1D("gen_e_hist","Generated Neutron Energy",num_bins,min_e,max_e)
for evt in n_energy_tot:
    gen_e_hist.Fill(evt)

total_detected_hist_in = ROOT.TH1D("total_detected_hist_in","Detected Neutron Events - Filter in",num_bins,min_e,max_e)
for evt in n_detected_in:
    total_detected_hist_in.Fill(evt)

total_detected_hist_out = ROOT.TH1D("total_detected_hist_out","Detected Neutron Events - Filter out",num_bins,min_e,max_e)
for evt in n_detected_out:
    total_detected_hist_out.Fill(evt)
###

### Calculating weights
total_evt_weights = []
for i in range(n_energy_tot.size):
    bin_num = gen_e_hist.GetXaxis().FindBin(n_energy_tot[i])
    total_evt_weights.append(nTOF_flux_hist.GetBinContent(bin_num) / gen_e_hist.GetBinContent(bin_num))
total_evt_weights = np.array(total_evt_weights)

detected_weights_in = np.array( total_evt_weights[n_detectionStatus_in != 0] )
detected_weights_out = np.array( total_evt_weights[n_detectionStatus_out != 0] )
###

### Calculating Bin uncertainties
weighted_total_evt_binUnc = []
for i in range(num_bins):
    weighted_total_evt_binUnc.append(nTOF_flux_hist.GetBinContent(i+1) / np.sqrt(gen_e_hist.GetBinContent(i+1)))
weighted_total_evt_binUnc = np.array(weighted_total_evt_binUnc)

weighted_det_in_binUnc = []
for i in range(num_bins):
    weighted_det_in_binUnc.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(total_detected_hist_in.GetBinContent(i+1)) / gen_e_hist.GetBinContent(i+1) )
weighted_det_in_binUnc = np.array(weighted_det_in_binUnc)

weighted_det_out_binUnc = []
for i in range(num_bins):
    weighted_det_out_binUnc.append( nTOF_flux_hist.GetBinContent(i+1) * np.sqrt(total_detected_hist_out.GetBinContent(i+1)) / gen_e_hist.GetBinContent(i+1) )
weighted_det_out_binUnc = np.array(weighted_det_out_binUnc)

total_evt_binUnc = []
for i in range(num_bins):
    total_evt_binUnc.append(np.sqrt(gen_e_hist.GetBinContent(i+1)))
total_evt_binUnc = np.array(total_evt_binUnc)

detected_in_binUnc = []
for i in range(num_bins):
    detected_in_binUnc.append(np.sqrt(total_detected_hist_in.GetBinContent(i+1)))
detected_in_binUnc = np.array(detected_in_binUnc)

detected_out_binUnc = []
for i in range(num_bins):
    detected_out_binUnc.append(np.sqrt(total_detected_hist_out.GetBinContent(i+1)))
detected_out_binUnc = np.array(detected_out_binUnc)
###

### Saving arrays to the output file
np.savez('black_resonance_output',
         num_bins=num_bins,
         n_energy_tot=n_energy_tot,
         n_detected_in=n_detected_in,
         n_detected_out=n_detected_out,
         total_evt_weights=total_evt_weights,
         detected_weights_in=detected_weights_in,
         detected_weights_out=detected_weights_out,
         weighted_total_evt_binUnc=weighted_total_evt_binUnc,
         weighted_det_in_binUnc=weighted_det_in_binUnc,
         weighted_det_out_binUnc=weighted_det_out_binUnc,
         total_evt_binUnc=total_evt_binUnc,
         detected_in_binUnc=detected_in_binUnc,
         detected_out_binUnc=detected_out_binUnc
         )
