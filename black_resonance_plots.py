import numpy as np
import matplotlib.pyplot as plt

outputFile = np.load('black_resonance_output.npz')
num_bins = outputFile['num_bins']
n_energy_tot = outputFile['n_energy_tot']
n_detected_in = outputFile['n_detected_in']
n_detected_out = outputFile['n_detected_out']
total_evt_weights = outputFile['total_evt_weights']
detected_weights_in = outputFile['detected_weights_in']
detected_weights_out = outputFile['detected_weights_out']
weighted_total_evt_binUnc = outputFile['weighted_total_evt_binUnc']
weighted_det_in_binUnc = outputFile['weighted_det_in_binUnc']
weighted_det_out_binUnc = outputFile['weighted_det_out_binUnc']
total_evt_binUnc = outputFile['total_evt_binUnc']
detected_in_binUnc = outputFile['detected_in_binUnc']
detected_out_binUnc = outputFile['detected_out_binUnc']

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

i = 0

plt.figure(i)
counts_tot, edges_tot = np.histogram(n_energy_tot, bins=num_bins, weights=total_evt_weights) # , range=[0, 0.1]
cbins_tot = (edges_tot[:-1] + edges_tot[1:])/2.0
plt.errorbar(cbins_tot, counts_tot, yerr=weighted_total_evt_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Generated Neutron Energy (Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/blackRes/weighted_totalE_hist.png")
# plt.show()

i+=1

plt.figure(i)
counts_in, edges_in = np.histogram(n_detected_in, bins=num_bins, weights=detected_weights_in) # , range=[0, 0.1]
cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
plt.errorbar(cbins_in, counts_in, yerr=weighted_det_in_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron Energy Filter In - Al 8 cm (Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/blackRes/weighted_detectedE_hist_filter_in.png")
# plt.show()

i+=1

plt.figure(i)
counts_out, edges_out = np.histogram(n_detected_out, bins=num_bins, weights=detected_weights_out)
cbins_out = (edges_out[:-1] + edges_out[1:])/2.0
plt.errorbar(cbins_out, counts_out, yerr=weighted_det_out_binUnc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Detected Neutron Energy Filter Out (Weighted)")
# plt.ylim(bottom=90)
# plt.xscale("log")
plt.yscale("log")
plt.savefig("Plots/blackRes/weighted_detectedE_hist_filter_out.png")

# i+=1

# plt.figure(i)
# counts_in, edges_in = np.histogram(n_detected_in, bins=num_bins, weights=detected_weights_in) # , range=[0, 0.1]
# cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
# plt.errorbar(cbins_in, counts_in, yerr=weighted_det_in_binUnc, fmt="ko", markersize=5)
# plt.xlabel("Energy (MeV)")
# plt.ylabel("Neutron Count")
# plt.title("Detected Neutron Energy Filter In - Al 8 cm (Weighted)")
# # plt.ylim(bottom=90)
# # plt.xscale("log")
# plt.yscale("log")
# plt.savefig("Plots/blackRes/weighted_detectedE_hist_filter_in.png")
# # plt.show()

# i+=1

# plt.figure(i)
# counts_out, edges_out = np.histogram(n_detected_out, bins=num_bins, weights=detected_weights_out)
# cbins_out = (edges_out[:-1] + edges_out[1:])/2.0
# plt.errorbar(cbins_out, counts_out, yerr=weighted_det_out_binUnc, fmt="ko", markersize=5)
# plt.xlabel("Energy (MeV)")
# plt.ylabel("Neutron Count")
# plt.title("Detected Neutron Energy Filter Out (Weighted)")
# # plt.ylim(bottom=90)
# # plt.xscale("log")
# plt.yscale("log")
# plt.savefig("Plots/blackRes/weighted_detectedE_hist_filter_out.png")

# i+=1

# plt.figure(i)
# counts_tot, edges_tot = np.histogram(n_energy_tot, bins=num_bins)
# cbins_tot = (edges_tot[:-1] + edges_tot[1:])/2.0
# plt.errorbar(cbins_tot, counts_tot, yerr=total_evt_binUnc, fmt="ko", markersize=5)
# plt.xlabel("Energy (keV)")
# plt.ylabel("Neutron Count")
# plt.title("Generated Neutron Energy")
# # plt.ylim(bottom=8000, top=50000)
# # plt.xscale("log")
# plt.yscale("log")
# plt.savefig("Plots/blackRes/totalE_hist.png")
# # plt.show()

# i+=1

# plt.figure(i)
# counts_in, edges_in = np.histogram(n_detected_in, bins=num_bins)
# cbins_in = (edges_in[:-1] + edges_in[1:])/2.0
# plt.errorbar(cbins_in, counts_in, yerr=detected_in_binUnc, fmt="ko", markersize=5)
# plt.xlabel("Energy (keV)")
# plt.ylabel("Neutron Count")
# plt.title("Detected Neutron Energy Filter In")
# # plt.ylim(bottom=90, top=1000)
# # plt.xscale("log")
# plt.yscale("log")
# plt.savefig("Plots/blackRes/detectedE_hist.png")
# # plt.show()

i+=1

plt.figure(i)
counts_trans = counts_in / counts_out

trans_unc = []
for i in range(counts_trans.size):
    a = weighted_det_in_binUnc[i] / counts_in[i]
    b = weighted_det_out_binUnc[i] / counts_out[i]
    trans_unc.append( counts_trans[i] * np.sqrt(a*a + b*b) )
trans_unc = np.array(trans_unc)
plt.errorbar(cbins_in, counts_trans, yerr=trans_unc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Transmission Histogram (8cm Al-27 filter)")
plt.savefig("Plots/blackRes/sim_transmission_hist.png")

i+=1

plt.figure(i)
plt.errorbar(cbins_in, counts_trans, yerr=trans_unc, fmt="ko", markersize=5)
plt.xlabel("Energy (MeV)")
plt.ylabel("Neutron Count")
plt.title("Transmission Histogram (8cm Al-27 filter)")
plt.yscale("log")
plt.savefig("Plots/blackRes/sim_transmission_hist_logy.png")