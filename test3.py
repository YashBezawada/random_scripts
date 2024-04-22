import numpy as np
import ROOT

#Artie data (in KeV and barns)
fileName = ROOT.TFile.Open("/home/yash/Downloads/Artie-plots.root", "READ")
h1 = fileName.Get("grXsec_vs_energy_from_tof_wo_bkg_total_error")
x_artie_all = np.array(h1.GetX())
y_artie_all = np.array(h1.GetY())
total_pts = h1.GetN()

x_artie_err_low = []
x_artie_err_up = []
y_artie_err_low = []
y_artie_err_up = []

for i in range(0,total_pts):
    if (h1.GetPointX(i) < 70):
        if (h1.GetPointX(i) < 20):
            break
        x_artie_err_low.append(h1.GetErrorXlow(i))
        x_artie_err_up.append(h1.GetErrorXhigh(i))
        y_artie_err_low.append(h1.GetErrorYlow(i))
        y_artie_err_up.append(h1.GetErrorYhigh(i))

x_artie_err_low = np.array(x_artie_err_low)
x_artie_err_up = np.array(x_artie_err_up)
y_artie_err_low = np.array(y_artie_err_low)
y_artie_err_up = np.array(y_artie_err_up)

x_artie = np.array([e for e in x_artie_all if (e >= 20) and (e <= 70)])
mask = np.array([True if (e >= 20) and (e <= 70) else False for e in x_artie_all])
y_artie = y_artie_all[mask]

headers = ['Energy (keV)', 'Cross Section (B)', 'E err low (keV)', 'E err high (keV)', 'xsec err low (B)', 'xsec err high (B)']
file_name = "artie_published_data.csv"
header_string = ",".join(headers)

stacked_array = np.column_stack((x_artie, y_artie, x_artie_err_low, x_artie_err_up, y_artie_err_low, y_artie_err_up))
np.savetxt(file_name, stacked_array, delimiter=",", header=header_string)
