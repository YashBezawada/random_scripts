import csv
import numpy as np
import ROOT
import uproot

file = uproot.recreate("moderator_hist.root")

with open('DICER_MARKIV_1.0.csv') as csv_file:

    time = []
    pdf = []

    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        pdf.append(float(row[1]))
        time.append(float(row[0]))

    pdf_np = np.array(pdf)
    time_np = np.array(time)

    bins = (time[-1]-time[0])/0.05 + 1
    min = time[0] - 0.025
    max = time[-1] + 0.025


    h1 = ROOT.TH1F("h1","time pdf hist",int(bins),min,max)

    for i in range(int(bins)):
        h1.SetBinContent(i, pdf_np[i])

    file["hist_1ev"] = h1