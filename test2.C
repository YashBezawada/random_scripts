#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

#include "TFile.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TFrame.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TLine.h"
#include "TAxis.h"
#include "TColor.h"
#include "TAttMarker.h"
#include "TRandom3.h"

void test2(){
    TFile *file = TFile::Open("/home/yash/Downloads/Artie-plots.root","read");
    TGraphAsymmErrors* artie_graph = (TGraphAsymmErrors*)file->Get("grXsec_vs_energy_from_tof_wo_bkg_total_error");

    // Double_t* y_err_low = artie_graph->GetEYlow();
    Double_t x_err_high = artie_graph->GetErrorXhigh(0);
    Double_t x_err_low = artie_graph->GetErrorXlow(0);
    Double_t y_err_high = artie_graph->GetErrorYhigh(0);
    Double_t y_err_low = artie_graph->GetErrorYlow(0);

    cout << x_err_high << endl;
    cout << x_err_low << endl;
    cout << y_err_high << endl;
    cout << y_err_low << endl;

    // Int_t j =0;

    // TCanvas *c[1];
    // c[j] = new TCanvas(Form("c%d", j)," ");
    // c[j]->cd();
    // c[j]->Draw();

    // artie_graph->GetYaxis()->SetRangeUser(1e-3,1e3);
    // artie_graph->Draw();
}