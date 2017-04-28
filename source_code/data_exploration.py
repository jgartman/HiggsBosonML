# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:55:50 2017

@author: Josh
"""
import pandas
import pickle
import matplotlib.pyplot as plt

#data = pandas.read_csv("../higgs_data/atlas-higgs-challenge-2014-v2.csv")
#
#s = data[data["Label"] == "s"]
#b = data[data["Label"] == "b"]
#
#high_level_features = [
# 'DER_mass_MMC',
# 'DER_mass_transverse_met_lep',
# 'DER_mass_vis',
# 'DER_pt_h',
# 'DER_deltar_tau_lep',
# 'DER_pt_tot',
# 'DER_sum_pt',
# 'DER_pt_ratio_lep_tau',
# 'DER_met_phi_centrality',
# 'DER_lep_eta_centrality']
#
#low_level_features = [
# 'PRI_tau_pt',
# 'PRI_tau_eta',
# 'PRI_tau_phi',
# 'PRI_lep_pt',
# 'PRI_lep_eta',
# 'PRI_lep_phi',
# 'PRI_met',
# 'PRI_met_phi',
# 'PRI_met_sumet',
# 'PRI_jet_num',
# 'PRI_jet_leading_pt',
# 'PRI_jet_all_pt']
#
#plt.hist(s[s["DER_mass_MMC"] != -999]["DER_mass_MMC"],bins=50,histtype="step",range=(0,500),color="blue",normed=True)
#plt.hist(b[b["DER_mass_MMC"] != -999]["DER_mass_MMC"],bins=50,histtype="step",range=(0,500),color="green",normed=True)
#
#plt.title('Estimated mass of Higgs boson candidate')
#plt.xlabel('GeV')
#plt.savefig('../plots/DER_mass_MMC.png')
#plt.show()
#
#plt.hist(s["DER_mass_transverse_met_lep"], bins=50,range=(0,300),histtype="step",color="blue",normed=True)
#plt.hist(b["DER_mass_transverse_met_lep"], bins=50,range=(0,300),histtype="step",color="green",normed=True)
#
#plt.title('Transverse mass')
#plt.xlabel('GeV')
#plt.savefig('../plots/DER_mass_transverse_met_lep.png')
#plt.show()
#
#plt.hist(s["DER_sum_pt"], bins=50,range=(0,500),histtype="step",color="blue",normed=True)
#plt.hist(b["DER_sum_pt"], bins=50,range=(0,500),histtype="step",color="green",normed=True)
#
#plt.title('Sum of modulii of transverse momenta')
#plt.xlabel('GeV')
#plt.savefig('../plots/DER_sum_pt.png')
#plt.show()
#
#plt.hist(s["PRI_tau_pt"], bins=50,range=(0,500),histtype="step",color="blue",normed=True)
#plt.hist(b["PRI_tau_pt"], bins=50,range=(0,500),histtype="step",color="green",normed=True)
#
#plt.title('Transverse momenta of the tau')
#plt.xlabel('GeV')
#plt.savefig('../plots/PRI_tau_pt.png')
#plt.show()
#
#plt.hist(s["PRI_lep_pt"], bins=50,range=(0,500),histtype="step",color="blue",normed=True)
#plt.hist(b["PRI_lep_pt"], bins=50,range=(0,500),histtype="step",color="green",normed=True)
#
#plt.title('Transverse momenta of the lepton (electron or muon)')
#plt.xlabel('GeV')
#plt.savefig('../plots/PRI_lep_pt.png')
#plt.show()
#
#plt.hist(s["PRI_met"], bins=50,range=(0,200),histtype="step",color="blue",normed=True)
#plt.hist(b["PRI_met"], bins=50,range=(0,200),histtype="step",color="green",normed=True)
#
#plt.title('Missing transverse energy')
#plt.xlabel('GeV')
#plt.savefig('../plots/PRI_met.png')
#plt.show()

#der_train_accuracy_f = open('./output/2017-04-23-13-57-35-390150/train_accuracy.p')
#pri_train_accuracy_f = open('./output/2017-04-23-20-02-26-675737/train_accuracy.p')
#all_train_accuracy_f = open('./output/2017-04-24-02-24-35-708578/train_accuracy.p')
#
#der_test_accuracy_f = open('./output/2017-04-23-13-57-35-390150/test_accuracy.p')
#pri_test_accuracy_f = open('./output/2017-04-23-20-02-26-675737/test_accuracy.p')
#all_test_accuracy_f = open('./output/2017-04-24-02-24-35-708578/test_accuracy.p')
#
#der_accuracy_train = pickle.load(der_train_accuracy_f)
#pri_accuracy_train = pickle.load(pri_train_accuracy_f)
#all_accuracy_train = pickle.load(all_train_accuracy_f)
#
#der_accuracy_test = pickle.load(der_test_accuracy_f)
#pri_accuracy_test = pickle.load(pri_test_accuracy_f)
#all_accuracy_test = pickle.load(all_test_accuracy_f)
#
#plt.plot(range(len(der_accuracy_test)), der_accuracy_test, color='red')
#plt.plot(range(len(pri_accuracy_test)), pri_accuracy_test, color='blue')
#plt.plot(range(len(all_accuracy_test)), all_accuracy_test, color='green')
#plt.ylim([.70,.84])
#plt.xlim([0,2000])
#plt.xlabel('Training Epochs')
#plt.ylabel('Accuracy')
#plt.savefig('../plots/accuracy.png')
#
#
#der_test_auc_f = open('./output/2017-04-23-13-57-35-390150/test_auc.p')
#pri_test_auc_f = open('./output/2017-04-23-20-02-26-675737/test_auc.p')
#all_test_auc_f = open('./output/2017-04-24-02-24-35-708578/test_auc.p')
#
#der_auc_test = pickle.load(der_test_auc_f)
#pri_auc_test = pickle.load(pri_test_auc_f)
#all_auc_test = pickle.load(all_test_auc_f)

#plt.plot(range(len(der_accuracy)), pri_accuracy, color='green')
#plt.plot(range(len(der_accuracy)), all_accuracy, color='red')
#plt.ylim([.70,.84])

der_train_accuracy_f = open('./output/2017-04-26-01-58-34-114255/train_accuracy.p')
pri_train_accuracy_f = open('./output/2017-04-26-10-16-41-095791/train_accuracy.p')
all_train_accuracy_f = open('./output/2017-04-26-18-51-54-310030/train_accuracy.p')

der_test_accuracy_f = open('./output/2017-04-26-01-58-34-114255/test_accuracy.p')
pri_test_accuracy_f = open('./output/2017-04-26-10-16-41-095791/test_accuracy.p')
all_test_accuracy_f = open('./output/2017-04-26-18-51-54-310030/test_accuracy.p')

der_accuracy_train = pickle.load(der_train_accuracy_f)
pri_accuracy_train = pickle.load(pri_train_accuracy_f)
all_accuracy_train = pickle.load(all_train_accuracy_f)

der_accuracy_test = pickle.load(der_test_accuracy_f)
pri_accuracy_test = pickle.load(pri_test_accuracy_f)
all_accuracy_test = pickle.load(all_test_accuracy_f)

plt.plot(range(len(der_accuracy_test)), der_accuracy_test, color='red')
plt.plot(range(len(pri_accuracy_test)), pri_accuracy_test, color='blue')
plt.plot(range(len(all_accuracy_test)), all_accuracy_test, color='green')
plt.ylim([.70,.84])
plt.xlim([0,2000])
plt.xlabel('Training Epochs')
plt.ylabel('Accuracy')
plt.savefig('../plots/accuracy_nn.png')


der_test_auc_f = open('./output/2017-04-26-01-58-34-114255/test_auc.p')
pri_test_auc_f = open('./output/2017-04-26-10-16-41-095791/test_auc.p')
all_test_auc_f = open('./output/2017-04-26-18-51-54-310030/test_auc.p')

der_auc_test = pickle.load(der_test_auc_f)
pri_auc_test = pickle.load(pri_test_auc_f)
all_auc_test = pickle.load(all_test_auc_f)

