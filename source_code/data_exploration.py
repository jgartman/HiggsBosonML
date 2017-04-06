# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:55:50 2017

@author: Josh
"""
import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("../higgs_data/atlas-higgs-challenge-2014-v2.csv")

s = data[data["Label"] == "s"]
b = data[data["Label"] == "b"]

plt.hist(s[s["DER_mass_MMC"] != -999]["DER_mass_MMC"],bins=50,histtype="step",range=(0,500),color="blue")
plt.hist(b[b["DER_mass_MMC"] != -999]["DER_mass_MMC"],bins=50,histtype="step",range=(0,500),color="green")