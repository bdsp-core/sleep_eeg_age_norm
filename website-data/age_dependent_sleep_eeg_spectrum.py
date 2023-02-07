#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from os.path import join

#  which = ('C', 'W', 1)
def make_figures(which):
  X = specs[which]
  Y = spec_stds[which]
  df = pd.DataFrame(X.T)
  df.columns = np.around(ages,1)

  def pprint_which(which):
      sex = "Male" if which[2]==1 else "Female"
      return which[0]+which[1]+" "+ sex

  pp = pprint_which(which)
  ax = sns.heatmap(df, xticklabels=100, yticklabels=False,cbar= False, cmap="Spectral_r")
  ax.set(xlabel="age",ylabel="freq",title=pp)

  queried_age = 13
  idx = np.searchsorted(ages, queried_age)
  plt.axvline(idx,0,1)

  plt.savefig(join("imgs",pp+"_spec.png"))
  ax.clear()

  std = Y[idx]
  val = X[idx]
  ax = sns.lineplot(val,c='k', label="µ")
  cols = ["r", "g", "b"]

  for i in range(3):
      dev = (i + 1)*std
      sns.lineplot(val+dev,c=cols[i], label=f"+/-{i+1}σ")
      sns.lineplot(val-dev,c=cols[i])
      
  ax.set(xlabel="freq",ylabel="power",title=pp+" age: " +str(queried_age))
  plt.savefig(join("imgs",pp+"_ageslice.png"))
  ax.clear()

for what in ['spectra_age_norm_nn.pickle', 'spectra_age_norm_frontal_nn.pickle']:
  with open(what, 'rb') as f:
    specs, spec_stds, freq, ages = pickle.load(f)
  for which in specs:
    make_figures(which)


