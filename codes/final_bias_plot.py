# libraries
import os
from pathlib import Path
from collections import defaultdict
import scipy
import random
import numpy as np
import xarray as xr
import pandas as pd
import joblib
from skimage.filters import sobel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout
from statsmodels.nonparametric.smoothers_lowess import lowess

def create_bucket_detrend_bias():
  data_types = ["raw", "seasonal", "deseason"]
  map_data = defaultdict(float) # Data by ML approach
  map_data_ens = defaultdict(float) # Data by ML approach and ensemble model
  N_ens = 1
  N_tot = 1
  recon_dir = f"{recon_output_dir}/{approach}/{ens}/member_{member}"   
  recon_fname_out = f"{recon_dir}/{approach}_recon_temporal_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.nc"   
  DS_recon = xr.load_dataset(recon_fname_out)                 
  recon = {}
  recon["raw"] = DS_recon["pCO2_recon"]
  for i in data_types[1:]:
    recon[i] = DS_recon[f"pCO2_recon_{i}"]
  ref = {}
  ref["raw"] = DS_recon["pCO2"]
  for i in data_types[1:]:
    ref[i] = DS_recon[f"pCO2_{i}"]              
  for i in data_types:
    xmean = ref[i].mean("time")
    ymean = recon[i].mean("time")
    x_minus_mean = ref[i] - xmean
    y_minus_mean = recon[i] - ymean
    ssx = xr.ufuncs.sqrt((x_minus_mean**2).sum("time"))
    ssy = xr.ufuncs.sqrt((y_minus_mean**2).sum("time"))
                  
    corr = ( x_minus_mean * y_minus_mean ).sum("time") / (ssx*ssy)
    std_x = ref[i].std("time")
    std_y = recon[i].std("time")
    bias = (ymean - xmean)
                  
                  # Average bias
    map_data[(approach,i,"bias_mean")] += bias / N_tot
    map_data_ens[(ens,approach,i,"bias_mean")] += bias / N_ens
                  
                  # Average bias**2
    map_data[(approach,i,"bias_sq")] += bias**2 / N_tot
    map_data_ens[(ens,approach,i,"bias_sq")] += bias**2 / N_ens
                  
                  # Max bias
    map_data[(approach,i,"bias_max")] = np.maximum(map_data[(approach,i,"bias_max")], bias)
    map_data_ens[(ens,approach,i,"bias_max")] = np.maximum(map_data_ens[(ens,approach,i,"bias_max")], bias)
                  
                  # Min bias
    map_data[(approach,i,"bias_min")] = np.minimum(map_data[(approach,i,"bias_min")], bias)
    map_data_ens[(ens,approach,i,"bias_min")] = np.minimum(map_data_ens[(ens,approach,i,"bias_min")], bias)
                  
                  # Mean absolute error
    map_data[(approach,i,"mae")] += np.abs(bias) / N_tot
    map_data_ens[(ens,approach,i,"mae")] += np.abs(bias) / N_ens
                  
                  # Mean % bias
    map_data[(approach,i,"bias_%error")] += bias/xmean / N_tot
    map_data_ens[(ens,approach,i,"bias_%error")] += bias/xmean / N_ens
                  
                  # Mean absolute % bias
    map_data[(approach,i,"mae_%error")] += np.abs(bias)/xmean / N_tot
    map_data_ens[(ens,approach,i,"mae_%error")] += np.abs(bias)/xmean / N_ens
                  
                  # Mean correlation
    map_data[(approach,i,"corr_mean")] += corr / N_tot
    map_data_ens[(ens,approach,i,"corr_mean")] += corr / N_ens
                  
                  # Average corr**2
    map_data[(approach,i,"corr_sq")] += corr**2 / N_tot
    map_data_ens[(ens,approach,i,"corr_sq")] += corr**2 / N_ens

                  # Stdev (amplitude) percentage error
    map_data[(approach,i,"amp_%error")] += (std_y-std_x)/std_x / N_tot
    map_data_ens[(ens,approach,i,"amp_%error")] += (std_y-std_x)/std_x / N_ens

                  # Stdev (amplitude) absolute percentage error           
    map_data[(approach,i,"stdev_%error")] += np.abs(std_y-std_x)/std_x / N_tot
    map_data_ens[(ens,approach,i,"stdev_%error")] += np.abs(std_y-std_x)/std_x / N_ens


  for i in data_types:
    map_data[(approach,i,"bias_std")] = np.sqrt(map_data[(approach,i,"bias_sq")] - map_data[(approach,i,"bias_mean")]**2)
    map_data[(approach,i,"corr_std")] = np.sqrt(map_data[(approach,i,"corr_sq")] - map_data[(approach,i,"corr_mean")]**2)

    map_data_ens[(ens,approach,i,"bias_std")] = np.sqrt(map_data_ens[(ens,approach,i,"bias_sq")] - map_data_ens[(ens,approach,i,"bias_mean")]**2)
    map_data_ens[(ens,approach,i,"corr_std")] = np.sqrt(map_data_ens[(ens,approach,i,"corr_sq")] - map_data_ens[(ens,approach,i,"corr_mean")]**2)
  map_data_fname = f"{other_output_dir}/map_data_approach.pickle"
  map_data_ens_fname = f"{other_output_dir}/map_data_ens.pickle"

  with open(map_data_fname,"wb") as handle:
    pickle.dump(map_data, handle)

  with open(map_data_ens_fname,"wb") as handle:
    pickle.dump(map_data_ens, handle)

def plot_bucket_bias(label_title):
  other_output_dir = f"{root_dir}/models/performance_metrics"
  map_data_fname = f"{other_output_dir}/map_data_approach.pickle"
  map_data_ens_fname = f"{other_output_dir}/map_data_ens.pickle"
  with open(map_data_fname,"rb") as handle:
    map_data = pickle.load(handle)
  with open(map_data_ens_fname,"rb") as handle:
    map_data_ens = pickle.load(handle)
  data_sel = "raw"
  metric_sel = "bias_mean"
  vrange = [-25, 25, 5]
  fig_shape = (1,1)
  cmap = cm.cm.balance
  lab = '$pCO_2$ mean bias ($\mu atm$) '+str(label_title)
  print(f"{metric_sel} for {data_sel} data by ML approach")      
  with plt.style.context('seaborn-talk'):
      fig = plt.figure(figsize=(6.5,9))
      dia = SpatialMap(nrows_ncols=fig_shape, fig=fig, cbar_location='bottom', cbar_orientation='horizontal')
      map_sel = map_data[(approach,data_sel,metric_sel)].roll(xlon=-30,roll_coords=True)
      sub = dia.add_plot(lat = map_sel['ylat'],
                        lon = map_sel['xlon'], 
                        data = map_sel.T, 
                        vrange=vrange[0:2], 
                        cmap=cmap,
                        ax = 0)
      col = dia.add_colorbar(sub)
      dia.set_cbar_xlabel(col, lab);
