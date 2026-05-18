import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import sys
from copy import deepcopy
import config
import utils
import nn_architectures

hold_out_site = int(sys.argv[1])
rep = int(sys.argv[2])

### file paths                       
path_out = config.fp_train + '/result_' + str(hold_out_site) + "_rep_" + str(rep) + '.txt'
pretrain_path = config.fp_train + '/pretrain_' + config.model_version + "_" + str(hold_out_site) + "_rep_" + str(rep) + '.sav'
if hold_out_site == 0:
    finetune_path = config.fp_train + '/production_rep_' + str(rep) + '.sav'
else:
    finetune_path = config.fp_train + '/finetune_' + config.model_version + "_" + str(hold_out_site) + "_rep_" + str(rep) + '.sav'

### params                                               
start_year, end_year = config.start_year, config.end_year
num_years=end_year-start_year+1
timesteps_per_year=config.timesteps_per_year
timesteps=timesteps_per_year*num_years
days_per_month = config.days_per_month
timesteps_per_year = config.timesteps_per_year
num_windows = config.num_windows
nonmissing_required = config.nonmissing_required
lr_adam=config.lr_adam
bsz_obs = config.bsz_obs
bsz_sim = config.bsz_sim
patience=config.patience
factor=config.factor
maxepoch=config.maxepoch

### load observed data
data0 = torch.load(config.fp_prep_fluxnet, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
I_obs = data0['I']
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
Z_vars_obs = data0['Z_vars']
X_stats = data0['X_stats']
Y_stats = data0['Y_stats']
I_stats = data0['I_stats']

# site ID 
ids = np.arange(X_obs.shape[0])

### unnormalize

# swap -9999 for nan
Y_obs[Y_obs == -9999] = np.nan
X_obs[X_obs == -9999] = np.nan
I_obs[I_obs == -9999] = np.nan

# un-norm
Y_obs[:,:,0] = Z_norm_reverse(Y_obs[:,:,0], Y_stats[0,:])
for v in range(len(X_vars_obs)):
    X_obs[:,:,v] = Z_norm_reverse(X_obs[:,:,v], X_stats[v,:])

for v in range(7):
   I_obs[:,:,v,:,:] = (I_obs[:,:,v,:,:]*I_stats[v,1]) + I_stats[v,0]

# change nan back to -9999
Y_obs = np.nan_to_num(Y_obs, nan=-9999)
X_obs = np.nan_to_num(X_obs, nan=-9999)

### go ahead and normalize the modis images using the global mean and sd 
means = np.load(config.fp_modis_tiles + "/global_means.npy")
sds = np.load(config.fp_modis_tiles + "/global_SDs.npy")

means = means.reshape(1, 1, 7, 1, 1)
I_obs -= means
sds = sds.reshape(1, 1, 7, 1, 1)
I_obs /= sds
I_obs = np.nan_to_num(I_obs, nan=-9999)


### Separate out F_CH4 (into variable "M")

# create new var
mind = list(X_vars_obs).index("FCH4")
M_obs = deepcopy(X_obs[:,:,mind])
M_vars_obs = deepcopy(X_vars_obs[mind])
M_stats = deepcopy(X_stats[mind, :])

# remove from X
X_obs = np.delete(X_obs, mind, axis=2)
X_vars_obs = np.delete(X_vars_obs, mind)
X_stats = np.delete(X_stats, mind, axis=0)

### Separate out FCH4_F_ANNOPTLM (into variable "G")

# create new var
gind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
G_obs = X_obs[:,:,gind]
G_vars_obs = X_vars_obs[gind]
G_stats = deepcopy(X_stats[gind, :])

# remove from X
X_obs = np.delete(X_obs, gind, axis=2)
X_vars_obs = np.delete(X_vars_obs, gind)
X_stats = np.delete(X_stats, gind, axis=0)

### prep and filter time windows
X_obs = torch.tensor(X_obs)
Y_obs = torch.tensor(Y_obs)
Z_obs = torch.tensor(Z_obs)
M_obs = torch.tensor(M_obs)
G_obs = torch.tensor(G_obs)
I_obs = torch.tensor(I_obs)

# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G, new_I = [],[],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g, new_site_i = [],[],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]#.to(device)
        y_piece = Y_obs[site, window_range_in, 0]#.to(device)
        z_piece = Z_obs[site, window_range_in, :]#.to(device)
        m_piece = M_obs[site, window_range_in]#.to(device)
        g_piece = G_obs[site, window_range_in]#.to(device)
        i_piece = I_obs[site, window_range_in]#.to(device)

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]
        i_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece[month, :] = -9999  # note this is the "full" x_piece that gets passed through
                i_piece[month] = -9999                        

        # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
        # year 2
        x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
        y_piece_2 = y_piece[timesteps_per_year:(timesteps_per_year*2)]
        i_piece_2 = i_piece[timesteps_per_year:(timesteps_per_year*2)]
        
        # second check: do positions of missing x values == missing y value positions?
        # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
        # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)
        y_missing_2 = (y_piece_2 == -9999) 
        for month in range(timesteps_per_year):
            if x_missing_2[month] == True:
                x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
                y_piece_2[month] = -9999
                x_piece[month + timesteps_per_year, :] = -9999  # (+timesteps_per_year because we're modifying second year of the "full" x_piece)
                y_piece[month + timesteps_per_year] = -9999
                i_piece[month + timesteps_per_year, :] = -9999  
        # 
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
        y_missing_2 = (y_piece_2 == -9999)  

        # third check: do we have enough months with non-missing data in year 2?
        if torch.sum(y_missing_2) <= (timesteps_per_year-nonmissing_required):
            y_piece = np.expand_dims(y_piece, axis=-1)
            new_site_x.append(x_piece)
            new_site_y.append(y_piece)
            new_site_z.append(z_piece)
            new_site_m.append(m_piece)
            new_site_g.append(g_piece)
            new_site_i.append(i_piece)

    # add site to 4d list
    if len(new_site_x) > 0:
        print("site", site, ", num windows", len(new_site_x))
    else:
        print("site", site, ", num windows", len(new_site_x), "*")
    new_X.append(torch.tensor(np.array(new_site_x)))
    new_Y.append(torch.tensor(np.array(new_site_y)))
    new_Z.append(torch.tensor(np.array(new_site_z)))
    new_M.append(torch.tensor(np.array(new_site_m)))
    new_G.append(torch.tensor(np.array(new_site_g)))
    new_I.append(torch.tensor(np.array(new_site_i)))

X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)
I_obs_windows = list(new_I)

### load simulated data
data0 = torch.load(config.fp_prep_TEM, weights_only=False)
X_sim = data0['X']
Y_sim = torch.tensor(data0['Y'])
Z_sim = data0['Z']
X_vars_sim = data0['X_vars']
Y_vars_sim = data0['Y_vars']
Z_vars_sim = data0['Z_vars']
X_stats_sim = data0['X_stats']
Y_stats_sim = data0['Y_stats']

# site ID 
ids = np.arange(X_sim.shape[0])

### unnormalize

# swap -9999 for nan
Y_sim[Y_sim == -9999] = np.nan
X_sim[X_sim == -9999] = np.nan

# un-norm
Y_sim[:,:,0] = Z_norm_reverse(Y_sim[:,:,0], Y_stats_sim[0,:])
for v in range(len(X_vars_sim)):
    X_sim[:,:,v] = Z_norm_reverse(X_sim[:,:,v], X_stats_sim[v,:])

# change nan back to -9999
Y_sim = np.nan_to_num(Y_sim, nan=-9999)
X_sim = np.nan_to_num(X_sim, nan=-9999)

### filter eddy covariance sites from TEM

# get obs grid cells
bad_cells = {}
num_sites = X_obs.shape[0]
for site in range(num_sites):
    lat_ind = list(Z_vars_obs).index("LAT")
    long_ind = list(Z_vars_obs).index("LON")
    lat = np.nanmax(Z_obs[site,:,lat_ind])
    long = np.nanmax(Z_obs[site,:,long_ind])
    long,lat = coords2index(long, lat)
    coords = str(long) + "_" + str(lat)
    if coords in bad_cells:
        print("repeat:", lat, long)
    else:
        bad_cells[coords] = 0

# loop through TEM sites and make sure they don't overlap with flux towers
new_X = []
new_Y = []
new_Z = []
num_sites = X_sim.shape[0]
good_sites_sim = []
for site in range(num_sites):
    lat_ind = list(Z_vars_sim).index("lat")
    long_ind = list(Z_vars_sim).index("long")
    lat = np.nanmax(Z_sim[site,:,lat_ind])
    long = np.nanmax(Z_sim[site,:,long_ind])
    long,lat = coords2index(long, lat)
    coords = str(long) + "_" + str(lat)
    
    if coords in bad_cells:
        print("match", lat, long)
    else:
        new_X.append(X_sim[site])
        new_Y.append(Y_sim[site])
        new_Z.append(Z_sim[site])
        good_sites_sim.append(site)
#
X_sim = torch.tensor(np.array(new_X))
Y_sim = torch.tensor(np.array(new_Y))
Z_sim = torch.tensor(np.array(new_Z))

### prep and filter time windows
 
# chunk up train sites into windows
windowed_indices_sim = {}
new_X, new_Y, new_Z = [],[],[]
num_sites = X_sim.shape[0]
for site in range(num_sites):
    windowed_indices_sim[site] = []
    if site % 1000 == 0:
        print(site)
    new_site_x, new_site_y, new_site_z = [],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_sim[site, window_range_in, :]#.to(device)
        y_piece = Y_sim[site, window_range_in, 0]#.to(device)
        z_piece = Z_sim[site, window_range_in, :]#.to(device)

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):  # timesteps 1-12
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece[month, :] = -9999  # note this is the "full" x_piece that gets passed through
                
        # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
        # year 2
        x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
        y_piece_2 = y_piece[timesteps_per_year:(timesteps_per_year*2)]
        
        # second check: do positions of missing x values == missing y value positions?
        # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
        # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)
        y_missing_2 = (y_piece_2 == -9999) 
        for month in range(timesteps_per_year):
            if x_missing_2[month] == True:
                x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
                y_piece_2[month] = -9999
                x_piece[month + timesteps_per_year, :] = -9999  # (+timesteps_per_year because we're modifying second year of the "full" x_piece)
                y_piece[month + timesteps_per_year] = -9999
        # 
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
        y_missing_2 = (y_piece_2 == -9999)  
        
        # third check: do we have enough months with non-missing data in year 2?
        if torch.sum(y_missing_2) <= (timesteps_per_year-nonmissing_required):
            y_piece = np.expand_dims(y_piece, axis=-1)
            new_site_x.append(x_piece)
            new_site_y.append(y_piece)
            new_site_z.append(z_piece)
            windowed_indices_sim[site].append(it)

    # add site to 4d list
    if len(new_site_x) > 0:        
        new_X.append(torch.tensor(np.array(new_site_x)))
        new_Y.append(torch.tensor(np.array(new_site_y)))
        new_Z.append(torch.tensor(np.array(new_site_z)))

X_sim_windows = list(new_X)
Y_sim_windows = list(new_Y)
Z_sim_windows = list(new_Z)

### Here, we do some shenanigans to normalize the real and sim data together

# add indicator to real data
X_temp_obs = []
Y_temp_obs = []
X_vars = np.append(X_vars_obs, "indicator")
for i in range(len(X_obs_windows)):
    site_temp = deepcopy(X_obs_windows[i])
    site_temp[site_temp == -9999] = np.nan
    indicator = np.ones((site_temp.shape[0], site_temp.shape[1], 1))  # make indicator variable full of 1's
    nan_inds = np.where(np.isnan(site_temp[:,:,0]))  # see where other vars are missing (can look at just index 0, since all matching within windows)
    indicator[nan_inds] = np.nan  # assign missing to the indicator var
    site_temp = np.append(site_temp, indicator, axis=-1)  # append indicator to the other vars
    X_temp_obs.append(site_temp)  # append modified site to X
    site_temp = deepcopy(Y_obs_windows[i])  # here, in Y, we replace -9999 with nan
    site_temp[site_temp == -9999] = np.nan
    Y_temp_obs.append(site_temp)
#
X_temp_obs = np.concatenate(X_temp_obs, axis=0)
Y_temp_obs = np.concatenate(Y_temp_obs, axis=0)

# add indicator to TEM data
X_temp_sim = []
Y_temp_sim = []
for i in range(len(X_sim_windows)):
    if len(X_sim_windows[i]) > 0:
        site_temp = deepcopy(X_sim_windows[i])
        site_temp[site_temp == -9999] = np.nan
        indicator = np.zeros((site_temp.shape[0], site_temp.shape[1], 1))  # zeros
        nan_inds = np.where(np.isnan(site_temp[:,:,0]))
        indicator[nan_inds] = np.nan
        site_temp = np.append(site_temp, indicator, axis=-1)
        X_temp_sim.append(site_temp)
        site_temp = deepcopy(Y_sim_windows[i])
        site_temp[site_temp == -9999] = np.nan
        Y_temp_sim.append(site_temp)
#
X_temp_sim = np.concatenate(X_temp_sim, axis=0)
Y_temp_sim = np.concatenate(Y_temp_sim, axis=0)

### extra settings
obs_per_batch = 5
sims_per_batch = 5
prop_loss_obs = 0.9  # fraction of the loss I want to reflect obs
prop_loss_sim = 1-prop_loss_obs
w_obs = prop_loss_obs / obs_per_batch
w_sim = prop_loss_sim / sims_per_batch
bsz = obs_per_batch + sims_per_batch
obs_n = X_temp_obs.shape[0]
total_batches = int(np.ceil(obs_n/obs_per_batch))
sims_n = total_batches * sims_per_batch

### here I want to use a monte carlo approach to obtain the mean and sd I will use for normalization

# simulate random draws, calculate weighted mean and sd each iteration
its = 10000  # num draws to simulate
combined_X_stats = []
combined_Y_stats = []
for i in range(its):
    if i % 1000 == 0:
        print(i)
    ## Y first
    # draw random subset of sims
    rand_inds = torch.randperm(len(X_temp_sim))[0:sims_n].numpy()
    sim_draw = Y_temp_sim[rand_inds].flatten()

    # apply weights
    sim_weighted = sim_draw * w_sim
    obs_weighted = Y_temp_obs.flatten() * w_obs 

    # get missing mask
    sim_mask = ~np.isnan(sim_draw)
    obs_mask = ~np.isnan(Y_temp_obs)    

    # calc weighted mean
    weighted_sum = np.sum(sim_draw[sim_mask] * w_sim) + np.sum(Y_temp_obs[obs_mask] * w_obs)
    valid_weight_sum = (np.sum(sim_mask) * w_sim) + (np.sum(obs_mask) * w_obs)
    weighted_mean = weighted_sum / valid_weight_sum
    
    # calc weighted sd
    weighted_sum = np.sum(((sim_draw[sim_mask]-weighted_mean)**2) * w_sim) + np.sum(((Y_temp_obs[obs_mask]-weighted_mean)**2) * w_obs)
    weighted_sd = (weighted_sum / valid_weight_sum)**(1/2)

    # add to list
    combined_Y_stats.append( np.array([weighted_mean, weighted_sd]) )

    ## now X
    # draw random subset of sims
    sim_draw = X_temp_sim[rand_inds]

    # apply weights
    sim_weighted = sim_draw * w_sim
    obs_weighted = X_temp_obs * w_obs 

    # loop through each predictor
    x_stats = np.zeros((num_input_vars, 2))
    for v in range(num_input_vars):
        sim_var = sim_draw[:,:,v].flatten()
        obs_var = X_temp_obs[:,:,v].flatten()
        
        # get missing mask
        sim_mask = ~np.isnan(sim_var)
        obs_mask = ~np.isnan(obs_var)    
    
        # calc weighted mean
        weighted_sum = np.sum(sim_var[sim_mask] * w_sim) + np.sum(obs_var[obs_mask] * w_obs)
        valid_weight_sum = (np.sum(sim_mask) * w_sim) + (np.sum(obs_mask) * w_obs)
        weighted_mean = weighted_sum / valid_weight_sum
        
        # calc weighted sd
        weighted_sum = np.sum(((sim_var[sim_mask]-weighted_mean)**2) * w_sim) + np.sum(((obs_var[obs_mask]-weighted_mean)**2) * w_obs)
        weighted_sd = (weighted_sum / valid_weight_sum)**(1/2)

        # add to list
        x_stats[v,:] = np.array([weighted_mean, weighted_sd])
    #    
    combined_X_stats.append(x_stats)

# unpack
combined_X_stats = np.array(combined_X_stats)
combined_Y_stats = np.array(combined_Y_stats)
final_mean_Y = np.mean(combined_Y_stats[:,0])
final_sd_Y = np.mean(combined_Y_stats[:,1])
final_mean_X = np.mean(combined_X_stats[:,:,0], axis=0)
final_sd_X = np.mean(combined_X_stats[:,:,1], axis=0)

### normalize obs
for ind in range(len(X_obs_windows)):
    if len(X_obs_windows[ind]) > 0:
        site_temp = deepcopy(X_obs_windows[ind])
        site_temp[site_temp == -9999] = np.nan
        indicator = np.ones((site_temp.shape[0], site_temp.shape[1], 1))  # ones
        nan_inds = np.where(np.isnan(site_temp[:,:,0]))
        indicator[nan_inds] = np.nan
        site_temp = np.append(site_temp, indicator, axis=-1)
        for v in range(num_input_vars):
            site_temp[:,:,v] = (site_temp[:,:,v] - final_mean_X[v]) / final_sd_X[v]
        site_temp = np.nan_to_num(site_temp, nan=-9999) 
        X_obs_windows[ind] = torch.tensor(site_temp)
        
        site_temp = deepcopy(Y_obs_windows[ind])
        site_temp[site_temp == -9999] = np.nan
        site_temp[:,:,0] = (site_temp[:,:,0] - final_mean_Y) / final_sd_Y
        site_temp = np.nan_to_num(site_temp, nan=-9999) 
        Y_obs_windows[ind] = torch.tensor(site_temp)

### train/val/test split

# separate single test site
X_test = deepcopy(X_obs_windows[test_ind]).to(torch.float32)
Y_test = deepcopy(Y_obs_windows[test_ind]).to(torch.float32)
Z_test = deepcopy(Z_obs_windows[test_ind]).to(torch.float32)
I_test = deepcopy(I_obs_windows[test_ind]).to(torch.float32)

# shuffle remaining
X_temp = deepcopy(X_obs_windows)
Y_temp = deepcopy(Y_obs_windows)
Z_temp = deepcopy(Z_obs_windows)
I_temp = deepcopy(I_obs_windows)
del X_temp[test_ind]
del Y_temp[test_ind]
del Z_temp[test_ind]
del I_temp[test_ind]
shuffled_ind = torch.randperm(len(X_temp))

# X
X_train_obs, X_val_obs = utils.split_data_group(X_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Y
Y_train_obs, Y_val_obs = utils.split_data_group(Y_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Z
Z_train_obs, Z_val_obs = utils.split_data_group(Z_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# I
I_train_obs, I_val_obs = utils.split_data_group(I_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

### normalize sims
for ind in range(len(X_sim_windows)):
    if len(X_sim_windows[ind]) > 0:
        site_temp = deepcopy(X_sim_windows[ind])
        site_temp[site_temp == -9999] = np.nan
        indicator = np.zeros((site_temp.shape[0], site_temp.shape[1], 1))  # zeros
        nan_inds = np.where(np.isnan(site_temp[:,:,0]))
        indicator[nan_inds] = np.nan
        site_temp = np.append(site_temp, indicator, axis=-1)        
        for v in range(num_input_vars):
            site_temp[:,:,v] = (site_temp[:,:,v] - final_mean_X[v]) / final_sd_X[v]
        site_temp = np.nan_to_num(site_temp, nan=-9999) 
        X_sim_windows[ind] = torch.tensor(site_temp)

        site_temp = deepcopy(Y_sim_windows[ind])
        site_temp[site_temp == -9999] = np.nan
        site_temp[:,:,0] = (site_temp[:,:,0] - final_mean_Y) / final_sd_Y
        site_temp = np.nan_to_num(site_temp, nan=-9999) 
        Y_sim_windows[ind] = torch.tensor(site_temp)

### train/val/test split

# shuffle remaining
X_temp = list(X_sim_windows)
Y_temp = list(Y_sim_windows)
Z_temp = list(Z_sim_windows)
shuffled_ind = torch.randperm(len(X_temp))

# X
X_train_sim, X_val_sim, X_test_sim = utils.split_data_group(X_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Y
Y_train_sim, Y_val_sim, Y_test_sim = utils.split_data_group(Y_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Z
Z_train_sim, Z_val_sim, Z_test_sim = utils.split_data_group(Z_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

### initialize params for training
n_a=8 #hidden state number
n_l=2 #layer of gru
loss_val_best = 500000
best_epoch = 1000
lr_adam=0.001 #orginal 0.0001
train_losses = []
val_losses = []

### initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dropout = 0
model = gru_w_cnn(num_input_vars,n_a,n_l,1,dropout)
model.to(device)
params = list(model.parameters())
    
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
maxepoch=100

def load_tile(epoch):
    means = np.load(config.fp_modis_tiles + "/global_means.npy")
    sds = np.load(config.fp_modis_tiles + "/global_SDs.npy")
    tile = np.load(config.fp_modis_tiles + "/tile_" + str(epoch) + ".npy")  # (12519, 156, 7, 10, 10)
    
    # normalize
    means = means.reshape(1, 1, 7, 1, 1)
    tile = tile - means
    sds = sds.reshape(1, 1, 7, 1, 1)
    tile = tile / sds
    
    # remove fluxnet sites
    tile = tile[good_sites_sim]  # (12499, 156, 7, 10, 10)
    
    # filter for approved windows
    new_tile = []  # 12498
    for site in range(len(windowed_indices_sim)):
        new_site = []
        for it in windowed_indices_sim[site]:  # this skips any windows that aren't pre-approved
            window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))
            new_site.append(tile[site, window_range_in, :, :, :])
        #
        if len(new_site) > 0:
            new_tile.append(torch.tensor(np.array(new_site)))
    #
    del tile
    gc.collect()
    
    # train val test split
    train_frac=0.7; val_frac=0.3
    sample_size = len(new_tile)
    train_n=int(train_frac*sample_size)
    val_n=int(val_frac*sample_size)
    #test_n=sample_size - train_n -val_n
    I_train_sim, I_val_sim, I_test_sim = [],[],[]
    for i in range(train_n):
        I_train_sim.append(new_tile[shuffled_ind[i]])
    for i in range(train_n, train_n+val_n):
        I_val_sim.append(new_tile[shuffled_ind[i]])
    # for i in range(train_n+val_n, sample_size):
    #     I_test_sim.append(new_tile[shuffled_ind[i]])
    I_train_sim = torch.cat(I_train_sim, dim=0)
    I_val_sim = torch.cat(I_val_sim, dim=0)
    #I_test_sim = torch.cat(I_test_sim, dim=0)
    I_train_sim = I_train_sim.to(torch.float32)
    I_val_sim = I_val_sim.to(torch.float32)
    # I_test_sim = I_test_sim.to(torch.float32)
    del new_tile
    gc.collect()

    return I_train_sim, I_val_sim

### train
train_n = X_train_obs.size(0)
val_n = X_val_obs.size(0)
X_val_obs = X_val_obs.to(device)
Y_val_obs = Y_val_obs.to(device)
X_val_sim = X_val_sim.to(device)
Y_val_sim = Y_val_sim.to(device)

for epoch in range(maxepoch):

    # load image data
    I_train_sim, I_val_sim = load_tile(epoch)
    I_val_sim = I_val_sim.to(device)    
    
    train_loss=0.0
    val_loss=0.0
    shuffled_b=torch.randperm(X_train_obs.size(0)) 
    X_train_shuff_obs = X_train_obs[shuffled_b,:,:] 
    Y_train_shuff_obs = Y_train_obs[shuffled_b,:,:]
    I_train_shuff_obs = I_train_obs[shuffled_b,:,:]
    shuffled_b=torch.randperm(X_train_sim.size(0)) 
    X_train_shuff_sim = X_train_sim[shuffled_b,:,:] 
    Y_train_shuff_sim = Y_train_sim[shuffled_b,:,:]
    I_train_shuff_sim = I_train_sim[shuffled_b,:,:]

    # forward
    model.train()  # (switch on dropout; and optimization?)
    model.zero_grad()
    total_batches = int(np.ceil(train_n/obs_per_batch))  # updating based on training size
    for bb in range(total_batches):
        if bb != (total_batches-1):
            sbb_obs = bb*obs_per_batch
            ebb_obs = (bb+1)*obs_per_batch
        else:
            sbb_obs = bb*obs_per_batch
            ebb_obs = train_n
        #
        sbb_sim = bb*sims_per_batch
        ebb_sim = (bb+1)*sims_per_batch

        optimizer.zero_grad()
        
        # obs        
        hidden = model.init_hidden(ebb_obs-sbb_obs).to(device)
        X_input = X_train_shuff_obs[sbb_obs:ebb_obs, :, :].to(device)
        Y_true = Y_train_shuff_obs[sbb_obs:ebb_obs, :, :].to(device)
        I_input = I_train_shuff_obs[sbb_obs:ebb_obs].to(device)
        Y_est, _ = model(I_input, X_input, hidden)
        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
        loss_obs = mse_missing(Y_est, Y_true)

        # sim
        hidden = model.init_hidden(ebb_sim-sbb_sim).to(device)
        X_input = X_train_shuff_sim[sbb_sim:ebb_sim, :, :].to(device)
        Y_true = Y_train_shuff_sim[sbb_sim:ebb_sim, :, :].to(device)
        I_input = I_train_shuff_sim[sbb_sim:ebb_sim].to(device)
        Y_est, _ = model(I_input, X_input, hidden)
        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
        loss_sim = mse_missing(Y_est, Y_true)

        # weight loss
        loss = loss_obs*prop_loss_obs + loss_sim*prop_loss_sim
        hidden.detach_()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()  # (NOT "bsz", since some batches aren't full)
            
    # finalize training loss         
    train_loss /= total_batches
    train_losses.append(train_loss)

    # validation
    model.eval()  # "testing" model, it switches off dropout and batch norm.    
    with torch.no_grad():

        # obs
        hidden = model.init_hidden(X_val_obs.shape[0]).to(device)
        Y_val_pred_t, _ = model(I_val_obs, X_val_obs, hidden)
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up        
        loss_obs = mse_missing(Y_val_pred_t, Y_val_obs[:,timesteps_per_year:(timesteps_per_year*2),:])
        
        # sim
        hidden = model.init_hidden(X_val_sim.shape[0]).to(device)
        Y_val_pred_t, _ = model(I_val_sim, X_val_sim, hidden)
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up        
        loss_sim = mse_missing(Y_val_pred_t, Y_val_sim[:,timesteps_per_year:(timesteps_per_year*2),:])
        
        #
        val_loss = loss_obs*prop_loss_obs + loss_sim*prop_loss_sim
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Learning rate after epoch {epoch+1}: {param_group['lr']}")
                    
        # save model, update LR
        if np.array(val_loss.cpu()) < loss_val_best:
            loss_val_best=np.array(val_loss.cpu())
            best_epoch = epoch
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': train_loss,
                    'los_val': val_loss,
                    }, finetune_path)   
        print("finished training epoch", epoch+1, flush=True)
        print("learning rate: ", optimizer.param_groups[0]["lr"], "train_loss: ", train_loss, "val_loss:",val_loss, "best_val_loss:",loss_val_best, flush=True)
        path_fs = finetune_path+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model.state_dict(),
                    }, path_fs)  
print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)

### test
test_n = X_test.size(0)
def check_results(total_b, I_test, check_xset, check_y1set, Y_stats):
    Y_true_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))
    Y_pred_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))    
    for bb in range(int(total_b/1)):
        if bb != int(total_b/1)-1:
            sbb = bb*1
            ebb = (bb+1)*1
        else:
            sbb = bb*1
            ebb = total_b
        hidden = model.init_hidden(ebb-sbb)
        X_input = check_xset[sbb:ebb, :, :].to(device)
        Y_true = check_y1set[sbb:ebb, :, :].to(device)
        I_input = I_test[sbb:ebb, :, :].to(device)
        Y1_pred_t, hidden = model(I_input, X_input, hidden)

        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]  # chop off first year
        Y1_pred_t = Y1_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:] 

        # unnormalize before writing output, since every run has random sim IDs
        Y_true = Z_norm_reverse(Y_true[:,:,0], [final_mean_Y, final_sd_Y])
        Y1_pred_t = Z_norm_reverse(Y1_pred_t[:,:,0], [final_mean_Y, final_sd_Y])
        
        #         
        Y_true = torch.unsqueeze(Y_true, dim=-1)
        Y1_pred_t = torch.unsqueeze(Y1_pred_t, dim=-1)
        Y_true_all[sbb:ebb, :, :] = Y_true.to('cpu')  
        Y_pred_all[sbb:ebb, :, :] = Y1_pred_t.to('cpu')  
    #
    loss = []    
    for varn in range(check_y1set.size(2)):
        loss.append(mse_missing(Y_pred_all[:,:,varn], Y_true_all[:,:,varn]).numpy())

    return Y_pred_all, loss


with torch.no_grad():
    checkpoint=torch.load(finetune_path, map_location=torch.device('cpu'), weights_only=False)
    model=gru_w_cnn(num_input_vars,n_a,n_l,1,dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) #too large for GPU, kif not enough, change to cpu
    model.eval()  # this is "testing" model, it switches off dropout and batch norm.
    epoch = checkpoint['epoch']
    Y_test_pred,R_test,loss_test =  check_results(test_n, I_test.float(), X_test.float(), Y_test, Y_stats)    

    # write
    with open(path_out, "w") as outfile:
        for window in range(Y_test_pred.shape[0]):
            outline = list(Y_test_pred[window,:,0].numpy())
            outline = "\t".join(list(map(str, outline)))
            outfile.write(outline + "\n")
