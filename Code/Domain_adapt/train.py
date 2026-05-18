import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from io import open
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys
import subprocess as sp
from IPython.display import display
from scipy import stats
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

#####################
##### Pretrain  #####
#####################

### load observed data first (b/c want to filter these sites from TEM training)
data0 = torch.load(config.fp_prep_fluxnet, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
Z_vars_obs = data0['Z_vars']
X_stats_obs = data0['X_stats']
Y_stats_obs = data0['Y_stats']

Z_obs = Z_obs[:,:,1:3]  # just lat, long to match formatting of sims
Z_vars_obs = Z_vars_obs[1:3]

# site ID 
ids = np.arange(X_obs.shape[0])

### Separate out F_CH4 (into variable "M")

# create new var
mind = list(X_vars_obs).index("FCH4")
M_obs = X_obs[:,:,mind]
M_vars_obs = X_vars_obs[mind]

# remove from X
X_obs = np.delete(X_obs, mind, axis=2)
X_vars_obs = np.delete(X_vars_obs, mind)

### remove TEM output
ind = list(X_vars_obs).index("tem_flux")
X_obs = np.delete(X_obs, ind, axis=2)
X_vars_obs = np.delete(X_vars_obs, ind)

### Separate out FCH4_F_ANNOPTLM (into variable "G")

# create new var
gind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
G_obs = X_obs[:,:,gind]
G_vars_obs = X_vars_obs[gind]

# remove from X
X_obs = np.delete(X_obs, gind, axis=2)
X_vars_obs = np.delete(X_vars_obs, gind)
 
# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G = [],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g = [],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]#.to(device)
        y_piece = Y_obs[site, window_range_in, 0]#.to(device)
        z_piece = Z_obs[site, window_range_in, :]#.to(device)
        m_piece = M_obs[site, window_range_in]#.to(device)
        g_piece = G_obs[site, window_range_in]#.to(device)

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece_1[month, :] = -9999                        

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

X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)

### train/val/test split

# separate single test site
X_test = deepcopy(X_obs_windows[test_ind]).to(torch.float32)
Y_test = deepcopy(Y_obs_windows[test_ind]).to(torch.float32)
Z_test = deepcopy(Z_obs_windows[test_ind]).to(torch.float32)

# shuffle remaining
X_temp = deepcopy(X_obs_windows)
Y_temp = deepcopy(Y_obs_windows)
Z_temp = deepcopy(Z_obs_windows)
del X_temp[test_ind]
del Y_temp[test_ind]
del Z_temp[test_ind]
shuffled_ind = torch.randperm(len(X_temp))

# X
X_train_obs, X_val_obs = utils.split_data_group(X_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Y
Y_train_obs, Y_val_obs = utils.split_data_group(Y_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Z
Z_train_obs, Z_val_obs = utils.split_data_group(Z_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

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
#
X_sim = torch.tensor(np.array(new_X))
Y_sim = torch.tensor(np.array(new_Y))
Z_sim = torch.tensor(np.array(new_Z))
 
# chunk up train sites into windows
new_X, new_Y, new_Z = [],[],[]
num_sites = X_sim.shape[0]
for site in range(num_sites):
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
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece_1[month, :] = -9999                        

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
        # 
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
        y_missing_2 = (y_piece_2 == -9999)  

        # third check: do we have enough months with non-missing data in year 2?
        if torch.sum(y_missing_2) <= (timesteps_per_year-nonmissing_required):
            y_piece = np.expand_dims(y_piece, axis=-1)
            new_site_x.append(x_piece)
            new_site_y.append(y_piece)
            new_site_z.append(z_piece)

    # add site to 4d list
    if len(new_site_x) > 0:        
        new_X.append(torch.tensor(np.array(new_site_x)))
        new_Y.append(torch.tensor(np.array(new_site_y)))
        new_Z.append(torch.tensor(np.array(new_site_z)))

X_sim_windows = list(new_X)
Y_sim_windows = list(new_Y)
Z_sim_windows = list(new_Z)

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

### pre-train
stuck_value = 1.3862943649291992  # this is an empirically observed value it gets stuck at for some reason.
val_loss_domain=deepcopy(stuck_value)  # (default)

while abs(val_loss_domain - stuck_value) < 0.1:  # restart training if domain classifier gets stuck
    
    ### initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lambda_ = 0.00001  # ******************************
    n_a=8 #hidden state number
    n_l=2 #layer of gru
    bsz = 1000  # *********
    dropout = 0
    lr_adam=0.001 #orginal 0.0001
    best_epoch = 1000    
    loss_val_best = 500000
    model = domain_adapt(X_train_sim.shape[-1], n_a, lambda_)
    model.to(device)
        
    #get_gpu_memory()
    optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
    maxepoch = 100
    max_epochs_wo_improvement = 10
    train_losses = []
    val_losses = []
    
    bce = nn.BCELoss()
    train_n = X_train_sim.size(0)
    val_n = X_val_sim.size(0)
    epochs_wo_improvement = int(max_epochs_wo_improvement)
    for epoch in range(maxepoch):
        train_loss=0.0
        train_loss_pred=0.0
        train_loss_domain=0.0
        val_loss=0.0
        val_loss_pred=0.0
        val_loss_domain=0.0
        shuffled_inds=torch.randperm(X_train_sim.size(0)) 
        X_train_shuff=X_train_sim[shuffled_inds,:,:] 
        Y_train_shuff=Y_train_sim[shuffled_inds,:,:]
        
        # iterate over batches
        for bb in range(int(train_n/bsz)):
            if bb != int(train_n/bsz)-1:
                sbb = bb*bsz
                ebb = (bb+1)*bsz
            else:
                sbb = bb*bsz
                ebb = train_n
            #
            optimizer.zero_grad()
            
            # train feature extractor and output branch on source domain (sims)
            X_input = X_train_shuff[sbb:ebb,:,:].to(device)
            Y_true = Y_train_shuff[sbb:ebb,:,:].to(device)
            Y_est, domain_est = model(X_input)
            Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
            Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
            loss1 = mse_missing(Y_est, Y_true)
    
            # get smaller batch for domain classifier
            bsz_domain_classifier = 3  # separate batch size for domain classifier
                                       # 3 worked; haven't tried others. Consider num windows per site.
            bsz_current = domain_est.size(0)
            if bsz_domain_classifier > bsz_current:
                bsz_domain_classifier = int(bsz_current)
    
            # train domain classifier on source domain (sims)
            shuffled_inds = torch.randperm(bsz_current)  # random indices, size of current batch
            shuffled_inds = shuffled_inds[0:bsz_domain_classifier] # first <bsz_domain_classifier> indices
            domain_est = domain_est[shuffled_inds].float()  
            domain_labels = torch.unsqueeze(torch.tensor([0.0]*bsz_domain_classifier), 1).to(device)
            loss2 = bce(domain_est, domain_labels) 
    
            # train domain classifier on target domain (observed; batches overlapping since we have fewer data)
            shuffled_inds = torch.randperm(X_train_obs.size(0))  # random indices, size of observed training set
            shuffled_inds = shuffled_inds[0:bsz_domain_classifier] # first <bsz_domain_classifier> indices
            X_input = X_train_obs[shuffled_inds].float().to(device)
            _, domain_est = model(X_input)
            domain_labels = torch.unsqueeze(torch.tensor([1.0]*bsz_domain_classifier), 1).to(device)
            loss2 += bce(domain_est, domain_labels)
    
            # loss calculation and back prop
            bsz_ratio = (ebb-sbb) / bsz_domain_classifier
            loss = (loss1 + loss2*bsz_ratio) / 2 # equal contribution of MLE loss and domain classifier loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item() * (ebb-sbb) 
                train_loss_pred += loss1.item() * (ebb-sbb)
                train_loss_domain += loss2.item() * (bsz_domain_classifier*2) 
        #
        # finalize training loss         
        train_loss /= train_n
        train_loss_pred /= train_n
        train_loss_domain /= (bsz_domain_classifier*2*(bb+1))
        train_losses.append(train_loss)
        
        # validation
        model.eval()  # "testing" model, it switches off dropout and batch norm.    
        with torch.no_grad():
            for bb in range(int(val_n/bsz)):
                if bb != int(val_n/bsz)-1:
                    sbb = bb*bsz
                    ebb = (bb+1)*bsz
                else:
                    sbb = bb*bsz
                    ebb = val_n
    
                # val feature extractor and output branch on source domain (sims)
                X_input = X_val_sim[sbb:ebb,:,:].to(device)
                Y_true = Y_val_sim[sbb:ebb,:,:].to(device)
                Y_val_pred_t, domain_est = model(X_input)
                Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
                Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:] 
                loss1 = mse_missing(Y_val_pred_t, Y_true)
        
                # get smaller batch for domain classifier 
                bsz_current = domain_est.size(0)
                if bsz_domain_classifier > bsz_current:
                    bsz_domain_classifier = int(bsz_current)
        
                # val domain classifier on source domain (sims)
                shuffled_inds = torch.randperm(bsz_current)  # random indices, size of current batch
                shuffled_inds = shuffled_inds[0:bsz_domain_classifier] # first <bsz_domain_classifier> indices
                domain_est = domain_est[shuffled_inds].float()  
                domain_labels = torch.unsqueeze(torch.tensor([0.0]*bsz_domain_classifier), 1).to(device)
                loss2 = bce(domain_est, domain_labels) 
                
                # val domain classifier on target domain (observed; batches overlapping since we have fewer data)
                shuffled_inds = torch.randperm(X_val_obs.size(0))  # random indices, size of observed val set
                shuffled_inds = shuffled_inds[0:bsz_domain_classifier] # first <bsz_domain_classifier> indices
                X_input = X_val_obs[shuffled_inds].float().to(device)
                _, domain_est = model(X_input)
                domain_labels = torch.unsqueeze(torch.tensor([1.0]*bsz_domain_classifier), 1).to(device)
                loss2 += bce(domain_est, domain_labels)       
            
                # loss calculation
                bsz_ratio = (ebb-sbb) / bsz_domain_classifier
                loss = (loss1 + loss2*bsz_ratio) / 2 # equal contribution of MLE loss and domain classifier loss
                val_loss += loss.item() * (ebb-sbb) 
                val_loss_pred += loss1.item() * (ebb-sbb)
                val_loss_domain += loss2.item() * (bsz_domain_classifier*2) 
            
            # finalize val loss         
            val_loss /= val_n
            val_loss_pred /= val_n
            val_loss_domain /= (bsz_domain_classifier*2*(bb+1))
            if val_loss_domain < 0.5:  # restart training if the domain classifier is doing too well
                                       # after including missing TEM data, feature extractor doesn't do great 
                print("\t\tval_loss_domain", val_loss_domain)
                break
            val_losses.append(val_loss)
            
            # save model, update LR
            if val_loss_pred < loss_val_best:
                epochs_wo_improvement = int(max_epochs_wo_improvement)
                loss_val_best=np.array(val_loss_pred)
                best_epoch = epoch
                torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'loss': train_loss,
                        'los_val': val_loss,
                        }, pretrain_path)   
            else:
                epochs_wo_improvement -= 1
                if epochs_wo_improvement == 0:
                    optimizer.param_groups[0]['lr'] /= 2
                    print("\n\tno improvement in val_loss_pred; learning rate halved to", optimizer.param_groups[0]['lr'], "\n")
                    epochs_wo_improvement = int(max_epochs_wo_improvement)
            print("finished training epoch", epoch+1, flush=True)
            print("learning rate: ", optimizer.param_groups[0]["lr"], "train_loss: ", train_loss, "val_loss:",val_loss, "best_val_loss:",loss_val_best, flush=True)
            path_fs = pretrain_path+'fs'
            torch.save({'train_losses': train_losses,
                        'val_losses': val_losses,
                        'state_dict_fs': model.state_dict(),
                        }, path_fs)  
            if optimizer.param_groups[0]['lr'] < 1e-5 or abs(val_loss_domain - stuck_value) < 0.01:
                break
        #
        model.train()  # switch back to training mode after validation (e.g. dropout back on)
    #
    print("final train_loss:",train_loss, "val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)

    

#####################
##### Fine tune #####
#####################
data0 = torch.load(path_save_obs, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
Z_vars_obs = data0['Z_vars']
X_stats_obs = data0['X_stats']
Y_stats_obs = data0['Y_stats']
Z_obs = Z_obs[:,:,1:3]  # just lat, long to match formatting of sims
Z_vars_obs = Z_vars_obs[1:3]

# site ID 
ids = np.arange(X_obs.shape[0])

### Separate out F_CH4 (into variable "M")

# create new var
mind = list(X_vars_obs).index("FCH4")
M_obs = X_obs[:,:,mind]
M_vars_obs = X_vars_obs[mind]

# remove from X
X_obs = np.delete(X_obs, mind, axis=2)
X_vars_obs = np.delete(X_vars_obs, mind)

### remove TEM output
ind = list(X_vars_obs).index("tem_flux")
X_obs = np.delete(X_obs, ind, axis=2)
X_vars_obs = np.delete(X_vars_obs, ind)

### Separate out FCH4_F_ANNOPTLM (into variable "G")

# create new var
gind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
G_obs = X_obs[:,:,gind]
G_vars_obs = X_vars_obs[gind]

# remove from X
X_obs = np.delete(X_obs, gind, axis=2)
X_vars_obs = np.delete(X_vars_obs, gind)

### re-normalize

Y_obs[Y_obs == -9999] = np.nan
X_obs[X_obs == -9999] = np.nan

# un-normalize using TEM stats
Y_obs = Z_norm_reverse(Y_obs[:,:,0], Y_stats_obs[0,:])
for v in range(len(X_vars_obs)):
    X_obs[:,:,v] = Z_norm_reverse(X_obs[:,:,v],X_stats_obs[v,:])

# normalize 
Y_obs = np.reshape(Y_obs, (Y_obs.shape[0],Y_obs.shape[1],1))
X_stats_obs = np.zeros((X_obs.shape[-1], 2))
Y_stats_obs = np.zeros((1, 2))  
for v in range(1):
    var = np.nanvar(Y_obs[:,:,v])
    Y_obs[:,:,v], Y_stats_obs[v,0], Y_stats_obs[v,1] = Z_norm(Y_obs[:,:,v])
print()
for v in range(len(X_vars_obs)):
    var = np.nanvar(X_obs[:,:,v])
    if var > 0:
        X_obs[:,:,v], X_stats_obs[v,0], X_stats_obs[v,1] = Z_norm(X_obs[:,:,v])
    else:
        print("ZERO VARIANCE COLUMN")

X_obs = np.nan_to_num(X_obs, nan=-9999)
Y_obs = np.nan_to_num(Y_obs, nan=-9999)
X_obs = torch.tensor(X_obs)
Y_obs = torch.tensor(Y_obs)

# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G = [],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g = [],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]#.to(device)
        y_piece = Y_obs[site, window_range_in, 0]#.to(device)
        z_piece = Z_obs[site, window_range_in, :]#.to(device)
        m_piece = M_obs[site, window_range_in]#.to(device)
        g_piece = G_obs[site, window_range_in]#.to(device)

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece_1[month, :] = -9999                        

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

X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)

### train/val/test split

# separate single test site
X_test = deepcopy(X_obs_windows[test_ind]).to(torch.float32)
Y_test = deepcopy(Y_obs_windows[test_ind]).to(torch.float32)
Z_test = deepcopy(Z_obs_windows[test_ind]).to(torch.float32)

# shuffle remaining
X_temp = deepcopy(X_obs_windows)
Y_temp = deepcopy(Y_obs_windows)
Z_temp = deepcopy(Z_obs_windows)
del X_temp[test_ind]
del Y_temp[test_ind]
del Z_temp[test_ind]
shuffled_ind = torch.randperm(len(X_temp))

# X
X_train, X_val = utils.split_data_group(X_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Y
Y_train, Y_val = utils.split_data_group(Y_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

# Z
Z_train, Z_val = utils.split_data_group(Z_temp,shuffled_ind, train_frac=0.7,val_frac=0.3,test_frac=0.0)

### initialize params for training
n_a=8 #hidden state number
n_l=2 #layer of gru
loss_val_best = 500000
best_epoch = 1000
lr_adam=0.001 #orginal 0.0001
bsz = 10
train_losses = []
val_losses = []

### load pre-trained model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

dropout = 0

with torch.no_grad():
    checkpoint=torch.load(pretrain_path, map_location=device, weights_only=False)   
    model = domain_adapt(X_train_sim.shape[-1], n_a, lambda_)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device) #too large for GPU, kif not enough, change to cpu

params = list(model.parameters())
    
#get_gpu_memory()
optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
decay_time = 80  # og=80
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_time, gamma=0.5)
maxepoch=100

### fine-tune
train_n = X_train.size(0)
val_n = X_val.size(0)
test_n = X_test.size(0)
X_val = X_val.to(device)
Y_val = Y_val.to(device)

for epoch in range(maxepoch):
    train_loss=0.0
    val_loss=0.0
    shuffled_b=torch.randperm(X_train.size(0)) 
    X_train_shuff=X_train[shuffled_b,:,:] 
    Y_train_shuff=Y_train[shuffled_b,:,:]
    
    # forward
    model.train()  # (switch on dropout; and optimization?)
    model.zero_grad()
    for bb in range(int(train_n/bsz)):
        if bb != int(train_n/bsz)-1:
            sbb = bb*bsz
            ebb = (bb+1)*bsz
        else:
            sbb = bb*bsz
            ebb = train_n
        #
        optimizer.zero_grad()
        X_input = X_train_shuff[sbb:ebb, :, :].to(device)
        Y_true = Y_train_shuff[sbb:ebb, :, :].to(device)
        Y_est, _ = model(X_input)
        
        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
        
        loss = mse_missing(Y_est, Y_true)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item() * timesteps_per_year * (ebb-sbb)  # (NOT "bsz", since some batches aren't full)
    
    # validation
    model.eval()  # "testing" model, it switches off dropout and batch norm.    
    with torch.no_grad():

        # finalize training loss         
        train_loss /= (train_n*timesteps_per_year)
        train_losses.append(train_loss)
        
        Y_val_pred_t, _ = model(X_val)
        
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        
        loss = mse_missing(Y_val_pred_t, Y_val[:,timesteps_per_year:(timesteps_per_year*2),:])
        val_loss += loss.item() * timesteps_per_year * X_val.shape[0]
        #
        val_loss /= (val_n*timesteps_per_year)        
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Learning rate after epoch {epoch+1}: {param_group['lr']}")
        
        # save model, update LR
        if val_loss < loss_val_best:
            loss_val_best=np.array(val_loss)
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
#
print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)

### test
def check_results(total_b, check_xset, check_y1set, Y_stats):
    Y_true_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))
    Y_pred_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))    
    for bb in range(int(total_b/1)):
        if bb != int(total_b/1)-1:
            sbb = bb*1
            ebb = (bb+1)*1
        else:
            sbb = bb*1
            ebb = total_b
        X_input = check_xset[sbb:ebb, :, :].to(device)
        Y_true = check_y1set[sbb:ebb, :, :].to(device)
        Y1_pred_t, _ = model(X_input)

        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]  # chop off first year
        Y1_pred_t = Y1_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:] 
        
        # unnormalize before writing output, since every run has random sim IDs
        Y_true = Z_norm_reverse(Y_true[:,:,0], Y_stats[0,:])
        Y1_pred_t = Z_norm_reverse(Y1_pred_t[:,:,0],Y_stats[0,:])
        
        #            
        Y_true_all[sbb:ebb, :, 0] = Y_true.to('cpu')  
        Y_pred_all[sbb:ebb, :, 0] = Y1_pred_t.to('cpu')  
    #
    loss = []    
    for varn in range(check_y1set.size(2)):
        loss.append(mse_missing(Y_pred_all[:,:,varn], Y_true_all[:,:,varn]).numpy())
    return Y_pred_all, loss


with torch.no_grad():
    checkpoint=torch.load(finetune_path, map_location=torch.device('cpu'), weights_only=False)
    model=domain_adapt(len(X_vars_obs),n_a, lambda_)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) #too large for GPU, kif not enough, change to cpu
    model.eval()  # this is "testing" model, it switches off dropout and batch norm.
    epoch = checkpoint['epoch']
    Y_test_pred,R_test,loss_test =  check_results(X_test.size(0), X_test.float(), Y_test, Y_stats_obs)    

    # write
    with open(path_out, "w") as outfile:
        for window in range(Y_test_pred.shape[0]):
            outline = list(Y_test_pred[window,:,0].numpy())
            outline = "\t".join(list(map(str, outline)))
            outfile.write(outline + "\n")
