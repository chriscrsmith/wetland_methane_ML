import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from copy import deepcopy
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import config
import utils

### load params
days_per_month = config.days_per_month
timesteps_per_year = config.timesteps_per_year
num_windows = config.num_windows
nonmissing_required = config.nonmissing_required

### load observed data
fp = config.wd + "/Out/preprocessed_obs.sav"
data0 = torch.load(fp, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
Z_vars_obs = data0['Z_vars']
X_stats = data0['X_stats']
Y_stats = data0['Y_stats']

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
# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G = [],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g = [],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]
        y_piece = Y_obs[site, window_range_in, 0]
        z_piece = Z_obs[site, window_range_in, :]
        m_piece = M_obs[site, window_range_in]
        g_piece = G_obs[site, window_range_in]

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
#
X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)

### initial organization
num_sites = X_obs.shape[0]
num_reps = 100
X_temp = deepcopy(X_obs_windows)
FCH4 = deepcopy(M_obs_windows) 
FCH4_F_ANNOPTLM = deepcopy(G_obs_windows) 

### load outputs
outputs = []
X_test = []
for site in range(num_sites):
    new_site = []
    for rep in range(num_reps):
        new_rep = []
        path_out = config.fp_eval + '/results_site_' + str(site) + "_rep_" + str(rep) + '.txt'

        if os.path.exists(path_out) is False:
            pass 
        else:        
            with open(path_out) as infile:
                for line in infile:
                    newline = line.strip()
                    if "FINAL OUT:" in newline:
                        newline = newline.split()[2:]
                    window = np.array(list(map(float, newline)))  # this is one window, one rep
                    new_rep.append(window)  # current rep often has multiple windows

        # add rep to site list
        if len(new_rep) > 0:
            new_site.append(np.array(new_rep))

    # average across reps
    if len(new_site) > 0:
        new_site = np.array(new_site)  # (num_reps, num_windows, window_size)        
        if len(new_site.shape) < 3:
            print("expecting more reps to average, or just write code to deal with this")
            sys.exit()
        new_site = np.mean(new_site, axis=0)    
        outputs.append(new_site)
        X_test.append(X_temp[site])
    else:
        outputs.append("missing")
        X_test.append("missing")      



### replace missing vals with nan
for i in range(len(outputs)):
    if "missing" not in outputs[i]:

        # take second year from each time series
        FCH4[i] = FCH4[i][:,timesteps_per_year:(timesteps_per_year*2)]
        FCH4_F_ANNOPTLM[i] = FCH4_F_ANNOPTLM[i][:,timesteps_per_year:(timesteps_per_year*2)]
        X_test[i] = X_test[i][:, timesteps_per_year:(timesteps_per_year*2), :]  

        # identify indices where X OR Y-gap-filled are missing
        missing = ((X_test[i] == -9999).any(dim=-1) | (FCH4_F_ANNOPTLM[i] == -9999) )

        # only evaluate months where X or Y-gap-filled are non-missing
        outputs[i][missing] = np.nan  
        FCH4_F_ANNOPTLM[i][missing] = np.nan 

        # replace remaining -9999s with nan
        missing = (FCH4[i] == -9999)
        FCH4[i][missing] = np.nan 
        missing = (FCH4_F_ANNOPTLM[i] == -9999)
        FCH4_F_ANNOPTLM[i][missing] = np.nan 
        # (outputs is never -9999)

### un-normalize
for i in range(num_sites):
    if "missing" not in outputs[i]:
        outputs[i] = ((outputs[i] * Y_stats[0,1]) + Y_stats[0,0]) / 1000
        FCH4[i] = ((FCH4[i] * M_stats[1]) + M_stats[0]) /1000 
        FCH4_F_ANNOPTLM[i] = ((FCH4_F_ANNOPTLM[i] * G_stats[1]) + G_stats[0]) /1000 

### fxn to split windows/years into "valid", complete windows, and incomplete windows
def complete_windows(FCH4, FCH4_F_ANNOPTLM, outputs, site):
    num_windows = len(FCH4[site])
    complete_true = []
    missing_true = []
    complete_est = []
    missing_est = []
    for window in range(num_windows):
        nans = torch.sum(np.isnan(FCH4[site][window]))
        if nans == 0:
            complete_true.append(deepcopy(FCH4[site][window]))
            complete_est.append(deepcopy(outputs[site][window]))            
        else:
            missing_true.append(deepcopy(FCH4_F_ANNOPTLM[site][window]))  # years with missing months evaluated against gap-filled fch4
            missing_est.append(deepcopy(outputs[site][window]))            
    #
    return complete_true, missing_true, complete_est, missing_est

### fxn to calculate yearly emissions
def yearly(timeseries):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    tot = 0
    for month in range(12):
        if not np.isnan(timeseries[month]):  # skip nans
            tot += timeseries[month] * days_per_month[month]
    return tot

### site means and squared errors
true_site_means_valid = []
est_site_means_valid = []
ses_valid = []
est_site_sds_valid = []
annual_variation_valid = []
valid_site_indices = []
#
true_site_means_missing = []
est_site_means_missing = []
ses_missing = []
est_site_sds_missing = []
annual_variation_missing = []
missing_site_indices = []
for i in range(num_sites):
    if "missing" not in outputs[i]:
        complete_true, missing_true, complete_est, missing_est = complete_windows(FCH4, FCH4_F_ANNOPTLM, outputs, i)

        # sites with at least one complete window without missing months
        if len(complete_true) > 0:
            valid_site_indices.append(i)
            annual_sums_true = []
            annual_sums_est = []
            squared_errors = []
            for window in range(len(complete_true)):  # (ignoring incomplete windows)
                yearly_true = yearly(complete_true[window])
                yearly_est = yearly(complete_est[window])
                se = (yearly_true-yearly_est)**2
                annual_sums_true.append(yearly_true)
                annual_sums_est.append(yearly_est)
                squared_errors.append(se)
            #
            true_site_means_valid.append( np.mean(annual_sums_true) )
            est_site_means_valid.append( np.mean(annual_sums_est) )
            est_site_sds_valid.append( np.std(annual_sums_est) )
            annual_variation_valid.append( np.std(annual_sums_true) )
            ses_valid.append( np.mean(squared_errors) )
            #
            true_site_means_missing.append( np.mean(annual_sums_true) )  # "complete" sites also included in the "missing" analysis 
            est_site_means_missing.append( np.mean(annual_sums_est) )
            est_site_sds_missing.append( np.std(annual_sums_est) )    
            annual_variation_missing.append( np.std(annual_sums_true) )
            ses_missing.append( np.mean(squared_errors) )

        # sites with no complete windows
        else:  
            missing_site_indices.append(i)
            annual_sums_true = []
            annual_sums_est = []
            squared_errors = []
            for window in range(len(missing_true)):
                nonmissing_months = np.sum(~np.isnan(np.array(missing_true[window])))  # sometimes missing, even though gap filled
                yearly_true = yearly(missing_true[window]) * (12./nonmissing_months)
                yearly_est = yearly(missing_est[window]) * (12./nonmissing_months)  # missing data for outputs was set to match Y-gap-filled
                se = (yearly_true-yearly_est)**2
                annual_sums_true.append( yearly_true )  # scale to 12 months (making up data)
                annual_sums_est.append( yearly_est )
                squared_errors.append(se)                
            #
            true_site_means_missing.append( np.nanmean(annual_sums_true) )  # nan mean because sometimes we train with all gap-filled data
            est_site_means_missing.append( np.nanmean(annual_sums_est) ) 
            est_site_sds_missing.append( np.nanstd(annual_sums_est) )            
            annual_variation_missing.append( np.std(annual_sums_true) )
            ses_missing.append( np.nanmean(squared_errors) )

### MSE
# sites with complete years
mse_valid = np.sqrt(np.mean(ses_valid))

# including sites with missing months
mse_missing = np.sqrt(np.mean(ses_missing))

### bootstrap
rng = np.random.default_rng(12345)   # shared seed across all models
b = 10000  # don't change this, either.

s = len(ses_valid)
bs_valid = []
for i in range(b):
    indices = rng.choice(s, size=s, replace=True)
    new_ses = np.array(ses_valid)[indices]
    bs_valid.append(np.sqrt(np.mean(new_ses)))
#
#lower_valid, upper_valid = np.percentile(bs_valid, [2.5, 97.5])
np.save(config.fp_eval + "/bootstrap_valid", np.array(bs_valid))

s = len(ses_missing)
bs_missing = []
for i in range(b):
    indices = rng.choice(s, size=s, replace=True)
    new_ses = np.array(ses_missing)[indices]
    bs_missing.append(np.sqrt(np.mean(new_ses)))
#
#lower_missing, upper_missing = np.percentile(bs_missing, [2.5, 97.5])
np.save(config.fp_eval + "/bootstrap_missing", np.array(bs_missing))

### r2

# complete sites
r2_valid = r2_score(torch.tensor(true_site_means_valid), torch.tensor(est_site_means_valid))
if r2_valid < 0:
    r2_valid = "<0"

# including sites with missing months
r2_missing = r2_score(torch.tensor(true_site_means_missing), torch.tensor(est_site_means_missing))
if r2_missing < 0:
    r2_missing = "<0"

### correlation
# complete sites
corr_valid = np.corrcoef(torch.tensor(true_site_means_valid), torch.tensor(est_site_means_valid))

# including sites with missing months
corr = np.corrcoef(torch.tensor(true_site_means_missing), torch.tensor(est_site_means_missing))

### scatter
fig, ax = plt.subplots(figsize=(5, 5)) 
plt.tight_layout()
units=["(g C $m^{-2}$ $year^{-1}$)"]
#ax.set_title(plot_title, fontsize = 15,weight='bold')
ax.set_xlabel('True '+units[0],fontsize = 15)
ax.set_ylabel('Estimated',fontsize = 15)
lim_min=-25
lim_max=300
ax.set_xlim(lim_min,lim_max)
ax.set_ylim(lim_min,lim_max)
x = np.linspace(lim_min,lim_max, 1000)
ax.plot([lim_max*-2,lim_max*2], [lim_max*-2,lim_max*2], color='lightgrey',linestyle='--')
ax.text(-10, 225,
        'Valid sites RMSE = %0.1f\nAll sites RMSE = %0.1f' % (mse_valid, mse_missing),
        fontsize = 12, ha='left', va='top')

# all sites (bottom layer)
Y_true=torch.tensor(np.array(true_site_means_missing)[missing_site_indices])  # subset for just the missing sites to avoid plotting points twice
Y_est=torch.tensor(np.array(est_site_means_missing)[missing_site_indices])
Y_sd=torch.tensor(np.array(est_site_sds_missing)[missing_site_indices])
X_eb=torch.tensor(np.array(annual_variation_missing)[missing_site_indices])
# ax.errorbar(Y_true,Y_est, xerr=X_eb, yerr=Y_sd, fmt='^', alpha=0.75, ecolor='black',
#            color="orange", markersize=10)
ax.errorbar(Y_true,Y_est, fmt='^', alpha=0.75, ecolor='black',
           color="orange", markersize=10)

# sites with complete years
Y_true=torch.tensor(true_site_means_valid)
Y_est=torch.tensor(est_site_means_valid)
Y_sd=torch.tensor(est_site_sds_valid)
X_eb=torch.tensor(annual_variation_valid)
# ax.errorbar(Y_true,Y_est, xerr=X_eb, yerr=Y_sd, fmt='o', alpha=0.75, ecolor='black',
#              color="blue", markersize=10)
ax.errorbar(Y_true,Y_est, fmt='o', alpha=0.75, ecolor='black',
             color="blue", markersize=10)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='sites with complete year',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='sites with missing months',
           markerfacecolor='orange', markersize=10),
]
plt.legend(handles=legend_handles, loc='upper left', fontsize=12)

plt.show()
fig.savefig(config.fp_eval + '/scatter.pdf', bbox_inches='tight')
