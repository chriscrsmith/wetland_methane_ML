import numpy as np
import matplotlib.pyplot as plt
import sys
import rasterio
from copy import deepcopy
import os
import torch
import gc
import pandas as pd
import random
import config

rep = int(sys.argv[1])

if not os.path.isdir(config.fp_modis_intermediate):
    os.mkdir(config.fp_modis_intermediate)

### params
start_year, end_year = config.start_year, config.end_year
year_range = config.year_range
days_per_month = config.days_per_month


def fill_gaps(gappy_data):
    def interpolate_nan_time(arr):
        def interpolate_series(x):
            s = pd.Series(x)
            return s.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill').to_numpy()  # tries birectional, but then relies on forward in case it's the first time point
        
        return np.apply_along_axis(interpolate_series, axis=0, arr=arr)
    def interpolate_nan_rows(arr):
        return np.apply_along_axis(
            lambda row: pd.Series(row).interpolate(limit_direction='both').to_numpy(),
            axis=1,
            arr=arr
        )
    def interpolate_nan_columns(arr):
        return interpolate_nan_rows(arr.T).T
    temp_interp = interpolate_nan_time(gappy_data)
    spatially_filled = np.empty_like(temp_interp)
    for t in range(temp_interp.shape[0]):
        frame = temp_interp[t, :, :]
        row_interp = interpolate_nan_rows(frame)
        col_interp = interpolate_nan_columns(row_interp)
        spatially_filled[t, :, :] = col_interp

    return spatially_filled


def random_flip_rotate(image):
    if random.random() > 0.5:
        image = np.flip(image, axis=2)
    if random.random() > 0.5:
        image = np.flip(image, axis=1)
    k = random.randint(0, 3)
    image = np.rot90(image, k=k, axes=(1, 2))
    return image


### subset for current rep                                                                         
n = 24
bands = ["sur_refl_b01",
         "sur_refl_b02",
         "sur_refl_b03",
         "sur_refl_b04",
         "sur_refl_b05",
         "sur_refl_b06",
         "sur_refl_b07",
        ]

### read in MODIS data
I_obs = []
for site in range(rep*n, (rep*n)+n):
    print(site)
    folder = config.fp_modis_global + "/site_" + str(site) + "/"
    grid_cell = []
    for b in range(len(bands)):

        # read data
        fp = folder + "/" + bands[b]
        if os.path.isfile(fp + ".tif"):
            with rasterio.open(fp + ".tif") as src:
                data = src.read()
                
            # fill gaps
            data[data == -9999] = np.nan
            data = fill_gaps(data)
            grid_cell.append(data)
    if len(grid_cell) < 7:
        print("missing bands", site)
        1/0
    else:            
        grid_cell = np.array(grid_cell)
        grid_cell = np.transpose(grid_cell, (1, 0, 2, 3))  # (156, 7, 112, 113)
         
    # tile
    tile_size = 10  # 10x10 tiles
    sums = np.array([0.]*7)  # 7 bands
    num_tiles = 100  # 100 random tiles
    folder = config.fp_modis_intermediate + "/site_" + str(site) + "/"
    os.makedirs(folder, exist_ok=True)
    for i in range(num_tiles):  
        x = np.random.randint(0, grid_cell.shape[-1]-tile_size)  # columns
        y = np.random.randint(0, grid_cell.shape[-2]-tile_size)  # rows
        topleft = [y,x]
        tile = grid_cell[:, :, y:y+tile_size, x:x+tile_size]
    
        # augment
        tile = np.stack([random_flip_rotate(img) for img in tile])
    
        # calc sums
        sums += np.sum(tile, axis=(0, 2, 3))   # (156, 7, 10, 10)
    
        # save
        fp2 = folder + "/tile_" + str(i) + ".npy"
        np.save(fp2, tile)    
        
    ## write
    fp3 = folder + "/sums.npy"
    np.save(fp3, sums)
