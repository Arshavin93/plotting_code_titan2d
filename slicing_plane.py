#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:17:40 2024

@author: ashish
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import csv

files = ['/home/ashish/analytical_simulation/obstacle_interaction_sim/sim_square_bar/sim_new_tif/sim_3/vizout/xdmf_p0000_i00000000.h5',
         '/home/ashish/analytical_simulation/obstacle_interaction_sim/sim_square_bar/sim_new_tif/sim_3/vizout/xdmf_p0000_i00004258.h5'
         ]

fig, ax = plt.subplots(figsize=(10, 10))

for idx, file_path in enumerate(files):
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]

    # Create an empty array to store the new points
    ds1 = np.hstack((vertex, height.reshape(-1, 1)))
    xnew, ynew, znew, pile1 = [], [], [], []

    for i in range(len(ds1)):
        for j in range(4):
            index = int(ds1[i, j])
            if 0 <= index < len(cord):
                xnew.append(cord[index, 0])
                ynew.append(cord[index, 1])
                znew.append(cord[index, 2])
                pile1.append(ds1[i, 4])
    
    x = np.array(xnew)
    y = np.array(ynew)
    z = np.array(znew)
    pile_h = np.array(pile1)
    
    # Combine all data into a single array for easier manipulation
    height_info = np.column_stack((x, y, z, pile_h))
    
    # Filter data based on x and y criteria
    tol = 1e-5
    height_new = height_info[(height_info[:, 0] >= 0.54) & (height_info[:, 0] <= 0.90) & (np.abs(height_info[:, 1] - 0.501422) < tol)]
    
    # Define the labels for the plots
    label = "simulation bed" if idx == 0 else "simulation result"
    
    if len(height_new) > 0:
        ax.scatter(height_new[:, 0], height_new[:, 2] + height_new[:, 3], alpha=0.2, label=label)
    else:
        print(f"No data for the specified criteria in file: {file_path}")
    
df = pd.read_csv('/home/ashish/contour_code/edited_code/squarebar_dike/experiment_bed_slicing_data.csv')
    
# Assuming the CSV has columns 'x' and 'z'
x = df['x'].values
z = df['z'].values
    
# Directly plot x and z values from CSV
ax.scatter(x, z, alpha=0.4, label='Experimental bed')
    
df = pd.read_csv('/home/ashish/contour_code/edited_code/squarebar_dike/experiment_slicing_data.csv')
    
# Assuming the CSV has columns 'x' and 'z'
x = df['x'].values
z = df['z'].values
    
# Directly plot x and z values from CSV
ax.scatter(x, z, alpha=0.3, label='Experimental data')

# df = pd.read_csv('/home/ashish/contour_code/edited_code/squarebar_dike/numerical_slicing_data.csv')
    
# # Assuming the CSV has columns 'x' and 'z'
# x = df['x'].values
# z = df['z'].values
    
# # Directly plot x and z values from CSV
# ax.scatter(x, z, alpha=0.3, label='Numerical data')


ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Longitudinal section (y = 0.5 m) for semispherical obstacles at the final stage')
ax.legend()
ax.grid(True)

# Set y-axis limits with a gap of 20
y_min = 0.20
y_max = 0.38
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + 0.02, 0.02))


plt.show()
