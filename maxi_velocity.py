#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:27:32 2025

@author: arshavin
"""

import os
import numpy as np
import h5py
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 2:
    print("Error: Please provide a directory path as an argument.")
    exit(1)

# Get the directory path from the command-line argument
directories = sys.argv[1]

# Use glob to expand the wildcard pattern and get a list of matching file paths
files = glob.glob(os.path.join(directories + '/xdmf_p0000_i*.h5'))

# Sort the files based on the numerical part of their names
files = sorted(
    files,
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_i')[-1])
)

# Proceed only if HDF5 files are found
if not files:
    print("No HDF5 files found in the specified directory.")
    exit(1)

# Initialize arrays to store results
x_max_array = []
y_max_array = []
max_velocity_array = []

for file_path in files:
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]
        mom_x = hdf5_file['Properties/XMOMENTUM'][:]
        mom_y = hdf5_file['Properties/YMOMENTUM'][:]

    ds1 = np.hstack((vertex, height, mom_x, mom_y))
    filtered_data = []
    xnew, ynew, znew, pile1, x_mom, y_mom = [], [], [], [], [], []
    #c1 = list(range(0, 5))
    #c = np.array(c1)
    # for i in range(len(ds1)):
    #     for j in range(5):
    #         index = int(ds1[i, j])
    #         if 0 <= index < len(ds1):
    #             x, y, z = cord[index, :3]
    #             pile_h = ds1[index, 4]
    #             #pile1.append(pile)
    #             x_momentum = ds1[index, 5]
    #             #x_mom.append(xm)
    #             y_momentum = ds1[index, 6]
    #             #y_mom.append(ym)
    #             # Apply the filtering condition
    #             if (
    #                 abs(pile_h - 0.0001) > 1e-3 and
    #                 404720 < x < 405280 and
    #                 3807860 < y < 3808410
    #             ):
    #                 filtered_data.append((x, y, pile_h, x_momentum, y_momentum))
    for i in range(len(ds1)):
        for j in range(5):
            index = int(ds1[i, j])
        #for index in ds1[i, :4]:
            if 0 <= index < len(ds1):
                xn = cord[index, 0]
                xnew.append(xn)
                yn = cord[index, 1]
                ynew.append(yn)
                zn = cord[index, 2]
                znew.append(zn)
                pile = ds1[index, 4]
                pile1.append(pile)
                x_momentum = ds1[index, 5]
                x_mom.append(x_momentum)
                y_momentum = ds1[index, 6]
                y_mom.append(y_momentum)
    
    x = np.array(xnew)
    y = np.array(ynew)
    z = np.array(znew)
    pile_h = np.array(pile1)
    x_moment = np.array(x_mom)
    y_moment = np.array(y_mom)
    
    # Apply filtering condition: only store non-zero values
    if np.all(pile_h == 0) or np.all(x_moment == 0) or np.all(y_moment == 0):
        # If all values are zero, set velocities and max_velocity to zero
        x_vel = 0
        y_vel = 0
        max_velocity = 0

    else:
        # Create a mask to filter out zero values
        non_zero_mask = (pile_h != 0) & (x_moment != 0) & (y_moment != 0)
        
        # Apply the mask to get non-zero values
        pile_h_non_zero = pile_h[non_zero_mask]
        x_moment_non_zero = x_moment[non_zero_mask]
        y_moment_non_zero = y_moment[non_zero_mask]
        
        # Check if filtered arrays are empty
        if pile_h_non_zero.size == 0:
            x_vel = 0
            y_vel = 0
            max_velocity = 0
        else:
            # Avoid division by zero and compute velocities
            x_vel = x_moment_non_zero / pile_h_non_zero
            y_vel = y_moment_non_zero / pile_h_non_zero
            
            # Compute the velocity magnitude
            velocity = np.sqrt(x_vel**2 + y_vel**2)
            max_velocity = np.max(velocity)
        
        # Create a mask to filter based on the condition
    filter_mask = (
        (np.abs(pile_h - 0.06) < 1e-2) & 
        (404720 < x) & (x < 405280) & 
        (3807860 < y) & (y < 3808410)
    )
    
    # Apply the mask to get filtered values
    filtered_x = x[filter_mask]
    filtered_y = y[filter_mask]
    
    if filtered_x.size > 0 and filtered_y.size > 0:
        x_max = np.max(filtered_x)
        y_max = np.max(filtered_y)
    else:
        x_max = 0  # or another appropriate default value
        y_max = 0  # or another appropriate default value

    
    
    # Append results for each file
    x_max_array.append(x_max)
    y_max_array.append(y_max)
    max_velocity_array.append(max_velocity)

    
    # If no data matches the condition, skip this file
    # if not filtered_data:
    #     continue

    # # Extract filtered data
    # filtered_data = np.array(filtered_data)
    # filtered_x, filtered_y, filtered_pile, filtered_x_momentum, filtered_y_momentum = filtered_data.T

    # # Calculate the desired maximum values
    # x_max = np.max(filtered_x)
    # y_max = np.max(filtered_y)

    # filtered_x_vel = filtered_x_momentum / filtered_pile
    # x_vel_max = np.max(filtered_x_vel)

    # filtered_y_vel = filtered_y_momentum / filtered_pile
    # y_vel_max = np.max(filtered_y_vel)

    #max_velocity = np.sqrt(x_vel_max**2 + y_vel_max**2)

    # Store results
    #x_max_array.append(x_max)
    #y_max_array.append(y_max)
    #max_velocity_array.append(max_velocity)

    # print(f"File: {file_path}")
    # print(f"Maximum x velocity: {x_vel_max}, Maximum y velocity: {y_vel_max}")
    # print(f"x_max: {x_max}, y_max: {y_max}")
    # print(f"Maximum velocity: {max_velocity}")
    
    xdiff_front = np.diff(x_max_array)
    ydiff_front = np.diff(y_max_array)
    rel_dist = np.sqrt(np.square(xdiff_front) + np.square(ydiff_front))
    rel_dist = np.insert(rel_dist, 0, 0)
    dist_front = np.cumsum(rel_dist)

# Convert max_velocity_array to a NumPy array
max_velocity_array = np.array(max_velocity_array)

#print("\nSummary of Results:")
#print(f"x_max values: {x_max_array}")
#print(f"y_max values: {y_max_array}")
print(f"Max velocities: {max_velocity_array}")
print(f"frontal runout distance: {dist_front}")

## code to extract data of xcen and ycen
directories = [
        '/home/arshavin/TITAN2D/CT_8_simulation/sim_asf/sim_final1/',
        ]

file_pattern = ['output_summary.-00001']

xcen = []
ycen = []

# Add debugging output
print(f"Searching for files in directory: {directories}")

# Iterate over files and process data
for directory in directories:
    file_paths = glob.glob(os.path.join(directory, file_pattern[0]))
    if len(file_paths) == 0:
        print(f"No files found in directory: {directory}")
        exit(1)

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        
        # Read X and Y center data to calculate distance
        datafile3 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[4], delimiter="=", engine='python')
        datafile3[4] = datafile3[4].str.replace('[^\d.]', '', regex=True)
        datafile3[4] = pd.to_numeric(datafile3[4], errors='coerce')
        xcen.append(datafile3.iloc[:, 0].to_numpy().astype(float))
        
        datafile2 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[5], delimiter="=", engine='python')
        datafile2[5] = datafile2[5].str.replace('[^\d.]', '', regex=True)
        datafile2[5] = pd.to_numeric(datafile2[5], errors='coerce')
        ycen.append(datafile2.iloc[:, 0].to_numpy().astype(float))
        
        xdiff_out = np.diff(xcen)
        ydiff_out = np.diff(ycen)
        rel_dist_out = np.sqrt(np.square(xdiff_out) + np.square(ydiff_out))
        rel_dist_out = np.insert(rel_dist_out, 0, 0)
        dist_out = np.cumsum(rel_dist_out)
        
#print(f"x center values: {xcen}")
#print(f"y center values: {ycen}")        
print(f"centeral runout distance: {dist_out}") 
       
# # Plot maximum velocity vs distance
# plt.figure(figsize=(10, 6))

# # Path to your CSV file
# csv_file = "/home/arshavin/TITAN2D/CT_8_simulation/validation/velocity_data.csv"

# # Read the CSV file
# try:
#     csv_data = pd.read_csv(csv_file)
#     # Replace 'distance_column' and 'velocity_column' with actual column names or indices from the CSV
#     csv_distance = csv_data['distance'].to_numpy()  # Example: Replace with actual column name
#     csv_velocity = csv_data['velocity'].to_numpy()  # Example: Replace with actual column name
# except FileNotFoundError:
#     print(f"Error: CSV file '{csv_file}' not found.")
#     exit(1)
# except KeyError:
#     print(f"Error: Check column names in the CSV file. Column names must match your data.")
#     exit(1)

# # Plot data from the CSV file
# plt.plot(csv_distance, csv_velocity, marker='x', color='r', linestyle='--', label='CSV Data')

# plt.plot(dist, max_velocity_array, marker='o', color='b', label='Max Velocity')

# # Add axis labels and title
# plt.xlabel('Distance (m)', fontsize=12)
# plt.ylabel('Maximum Velocity (m/s)', fontsize=12)
# #plt.title('Maximum Velocity vs Distance', fontsize=14)

# # Add legend and grid
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.tight_layout()
# plt.show()