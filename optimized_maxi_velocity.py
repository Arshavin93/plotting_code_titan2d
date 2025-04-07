#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallelized version of HDF5 file processing with `dist_out` calculation.

Created on Sat Jan 11 10:27:32 2025
@author: arshavin
"""

import os
import numpy as np
import h5py
import sys
import glob
import pandas as pd
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt

def process_hdf5_file(file_path):
    """
    Process a single HDF5 file to calculate x_max, y_max, and max_velocity.
    """
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]
        mom_x = hdf5_file['Properties/XMOMENTUM'][:]
        mom_y = hdf5_file['Properties/YMOMENTUM'][:]

    ds1 = np.hstack((vertex, height, mom_x, mom_y))
    xnew, ynew, pile1, x_mom, y_mom = [], [], [], [], []

    for i in range(len(ds1)):
        for j in range(5):
            index = int(ds1[i, j])
            if 0 <= index < len(ds1):
                xnew.append(cord[index, 0])
                ynew.append(cord[index, 1])
                pile1.append(ds1[index, 4])
                x_mom.append(ds1[index, 5])
                y_mom.append(ds1[index, 6])

    x, y, pile_h = np.array(xnew), np.array(ynew), np.array(pile1)
    x_moment, y_moment = np.array(x_mom), np.array(y_mom)

    # Calculate velocities
    if np.all(pile_h == 0) or np.all(x_moment == 0) or np.all(y_moment == 0):
        max_velocity = 0
        max_h = 0
    else:
        non_zero_mask = ((pile_h > 0.095) & (404720 < x) & (x < 405280) & (3807990 < y) & (y < 3808100)) #& (x_moment != 0) & (y_moment != 0)
        velocity = np.sqrt(
            (x_moment[non_zero_mask] / pile_h[non_zero_mask]) ** 2 +
            (y_moment[non_zero_mask] / pile_h[non_zero_mask]) ** 2
        )
        max_velocity = np.max(velocity)
        max_h = np.max(pile_h[non_zero_mask])

    # Filter for x_max and y_max
    filter_mask = (
        (pile_h > 0.2) &
        #(np.abs(pile_h - 0.1) < 1e-2) &
        (404720 < x) & (x < 405280) &
        (3807860 < y) & (y < 3808410)
    )
    filtered_x, filtered_y = x[filter_mask], y[filter_mask]
    x_max = np.max(filtered_x) if filtered_x.size > 0 else 0
    y_max = np.max(filtered_y) if filtered_y.size > 0 else 0

    return x_max, y_max, max_velocity, max_h


def process_output_summary(file_path):
    """
    Process an output_summary file to calculate distances (dist_out).
    """
    # Read height data
    datafile1 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[3], delimiter="=", engine='python')
    datafile1[3] = datafile1[3].str.replace('[^\d.]', '', regex=True)
    #datafile1[3] = pd.to_numeric(datafile1[3], errors='coerce')
    height = datafile1.iloc[:, 0].to_numpy().astype(float)
    
    # datafile3 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[4], delimiter="=", engine='python')
    # datafile3[4] = datafile3[4].str.replace('[^\d.]', '', regex=True).astype(float)
    # xcen = datafile3.iloc[:, 0].to_numpy()

    # datafile2 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[5], delimiter="=", engine='python')
    # datafile2[5] = datafile2[5].str.replace('[^\d.]', '', regex=True).astype(float)
    # ycen = datafile2.iloc[:, 0].to_numpy()

    # xdiff_out = np.diff(xcen)
    # ydiff_out = np.diff(ycen)
    # rel_dist_out = np.sqrt(xdiff_out**2 + ydiff_out**2)
    # rel_dist_out = np.insert(rel_dist_out, 0, 0)
    # dist_out = np.cumsum(rel_dist_out)
    return height


def main():
    if len(sys.argv) < 2:
        print("Error: Please provide a directory path as an argument.")
        exit(1)

    directories = sys.argv[1]

    # Collect HDF5 files
    files = glob.glob(os.path.join(directories, 'xdmf_p0000_i*.h5'))
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_i')[-1]))
    if not files:
        print("No HDF5 files found.")
        exit(1)

    # Parallel processing of HDF5 files
    with Pool() as pool:
        results = pool.map(process_hdf5_file, files)

    # Unpack results
    x_max_array, y_max_array, max_velocity_array, max_height_array = zip(*results)

    # Calculate frontal runout distance
    xdiff_front = np.diff(x_max_array)
    ydiff_front = np.diff(y_max_array)
    #rel_dist = xdiff_front
    rel_dist = np.sqrt(xdiff_front**2 + ydiff_front**2)
    rel_dist = np.insert(rel_dist, 0, 0)
    dist_front = np.cumsum(rel_dist)

    print(f"Max velocities: {max_velocity_array}")
    print(f"Frontal runout distance: {dist_front}")

    # Process `output_summary` file
    output_summary_dir = "/home/arshavin/TITAN2D/CT_8_simulation/sim_asf/sim_final2/"
    file_pattern = "output_summary.-00001"
    summary_files = glob.glob(os.path.join(output_summary_dir, file_pattern))
    if not summary_files:
        print("No output_summary files found.")
        exit(1)

    dist_out = process_output_summary(summary_files[0])
    print(f"Central runout distance: {dist_out}")

    # Plot maximum velocity vs distance
    plt.figure(figsize=(10, 10))
    
    # Path to your CSV file
    csv_file = "/home/arshavin/TITAN2D/CT_8_simulation/validation/velocity_data.csv"
    
    # Read the CSV file
    try:
        csv_data = pd.read_csv(csv_file)
        # Replace 'distance_column' and 'velocity_column' with actual column names or indices from the CSV
        csv_distance = csv_data['distance'].to_numpy()  # Example: Replace with actual column name
        csv_velocity = csv_data['velocity'].to_numpy()  # Example: Replace with actual column name
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        exit(1)
    except KeyError:
        print(f"Error: Check column names in the CSV file. Column names must match your data.")
        exit(1)
    
    # Plot data from the CSV file
    plt.plot(csv_distance, csv_velocity, marker='x', color='r', linestyle='--', label='Singh et al. (2020)', markersize=10)
    
    plt.plot(dist_front, max_velocity_array, marker='o', color='b', label='Present work', markersize=5)

    # Add axis labels and title
    plt.xlabel('Distance (m)', fontsize=20)
    plt.ylabel('Maximum height (m)', fontsize=20)
    #plt.title('Maximum Velocity vs Distance', fontsize=14)
    
    # Add legend and grid
    plt.legend(fontsize=20)
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
