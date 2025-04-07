#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:17:19 2025

@author: arshavin
"""

import os
import sys
import numpy as np
import h5py
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt

def process_hdf5_file(file_path, filter_condition=None):
    """
    Process a single HDF5 file to calculate x_max, y_max, max_velocity, and max_h.
    The function now accepts a filter_condition argument.
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
        non_zero_mask = ((pile_h > 0.095) & (404720 < x) & (x < 405280) & (3807990 < y) & (y < 3808100))
        velocity = np.sqrt(
            (x_moment[non_zero_mask] / pile_h[non_zero_mask]) ** 2 +
            (y_moment[non_zero_mask] / pile_h[non_zero_mask]) ** 2
        )
        max_velocity = np.max(velocity)
        max_h = np.max(pile_h[non_zero_mask])

    # Apply custom filter condition if provided
    if filter_condition is not None:
        filter_mask = filter_condition(x, y, pile_h)
    # else:
    #     filter_mask = (
    #         (np.abs(pile_h - 0.06) < 1e-2) &
    #         (404720 < x) & (x < 405280) &
    #         (3807860 < y) & (y < 3808410)
    #     )

    filtered_x, filtered_y = x[filter_mask], y[filter_mask]
    x_max = np.max(filtered_x) if filtered_x.size > 0 else 0
    y_max = np.max(filtered_y) if filtered_y.size > 0 else 0

    return x_max, y_max, max_velocity, max_h

def process_directory(directory, filter_condition=None):
    """
    Process all HDF5 files in a directory.
    Now accepts filter_condition argument to apply different conditions for different directories.
    """
    files = glob.glob(os.path.join(directory, 'xdmf_p0000_i*.h5'))
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_i')[-1]))
    if not files:
        return [], [], []

    with Pool() as pool:
        results = pool.starmap(process_hdf5_file, [(file, filter_condition) for file in files])

    x_max_array, y_max_array, max_velocity_array, max_height_array = zip(*results)

    # Calculate frontal runout distance
    xdiff_front = np.diff(x_max_array)
    ydiff_front = np.diff(y_max_array)
    rel_dist = np.sqrt(xdiff_front**2 + ydiff_front**2)
    rel_dist = np.insert(rel_dist, 0, 0)
    dist_front = np.cumsum(rel_dist)

    return dist_front, max_height_array, max_velocity_array

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide directories as arguments.")
        exit(1)

    directories = sys.argv[1:]
    all_dist_front = []
    all_max_height = []

    # Predefined markers and colors
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'H', '*', '<', '>']
    colors = plt.cm.tab10.colors  # Use a colormap
    num_markers = len(markers)
    num_colors = len(colors)

    for directory in directories:
        # Set a custom filter condition based on directory
        if "directory_1" in directory:  # For example, use different condition for directory_1
            filter_condition = lambda x, y, pile_h: (np.abs(pile_h - 0.06) < 1e-2) & (404720 < x) & (x < 405280) & (3807860 < y) & (y < 3808410)
        elif "directory_2" in directory:  # For another directory, use the second condition
            filter_condition = lambda x, y, pile_h: (pile_h > 0.2) & (404720 < x) & (x < 405280) & (3807860 < y) & (y < 3808410)
        else:
            filter_condition = None  # Default condition

        dist_front, max_height, _ = process_directory(directory, filter_condition)
        print(f"Directory: {directory}, dist_front size: {len(dist_front)}, max_height size: {len(max_height)}")
        if np.array(dist_front).size > 0 and np.array(max_height).size > 0:
            all_dist_front.append((directory, dist_front, max_height))

    print(f"All dist_front data: {all_dist_front}")  # Check the collected data

    # Only plot if we have multiple directories' data
    if len(all_dist_front) > 1:
        plt.figure(figsize=(10, 10))

        for i, (directory, dist_front, max_height) in enumerate(all_dist_front):
            marker = markers[i % num_markers]
            color = colors[i % num_colors]
            label = f'Directory {i + 1}: {os.path.basename(directory)}'
            plt.plot(dist_front, max_height, marker=marker, color=color, label=label, markersize=6, linewidth=1.5)

        plt.xlabel('Distance (m)', fontsize=20)
        plt.ylabel('Maximum Height (m)', fontsize=20)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Error: No valid data found from multiple directories.")

if __name__ == "__main__":
    main()

