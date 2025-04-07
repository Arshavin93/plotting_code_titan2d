import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import csv
import sys
from joblib import Parallel, delayed

# Function to read HDF5 file and extract required datasets
def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]
    return cord, vertex, height

# Function to extract required data from coordinates and vertex arrays
def extract_data(cord, vertex, height):
    ds1 = np.hstack((vertex, height))
    xnew, ynew, znew, pile1 = [], [], [], []
    c1 = list(range(0, 4))
    c = np.array(c1)
    for i in range(len(ds1)):
        for j in range(len(c)):
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
    return np.array(xnew), np.array(ynew), np.array(znew), np.array(pile1)
\

# New function to process HDF5 file for a different target location
def process_hdf5_file_at_location(hdf5_file, target_x, target_y_min, target_y_max, tol):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    height_info = []

    for i in range(len(xnew)):
        if abs(xnew[i] - target_x) < tol:
            if ynew[i] < target_y_max and ynew[i] > target_y_min: 
                height_info.append(pile1[i])

    if height_info:
        h_max = np.max(height_info)
        #h_mean = np.mean(height_info)
    else:
        h_max = 0.00
        #h_mean = 0.00

    return h_max #, h_mean

def main(hdf5_directories):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Row target coordinates
    targets = [
        (0.690078, [0.245, 0.752]),
        (0.744477, [0.204, 0.794]),
        (0.798877, [0.162, 0.838])
    ]
    tol = 1e-5  # Tolerance for checking coordinates
    number_of_rows = [1, 2, 3]
    
    markers = ["o", "s", "D", "^", "v", "*", "x", "P", "h", "+"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    labels = ["R1cy_R2dia_R3thin", "R1cy_R2thin_R3dia", "R1dia_R2cy_R3thin", "R1dia_R2thin_R3cy", "R1thin_R2cy_R3dia", "R1thin_R2dia_R3cy"]

    # Loop through each specified directory
    for dir_idx, hdf5_directory in enumerate(hdf5_directories):
        # Collect mean height values for each row in the current directory
        mean_heights = []
        
        hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
        hdf5_files.sort()

        for target_x, (y_min, y_max) in targets:
            mean_height_values = Parallel(n_jobs=6)(
                delayed(process_hdf5_file_at_location)(hdf5_file, target_x, y_min, y_max, tol)
                for hdf5_file in hdf5_files
            )
            mean_heights.append(np.max(mean_height_values))

        marker_style = markers[dir_idx % len(markers)]
        color_style = colors[dir_idx % len(colors)]
        label_style = labels[dir_idx % len(labels)]
        
        # Plot mean heights for the current directory
        ax.plot(number_of_rows, mean_heights, marker=marker_style, color=color_style, linestyle='-', 
                label=label_style)

    # Customize plot
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Maximum Pile Height')
    ax.set_title('Maximum Pile Height Comparison for Different Obstacles Shape.')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script_name.py <hdf5_directory_1> <hdf5_directory_2> ...")
        sys.exit(1)
    hdf5_directories = sys.argv[1:]
    main(hdf5_directories)
    
