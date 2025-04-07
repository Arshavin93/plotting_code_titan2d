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

# Function to convert time string to seconds
def time_to_seconds(time_str):
    match = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
    if match:
        hours, minutes, seconds = map(float, match.groups())
        return (hours * 3600) + (minutes * 60) + seconds
    else:
        return None

# Function to process output summary file
def process_output_summary(file_path):
    time = [] # or sometimes if it does not consider 00 sec then write time = [0]
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\d+:\d+:\d+(?:\.\d+)?) .* Vave=(\d+\.\d+).* hmax=(\d+\.\d+).* Xcen=(-?\d+\.\d+) .*', line)
            if match:
                time_str, _, _, _ = match.groups()
                time_seconds = time_to_seconds(time_str)
                if time_seconds is not None:
                    time.append(time_seconds)
    return np.array(time)

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

def main(hdf5_directory, time_directory):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    max_height_values_1 = []
    max_height_values_2 = []
    max_height_values_3 = []
    target_x_1 = 0.690078
    target_y_1 = [0.245, 0.752]  # First y-coordinate
    target_x_2 = 0.744477      # Second x-coordinate
    target_y_2 = [0.204, 0.794]  # Second y-coordinate
    target_x_3 = 0.798877
    target_y_3 = [0.162, 0.838]
    tol = 1e-5  # Tolerance for checking coordinates
    
    # Search for HDF5 files in the HDF5 directory
    hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
    hdf5_files.sort()  # Sort the list of file paths

    # Process HDF5 files for the first target location
    max_height_values_1 = Parallel(n_jobs=6)(delayed(process_hdf5_file_at_location)(hdf5_file, target_x_1, target_y_1[0], target_y_1[1], tol) for hdf5_file in hdf5_files)
    
    # Process HDF5 files for the second target location
    max_height_values_2 = Parallel(n_jobs=6)(delayed(process_hdf5_file_at_location)(hdf5_file, target_x_2, target_y_2[0], target_y_2[1], tol) for hdf5_file in hdf5_files)
    
    max_height_values_3 = Parallel(n_jobs=6)(delayed(process_hdf5_file_at_location)(hdf5_file, target_x_3, target_y_3[0], target_y_3[1], tol) for hdf5_file in hdf5_files)
    
    # Search for time file in the time directory
    time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
    if not time_files:
        print("No time file found in directory.")
        sys.exit(1)
    time_file_path = time_files[0]
    
    time_array = process_output_summary(time_file_path)
    
    # Plot for the first location
    plt.plot(time_array, np.array(max_height_values_1), '*-', label='Row 1')

    # Plot for the second location
    plt.plot(time_array, np.array(max_height_values_2), 'o-', label='Row 2')
    
    plt.plot(time_array, np.array(max_height_values_3), 'x-', label='Row 3')
    
    plt.grid(True)
    plt.legend()
    plt.title('Pile height variation for mixed configuration R1-thin cuboid, R2-diamond cuboid, and R3-cylindrical obstacles shape.')
    plt.xlabel('time')
    plt.ylabel('pile height')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <hdf5_directory> <time_directory>")
        sys.exit(1)
    hdf5_directory = sys.argv[1]
    time_directory = sys.argv[2]
    main(hdf5_directory, time_directory)

