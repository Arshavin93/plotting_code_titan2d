import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import sys
from joblib import Parallel, delayed

# Function to read HDF5 file and extract required datasets
def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]
    return cord, vertex, height

# Modify this function to find the last HDF5 file in the directory
def get_last_hdf5_file(hdf5_directory):
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_directory, '*.h5')), reverse=True)
    return hdf5_files[0] if hdf5_files else None

# Function to extract required data from coordinates and vertex arrays
def extract_data(cord, vertex, height):
    ds1 = np.hstack((vertex, height))
    xnew, ynew, znew, pile1 = [], [], [], []
    c1 = list(range(0, 4))
    c = np.array(c1)
    for i in range(len(ds1)):
        for j in range(len(c)):
            index = int(ds1[i, j])
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

# Function to filter y values based on pile height and calculate y_min, y_max
def compute_y_bounds(y_values, pile_values, threshold=0.01):
    filtered_y = [y for y, pile in zip(y_values, pile_values) if pile > threshold]
    if filtered_y:
        y_min = np.min(filtered_y)
        y_max = np.max(filtered_y)
        return y_min, y_max
    else:
        return None, None

def calculate_DE_for_directory(hdf5_directory):
    last_hdf5_file = get_last_hdf5_file(hdf5_directory)
    if not last_hdf5_file:
        print(f"No HDF5 files found in {hdf5_directory}.")
        return None
    
    # Read data from the last HDF5 file
    cord, vertex, height = read_hdf5(last_hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)

    # Parallel computation of y_min and y_max
    num_partitions = 4  # Adjust based on CPU count
    split_y = np.array_split(ynew, num_partitions)
    split_pile = np.array_split(pile1, num_partitions)
    
    results = Parallel(n_jobs=num_partitions)(
        delayed(compute_y_bounds)(split_y[i], split_pile[i]) for i in range(num_partitions)
    )

    # Aggregate min and max values from parallel results
    y_min = min(r[0] for r in results if r[0] is not None)
    y_max = max(r[1] for r in results if r[1] is not None)
    
    lat_dist = y_max - y_min
    ini_radi = 0.104
    DE = lat_dist / ini_radi
    return DE

def main(directories):
    DE_values = []
    for directory in directories:
        DE = calculate_DE_for_directory(directory)
        if DE is not None:
            DE_values.append(DE)
    
    # Plot the DE values for each directory
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(DE_values)), DE_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Directory Index")
    plt.ylabel("DE Value")
    plt.title("DE Values Across Directories")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <hdf5_directory1> <hdf5_directory2> ...")
        sys.exit(1)
    
    directories = sys.argv[1:]
    main(directories)
