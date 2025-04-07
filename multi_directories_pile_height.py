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
    for i in range(len(ds1)):
        for index in ds1[i, :4]:
            index = int(index)
            if 0 <= index < len(ds1):
                xnew.append(cord[index, 0])
                ynew.append(cord[index, 1])
                znew.append(cord[index, 2])
                pile1.append(ds1[index, 4])
    return np.array(xnew), np.array(ynew), np.array(znew), np.array(pile1)

# Function to convert time string to seconds
def time_to_seconds(time_str):
    match = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
    if match:
        hours, minutes, seconds = map(float, match.groups())
        return (hours * 3600) + (minutes * 60) + seconds
    return None

# Function to process output summary file
def process_output_summary(file_path):
    time = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\d+:\d+:\d+(?:\.\d+)?) .* Vave=(\d+\.\d+).* hmax=(\d+\.\d+).* Xcen=(-?\d+\.\d+) .*', line)
            if match:
                time_str, _, _, _ = match.groups()
                time_seconds = time_to_seconds(time_str)
                if time_seconds is not None:
                    time.append(time_seconds)
    return np.array(time)

# Function to read data from CSV file
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            data.append(row)
    return data

def process_hdf5_file(hdf5_file, target_y, tol):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    height_info = [pile1[i] for i in range(len(xnew)) if abs(ynew[i] - target_y) < tol and 0.6 < xnew[i] < 0.615]
    return np.max(height_info) if height_info else 0.00

def main(hdf5_directories, time_directories, csv_file_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    target_y = 0.501422
    tol = 1e-5
    
    # Ensure each HDF5 directory has a corresponding time directory
    if len(hdf5_directories) != len(time_directories):
        print("Mismatch between the number of HDF5 directories and time directories.")
        sys.exit(1)
    
    # Define markers and labels for each dataset
    markers = ['o', 'x', '*', 's', 'd', '^', 'v', '<', '>', 'p']
    labels = ['MC rheology', 'VS rheology']
    
    # Process each HDF5 directory with its corresponding time directory
    for i, (hdf5_directory, time_directory) in enumerate(zip(hdf5_directories, time_directories)):
        # Gather HDF5 files from the current HDF5 directory
        hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
        hdf5_files.sort()
        
        # Calculate max height for each HDF5 file
        max_height_values = Parallel(n_jobs=4)(delayed(process_hdf5_file)(hdf5_file, target_y, tol) for hdf5_file in hdf5_files)
        
        # Process the time file in the current time directory
        time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
        if not time_files:
            print(f"No time file found in directory {time_directory}.")
            sys.exit(1)
        time_file_path = time_files[0]
        time_array = process_output_summary(time_file_path)
        
        # Plot with specific marker and label for each directory
        marker = markers[i % len(markers)]  # Cycle through markers if more directories than markers
        label = labels[i]
        plt.plot(time_array, max_height_values, marker=marker, linestyle='-', label=label)
    
    # Read and plot CSV data
    csv_data = read_csv(csv_file_path)
    csv_time_array, csv_pile_height_values = zip(*[(float(row[0]), float(row[2])) for row in csv_data])
    plt.plot(csv_time_array, csv_pile_height_values, 's-', label='Juez et al.(2014) Experiment')
    
    plt.grid(True)
    plt.legend()
    plt.title('Comparison of semispherical obstacle at probe location PD1')
    plt.xlabel('time')
    plt.ylabel('pile height')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script_name.py <hdf5_directory_1> <time_directory_1> [<hdf5_directory_2> <time_directory_2> ...] <csv_file_path>")
        sys.exit(1)
    
    # Collect alternating HDF5 and time directories from arguments
    hdf5_directories = sys.argv[1:-1:2]
    time_directories = sys.argv[2:-1:2]
    csv_file_path = sys.argv[-1]
    
    main(hdf5_directories, time_directories, csv_file_path)

