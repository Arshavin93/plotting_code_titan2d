# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:10:56 2024

@author: visha
"""
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

# Function to read data from CSV file
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            data.append(row)
    return data

def process_hdf5_file(hdf5_file, target_pile):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    frontX = []

    for i in range(len(xnew)):
        if pile1[i] >= target_pile:
            if ynew[i] > 0.47 and ynew[i] < 0.53:
                frontX.append(xnew[i])
                front_x = np.array(frontX)

    if frontX:
        x_frontal = np.max(front_x)
    else:
        x_frontal = 0.00

    return x_frontal

def main(hdf5_directories, time_directories):#, csv_directory):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    target_pile = 0.001  # Specify the pile height 
    
    labels = ["Simulation data bed friction 31", "Simulation data bed friction 33"]  # Add more labels if needed

    for idx, (hdf5_directory, time_directory) in enumerate(zip(hdf5_directories, time_directories)):
        max_front_x = []
    
        # Search for HDF5 files in the HDF5 directory
        hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
        hdf5_files.sort()  # Sort the list of file paths

        max_front_x = Parallel(n_jobs=4)(delayed(process_hdf5_file)(hdf5_file, target_pile) for hdf5_file in hdf5_files)
    
        print(f"Position of the frontal for {labels} is = ", max_front_x)
    
        # Search for time file in the time directory
        time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
        if not time_files:
            print("No time file found in directory.")
            sys.exit(1)
        time_file_path = time_files[0]
    
        time_array = process_output_summary(time_file_path)
        
        plt.plot(time_array, np.array(max_front_x), '*', label=labels[idx])
    
    # Read data from CSV file
    #csv_file_paths = glob.glob(os.path.join(csv_directory, 'hemi_piledata.csv'))
    #if not csv_file_paths:
    #    print("No CSV file found in directory.")
    #    sys.exit(1)
    #csv_file_path = csv_file_paths[0]
    #csv_data = read_csv(csv_file_path)

    #Extract relevant columns from CSV data
    #csv_time_array = []
    #csv_pile_height_values = []
    #for row in csv_data:
    #    time_value = float(row[0])  # Assuming time is in the first column
    #    pile_height_value = float(row[1])  # Assuming pile height is in the second column
    #    csv_time_array.append(time_value)
    #    csv_pile_height_values.append(pile_height_value)

    #Plot CSV data
    #plt.plot(csv_time_array, csv_pile_height_values, 'x', label='Juez et al.(2014)')
    #output_plot_path = os.path.join(time_directory, 'pile_height_vs_time.png')
    
    #ax.set_xlim(0.0, 2.0)
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('frontal position')
    #plt.savefig(output_plot_path)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script_name.py <hdf5_directory_1> <time_directory_1> <hdf5_directory_2> <time_directory_2>")
        sys.exit(1)
    hdf5_directories = [sys.argv[1], sys.argv[3]]
    time_directories = [sys.argv[2], sys.argv[4]]
    #csv_directory = sys.argv[3]
    main(hdf5_directories, time_directories)#, csv_directory)
