import h5py
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import geopandas as gpd
import sys
#from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
#from shapely.geometry import Point
import glob
import os
import re


 # Check if at least one argument is provided (excluding the script name)
if len(sys.argv) < 2:
    print("Error: Please provide a directory path as an argument.")
    exit(1)


# # Get the directory path from the command-line argument
directories = sys.argv[1]

# # Use glob to expand the wildcard pattern and get a list of matching file paths
files = glob.glob(os.path.join(directories + '/xdmf_p0000_i0000*.h5'))

# # Proceed only if HDF5 files are found
if not files:
    print("No HDF5 files found in the specified directory.")
    exit(1)

#files = ['xdmf_p0000_i00000000.h5',
         #'xdmf_p0000_i00000262.h5',
         #'xdmf_p0000_i00000400.h5',
         #'xdmf_p0000_i00000600.h5',
         #'xdmf_p0000_i00000800.h5',
         #'xdmf_p0000_i00001000.h5',
         #'xdmf_p0000_i00001200.h5',
         #'xdmf_p0000_i00001400.h5',
         #'xdmf_p0000_i00001600.h5',
         #'xdmf_p0000_i00001800.h5',
         #'xdmf_p0000_i00002000.h5',
         #'xdmf_p0000_i00002200.h5',
         #'xdmf_p0000_i00002400.h5',
         #'xdmf_p0000_i00002600.h5',
         #'xdmf_p0000_i00002800.h5',
         #]

fig, ax = plt.subplots(figsize=(10, 10))

for file_path in files:
    with h5py.File(file_path, 'r') as hdf5_file:
        print("Datasets in the HDF5 file:")
        for dataset_name in hdf5_file.keys():
            print(dataset_name)

    with h5py.File(file_path, 'r') as hdf5_file:
        group = hdf5_file['Mesh']

        print("Datasets in the 'Mesh' group:")
        for dataset_name in group.keys():
            print(dataset_name)
        
        specific_dataset_name = 'Points'  # Replace with the actual dataset name
        if specific_dataset_name in group:
            specific_dataset = group[specific_dataset_name]
            #print("Data in the specific dataset:")
            cord=specific_dataset[:]
        else:
            print(f"Dataset '{specific_dataset_name}' not found in the 'Mesh' group.")
        
        specific_dataset_name = 'Connections'  # Replace with the actual dataset name
        if specific_dataset_name in group:
             specific_dataset = group[specific_dataset_name]
             #print("Data in the specific dataset:")
             vertex=specific_dataset[:]
    
    with h5py.File(file_path, 'r') as hdf5_file:
        prop = hdf5_file['Properties']
        
        print("Datasets in the 'Properties' group:")
        for dataset_name in prop.keys():
            print(dataset_name)
        
        specific_dataset_name = 'PILE_HEIGHT'  # Replace with the actual dataset name
        if specific_dataset_name in prop:
            specific_dataset = prop[specific_dataset_name]
            height=specific_dataset[:]
        else:
            print(f"Dataset '{specific_dataset_name}' not found in the 'Properties' group.")
    
    with h5py.File(file_path, 'r') as hdf5_file:
        prop = hdf5_file['Properties']
        
        print("Datasets in the 'Properties' group:")
        for dataset_name in prop.keys():
            print(dataset_name)
        
        specific_dataset_name = 'YMOMENTUM'  # Replace with the actual dataset name
        if specific_dataset_name in prop:
            specific_dataset = prop[specific_dataset_name]
            mom_y=specific_dataset[:]
        else:
            print(f"Dataset '{specific_dataset_name}' not found in the 'Properties' group.")
        
    ds1= np.hstack((vertex, height, mom_y))
    new = []
    cordn = []
    center = []

    c1 = list(range(0, 5))
    c = np.array(c1)
    
    for i in range(len(cord)):
        if cord[i, 2] <= 1:
            c2 = cord[i, :]
            cordn.append(c2)
        cord_new = np.array(cordn)  # cord_new store the coordinates of base plate only.
            
    for i in range(len(cord_new)):
        if cord_new[i, 1] == 0.00:
            c3 = cord_new[i, :]
            center.append(c3)
        center_cord = np.array(center) # store the coordinates of the center line.

    for i in range(len(ds1)):
        if ds1[i, 4] > 0.01:
            data = ds1[i, :]
            new.append(data)
        news = np.array(new)

    xnew = []  # Initialize an empty list to store extracted x-coordinates
    ynew = []
    znew = []
    pile1 = []
    for i in range(len(news)):
        for j in range(len(c)):
            index = int(news[i, j])
            if 0 <= index < len(center_cord):
                xn = center_cord[index, 0]
                xnew.append(xn)
                yn = center_cord[index, 1]
                ynew.append(yn)
                zn = center_cord[index, 2]
                znew.append(zn)
                pile = news[index, 4]
                pile1.append(pile)
    x = np.array(xnew)
    y = np.array(ynew)
    z = np.array(znew)
    pile_h = np.array(pile1)
    height_info = np.vstack((x,y,pile_h))

#plt.plot(x, pile_h, '*')
#plt.grid(True)
#plt.show()


# ## code to extract data of time
# directories = [
#         'D:\IIT Mandi\scripts',
#         ]

# file_pattern = ['output_summary.-00001']

# time = []

# # Add debugging output
# print(f"Searching for files in directory: {directories}")

# # Function to convert time string to seconds
# def time_to_seconds(time_str):
#     # Split time string into hours, minutes, and seconds
#     match = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
#     if match:
#         hours, minutes, seconds = map(float, match.groups())
#         total_seconds = (hours * 3600) + (minutes * 60) + seconds
#         return total_seconds
#     else:
#         print(f"Invalid time format: {time_str}")
#         return None

# # Iterate over files and process data
# for directory in directories:
#     file_paths = glob.glob(os.path.join(directory, file_pattern[0]))
#     if len(file_paths) == 0:
#         print(f"No files found in directory: {directory}")
#         exit(1)
    
#     for file_path in file_paths:
#         print(f"Processing file: {file_path}")
#         # Read lines from the file
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#         # Extract time, velocity, height, and distance from each line
#         for line in lines:
#             match = re.match(r'(\d+:\d+:\d+(?:\.\d+)?) .* Vave=(\d+\.\d+).* hmax=(\d+\.\d+).* Xcen=(-?\d+\.\d+) .*', line)
#             if match:
#                 time_str, velocity_str, height_str, distance_str = match.groups()
#                 time_seconds = time_to_seconds(time_str)
#                 if time_seconds is not None:
#                     time.append(time_seconds)
#         time_array = np.array(time)




# plt.plot(time_array[0], pile_h, '--r')
# plt.grid(True)
# plt.legend()
# plt.show()
