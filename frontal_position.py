import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# #import geopandas as gpd
# #import sys
# #from matplotlib.colors import ListedColormap
# #from matplotlib.patches import Polygon
# #from shapely.geometry import Point
import glob
import os
import re


files = ['/home/ashish/analytical_simulation/obstacle_interaction_sim/sim_semisphere/sim_paraboloid_pile/sim_2_bed_37_cell_60/vizout/xdmf_p0000_i00001340.h5',
          # 'xdmf_p0000_i00000185.h5',
          ]

fig, ax = plt.subplots(figsize=(10, 10))

for file_path in files:
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]

    c1 = list(range(0, 4))
    c = np.array(c1)
    

    ds1 = np.hstack((vertex, height))
    xnew, ynew, znew, pile1 = [], [], [], []
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
#         #         if pile == 0.1:
#         #             print('X = ', xn)
#         #             break
#         #         break
#         #     break
#         # break
    
    x = np.array(xnew)
    y = np.array(ynew)
    z = np.array(znew)
    pile_h = np.array(pile1)
    h1 = np.vstack((x, y, z, pile_h))
    height_info = np.transpose(h1)
    h_new = []
    h_center = []
    frontX = []
    max_height = []
    h_max= []
    
    for i in range(len(xnew)):
        if pile1[i] > 0.001:
            if ynew[i] < 0.51 and ynew[i] < 0.49:
                frontX.append(xnew[i])
                front_x = np.array(frontX)
                
    front_x_max = np.max(front_x)
    print('Position of the front is = ', front_x_max)
                
#     # for i in range(len(height_info)):
#     #     if height_info[i, 2] >= 50.07:
#     #         if height_info[i, 0] == 0.00:
#     #             if height_info[i, 3] >= 0.00:
#     #                 H = height_info[i, 3]
#     #                 max_height.append(H)
#     #                 if H:
#     #                     h_max = max(H)
    
#        # else:
#     #     # If no heights satisfy the conditions, set h_max to 0
#     #     h_max = 0
    # tol = 1e-5
    # for i in range(len(height_info)):
    #     if height_info[i, 1] > 0.495 and height_info[i, 1] < 0.505:
    #         if height_info[i, 0] > 0.495 and height_info[i, 0] < 0.505:
    #         #if abs(height_info[i, 0] - (0.501934)) < tol:
    #             #if height_info[i, 3] > 0.001:
    #                 c2 = height_info[i, :]
    #                 h_new.append(c2)
    #                 height_new = np.array(h_new)  # cord_new store the coordinates of base plate only.
    #                 h_max = np.max(height_new[:, 3])
    # max_height.append(h_max)
    
    #plt.plot(height_new[:,0], height_new[:,2], '--')
    #plt.show()
        
        
    
# #plt.plot(x, , '*')
# #plt.grid(True)
# #plt.show()


# code to extract data of time
directories = [
        'D:\IIT Mandi\scripts',
        ]

file_pattern = ['output_summary.-00001']

time = [0]

# Add debugging output
print(f"Searching for files in directory: {directories}")

# Function to convert time string to seconds
def time_to_seconds(time_str):
    # Split time string into hours, minutes, and seconds
    match = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
    if match:
        hours, minutes, seconds = map(float, match.groups())
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        return total_seconds
    else:        print(f"Invalid time format: {time_str}")
       return None

# Iterate over files and process data
for directory in directories:
    file_paths = glob.glob(os.path.join(directory, file_pattern[0]))
    if len(file_paths) == 0:
        print(f"No files found in directory: {directory}")
        exit(1)
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        # Read lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Extract time, velocity, height, and distance from each line
        for line in lines:
            match = re.match(r'^(\d+:\d+:\d+(?:\.\d+)?) .* Vave=(\d+\.\d+).* hmax=(\d+\.\d+).* Xcen=(-?\d+\.\d+) .*', line)
            if match:
                time_str, velocity_str, height_str, distance_str = match.groups()
                time_seconds = time_to_seconds(time_str)
                time.append(time_seconds)
        time_array = np.array(time)




plt.plot(time_array, front_x , '*')
plt.grid(True)
#plt.legend()
plt.show()
