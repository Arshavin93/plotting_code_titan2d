import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os
import h5py
from joblib import Parallel, delayed

# Polynomial coefficients and gravitational constant
coefficients = [0.1086, -0.5179, 1.3434, -1.6513, 0.89230]
g = 9.81  # Gravitational acceleration
rho = 2890  # Density (kg/m³)

# Polynomial function for terrain slope
def polynomial(x):
    return np.polyval(coefficients, x)

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

    return h_max

# Function to process time and velocity from text files
def process_time_velocity(file_path):
    datafile0 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[0], delimiter="=", engine='python')
    datafile0[0] = datafile0[0].str.replace('[^\d.]', '', regex=True)
    datafile0[0] = pd.to_numeric(datafile0[0], errors='coerce')
    time_sorted = datafile0[0].to_numpy().astype(float)

    datafile = pd.read_csv(file_path, header=None, skiprows=2, usecols=[2], delimiter="=", engine='python')
    datafile[2] = datafile[2].str.replace('[^\d.]', '', regex=True)
    datafile[2] = pd.to_numeric(datafile[2], errors='coerce')
    velocity = datafile.iloc[:, 0].to_numpy().astype(float)
    
    # Read X values
    datafile3 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[4], delimiter="=", engine='python')
    datafile3[4] = datafile3[4].str.replace('[^\d.]', '', regex=True)
    datafile3[4] = pd.to_numeric(datafile3[4], errors='coerce')
    x_values = datafile3.iloc[:, 0].to_numpy().astype(float)
    
    # Calculate slopes and angles
    z_values = polynomial(x_values)
    delta_x = np.diff(x_values)
    delta_x[-1] = delta_x[-2]
    delta_z = np.diff(z_values)
    slopes = np.append(delta_z / delta_x, 0)  # Append last slope for equal lengths
    angles = np.arctan(slopes)
    cos_angles = np.cos(angles)

    return time_sorted, velocity, cos_angles


def main(directory_pairs):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define target positions and tolerances
    target_positions = [
        {"x": 0.690078, "y_range": [0.245, 0.752]},
        {"x": 0.744477, "y_range": [0.204, 0.794]},
        {"x": 0.798877, "y_range": [0.162, 0.838]},
    ]
    tol = 1e-5  # Tolerance for checking coordinates
    number_of_rows = [1, 2, 3]
    
    markers = ["o", "s", "D", "^", "v", "*", "x", "P", "h", "+"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    labels = ['R1cy_R2dia_R3thin', 'R1cy_R2thin_R3dia', 'R1dia_R2cy_R3thin', 'R1dia_R2thin_R3cy', 'R1thin_R2cy_R3dia', 'R1thin_R2dia_R3cy']    


    # Process each pair of HDF5 and time directories
    for pair_idx, (hdf5_directory, time_directory) in enumerate(directory_pairs):
        label = labels[pair_idx % len(labels)]  # Assign label from the list
        marker = markers[pair_idx % len(markers)]  # Assign marker from the list
        
        
        # Search for HDF5 files
        hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
        hdf5_files.sort()  # Ensure files are processed in order

        # Read time and velocity data
        time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
        all_velocity_values, all_cos_angles_values = [], []
    
        for time_file in time_files:
            _, velocity, cos_angles = process_time_velocity(time_file)
            all_velocity_values.extend(velocity)
            all_cos_angles_values.extend(cos_angles)
    
        all_velocity_values = np.array(all_velocity_values)
        all_cos_angles_values = np.array(all_cos_angles_values)
    
        # Array to store maximum load values for each target
        max_load_values = []
    
        # Calculate maximum pile heights for all targets
        max_height_values = []
        for position in target_positions:
            heights = Parallel(n_jobs=6)(
                delayed(process_hdf5_file_at_location)(
                    hdf5_file, position["x"], position["y_range"][0], position["y_range"][1], tol
                ) for hdf5_file in hdf5_files
            )
            max_height_values.append(heights)
    
            # Overall maximum height for the current target position
            overall_max_height = max(heights)
            print('maximum height = ', overall_max_height)
    
            # Calculate the maximum load for this target position
            for i, heights in enumerate(max_height_values):
                froude_numbers, drag_coefficients, loads = [], [], []
                for vel, h, cos_angle in zip(all_velocity_values, heights, all_cos_angles_values):
                    if g * h * cos_angle > 0:
                        froude_number = vel / np.sqrt(g * h * cos_angle)
                    else:
                        froude_number = 0
                    froude_numbers.append(froude_number)
    
                    if froude_number > 0:
                        drag_coefficient = 9 * (froude_number ** -1.2)
                    else:
                        drag_coefficient = 0
                    drag_coefficients.append(drag_coefficient)
    
                    if drag_coefficient >= 0:
                        load = 0.5 * drag_coefficient * rho * (vel ** 2)
                    else:
                        load = 0
                    loads.append(load)
                overall_max_load = max(loads)
            max_load_values.append(overall_max_load)
            print('overall maximum load =  ', max_load_values)
    
        
        # Plot the impact load vs. time
        #marker = markers[i % len(markers)]
        #label = labels[i % len(labels)]
        plt.plot(number_of_rows, max_load_values, marker=marker, linestyle='-', label=label)
    
    plt.grid(True)
    plt.legend()
    plt.title('Maximum impact load in each row for different obstacle shapes')
    plt.xlabel('Number of rows')
    plt.ylabel('Maximum Impact Load (N)')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 13 or len(sys.argv) % 2 != 1:
        print("Usage: python script_name.py <hdf5_directory_1> <time_directory_1> ... <hdf5_directory_n> <time_directory_n>")
        sys.exit(1)

    # Parse directory pairs
    directory_pairs = [
        (sys.argv[i], sys.argv[i+1]) for i in range(1, len(sys.argv), 2)
    ]

    main(directory_pairs)
