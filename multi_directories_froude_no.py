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

def process_hdf5_file(hdf5_file, target_y, tol):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    height_info = []

    for i in range(len(xnew)):
        if abs(ynew[i] - target_y) < tol:
        #if ynew[i] > target_y - 0.005 and ynew[i] <= target_y:
            if xnew[i] < 0.761 and xnew[i] > 0.75:
            #if abs(xnew[i] - (0.511339)) < tol:
                #if 0.001 < pile1[i] < 0.015:
                height_info.append(pile1[i])

    if height_info:
        h_max = np.max(height_info)
    else:
        h_max = 0.00

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


# Main function
def main(hdf5_directories, time_directories):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    max_height_values = []
    target_y = 0.501422  # Specify the target y coordinate
    tol = 1e-5  # Tolerance for checking y coordinate
    
    # Ensure each HDF5 directory has a corresponding time directory
    if len(hdf5_directories) != len(time_directories):
        print("Mismatch between the number of HDF5 directories and time directories.")
        sys.exit(1)
    
    # Define markers and labels for each dataset
    markers = ['o', 'x', '*', 's', 'd', '^', 'v', '<', '>', 'p']
    labels = ['R1cy_R2dia_R3thin', 'R1cy_R2thin_R3dia', 'R1dia_R2cy_R3thin', 'R1dia_R2thin_R3cy', 'R1thin_R2cy_R3dia', 'R1thin_R2dia_R3cy']
    
    # Process each HDF5 directory with its corresponding time directory
    for i, (hdf5_directory, time_directory) in enumerate(zip(hdf5_directories, time_directories)):
        # Process HDF5 files
        hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
        hdf5_files.sort()
    
        # Parallel processing of HDF5 files to extract pile height information
        max_height_values = Parallel(n_jobs=6)(
            delayed(process_hdf5_file)(hdf5_file, target_y, tol) for hdf5_file in hdf5_files)

        # Process text files for time and velocity
        time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
        all_time_values = []
        all_velocity_values = []
        all_cos_angles_values = []
        all_froude_numbers = []
        all_drag_coefficients = []
        all_loads = []
    
        for idx, time_file in enumerate(time_files):
            time_sorted, velocity, cos_angles = process_time_velocity(time_file)
            # Prevent consecutive duplicates for time, velocity, and cos_angles
            for t, vel, cos_angle in zip(time_sorted, velocity, cos_angles):
                if not (all_time_values and t == all_time_values[-1]):
                   all_time_values.append(t)
                if not (all_velocity_values and vel == all_velocity_values[-1]):
                   all_velocity_values.append(vel)
                if not (all_cos_angles_values and cos_angle == all_cos_angles_values[-1]):
                   all_cos_angles_values.append(cos_angle)
    
            
        for vel, h, cos_angle in zip(velocity, max_height_values, cos_angles):
            if g * h * cos_angle > 0:
                froude_number = vel / np.sqrt(g * h * cos_angle)
                all_froude_numbers.append(froude_number)  # Append here
            else:
                froude_number = 0
                all_froude_numbers.append(froude_number)
        
            # Calculate Coefficient of Drag
            if froude_number > 0:
                drag_coefficient = 9 * (froude_number ** -1.2)
            else:
                drag_coefficient = 0
            all_drag_coefficients.append(drag_coefficient)

            # Calculate Impact Load
            if drag_coefficient >= 0:
                load = 0.5 * drag_coefficient * rho * (vel ** 2)
            else:
                load = 0
            all_loads.append(load)
    
        print("Array length of time:", len(all_time_values))
        print("Array length of velocity:", len(all_velocity_values))
        print("Array length of height values:", len(max_height_values))    
        # Print the array after the loop
        print("Array length of Froude Numbers:", len(all_froude_numbers))
        print("Array length of Drag Coefficients:", len(all_drag_coefficients))
        print("Array length of Loads:", len(all_loads))
        marker = markers[i % len(markers)]  # Cycle through markers if more directories than markers
        label = labels[i]
        
        plt.plot(np.array(all_time_values), np.array(all_drag_coefficients), marker=marker, linestyle='', label=label)
        
    plt.grid(True)
    plt.legend()
    plt.title('Comparision of drag coefficient with time for array of mixed obstacles.')
    plt.xlabel('time')
    plt.ylabel('Drag coefficient')
    #plt.savefig(output_plot_path)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 12:
        print("Usage: python script_name.py <hdf5_directory_1> <time_directory_1> [<hdf5_directory_2> <time_directory_2> ..")
        sys.exit(1)
    
    # Collect alternating HDF5 and time directories from arguments
    hdf5_directories = sys.argv[1::2]
    time_directories = sys.argv[2::2]
    main(hdf5_directories, time_directories)
