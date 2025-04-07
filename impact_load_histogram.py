import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from joblib import Parallel, delayed
import matplotlib.patches as mpatches


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
    for i in range(len(ds1)):
        for j in range(4):
            index = int(ds1[i, j])
            if 0 <= index < len(cord):
                xnew.append(cord[index, 0])
                ynew.append(cord[index, 1])
                znew.append(cord[index, 2])
                pile1.append(ds1[i, 4])
    return np.array(xnew), np.array(ynew), np.array(znew), np.array(pile1)

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

# Modified function to collect maximum pile height at a specific location
def collect_max_pile_height_at_location(hdf5_file, target_x, target_y, tol):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    heights = [pile1[i] for i in range(len(xnew)) if abs(ynew[i] - target_y) < tol and abs(xnew[i] - target_x) < tol]
    return max(heights) if heights else 0  # Return 0 if no heights found

def main(hdf5_directory, time_directory):
    # Define obstacle positions - specify as many as you need
    row_1_positions = [(0.690078, 0.245902), (0.690078, 0.337156), (0.690078, 0.423202), (0.690078, 0.501422), (0.690078, 0.58225), (0.690078, 0.668292), (0.690078, 0.751727)]
    row_2_positions = [(0.744477, 0.204185), (0.744477, 0.28762), (0.744477, 0.371055), (0.744477, 0.457097), (0.744477, 0.54314), (0.744477, 0.629182), (0.744477, 0.71001), (0.744477, 0.793445)]
    row_3_positions = [(0.798877, 0.162462), (0.798877, 0.24851), (0.798877, 0.334552), (0.798877, 0.41538), (0.798877, 0.504025), (0.798877, 0.584857), (0.798877, 0.665685), (0.798877, 0.756942), (0.798877, 0.837769)]
    
    
    x_labels_row_1 = [0.2474, 0.3316, 0.4158, 0.5, 0.5842, 0.6684, 0.7526]
    x_labels_row_2 = [0.2053, 0.2895, 0.3737, 0.4579, 0.5421, 0.6263, 0.7105, 0.7947]
    x_labels_row_3 = [0.1632, 0.2474, 0.3316, 0.4158, 0.5, 0.5842, 0.6684, 0.7526, 0.8368]
    
    max_heights_row_1 = []
    max_heights_row_2 = []
    max_heights_row_3 = []
    max_load_row_1 = []
    max_load_row_2 = []
    max_load_row_3 = []
    tol = 1e-5  # Tolerance for checking coordinates

    # Collect maximum pile height for each obstacle position in both rows
    hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
    hdf5_files.sort()
    
    # Read time and velocity data
    time_files = glob.glob(os.path.join(time_directory, 'output_summary.-00001'))
    all_time_values, all_velocity_values, all_cos_angles_values = [], [], []
    for time_file in time_files:
        time_sorted, velocity, cos_angles = process_time_velocity(time_file)
        all_time_values.extend(time_sorted)
        all_velocity_values.extend(velocity)
        all_cos_angles_values.extend(cos_angles)

    # Ensure unique and sorted time values
    all_time_values = np.unique(all_time_values)
    all_velocity_values = np.array(all_velocity_values)
    all_cos_angles_values = np.array(all_cos_angles_values)


    for target_x, target_y in row_1_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_1.append(max_heights)
        
        for i, heights in enumerate(max_heights_row_1):
            froude_numbers, drag_coefficients, loads = [], [], []
            for vel, h, cos_angle in zip(all_velocity_values, max_heights, all_cos_angles_values):
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
            overall_max_load_1 = max(loads)
        max_load_row_1.append(overall_max_load_1)
        print('maximum load in row 1 =', max_load_row_1)

    for target_x, target_y in row_2_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_2.append(max_heights)
        
        for i, heights in enumerate(max_heights_row_2):
            froude_numbers, drag_coefficients, loads = [], [], []
            for vel, h, cos_angle in zip(all_velocity_values, max_heights, all_cos_angles_values):
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
            overall_max_load_2 = max(loads)
        max_load_row_2.append(overall_max_load_2)
        print('maximum load in row 2 =', max_load_row_2)
    
    for target_x, target_y in row_3_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_3.append(max_heights)
        
        for i, heights in enumerate(max_heights_row_3):
            froude_numbers, drag_coefficients, loads = [], [], []
            for vel, h, cos_angle in zip(all_velocity_values, max_heights, all_cos_angles_values):
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
            overall_max_load_3 = max(loads)
        max_load_row_3.append(overall_max_load_3)
        print('maximum load in row 3 =', max_load_row_3)

    # 3D Bar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    y_offset_row_1 = 0  # Offset for first row along y-axis
    y_offset_row_2 = -0.5  # Offset for second row along y-axis
    y_offset_row_3 = -1

    # Plot row 1
    ax.bar3d(x_labels_row_1, [y_offset_row_1] * len(x_labels_row_1), np.zeros(len(max_heights_row_1)),
         dx=0.03, dy=0.05, dz=max_load_row_1, color='skyblue', label="Row 1")
    
    # Plot row 2
    ax.bar3d(x_labels_row_2, [y_offset_row_2] * len(x_labels_row_2), np.zeros(len(max_heights_row_2)),
         dx=0.03, dy=0.05, dz=max_load_row_2, color='salmon', label="Row 2")
    
    ax.bar3d(x_labels_row_3, [y_offset_row_3] * len(x_labels_row_3), np.zeros(len(max_heights_row_3)),
         dx=0.03, dy=0.05, dz=max_load_row_3, color='Green', label="Row 3")

    ax.set_xlabel('Obstacle Location')
    #ax.set_ylabel('Row')
    ax.set_zlabel('Impact load')
    ax.set_yticks([y_offset_row_1, y_offset_row_2, y_offset_row_3])
    ax.set_yticklabels(['Row 1', 'Row 2', 'Row 3'])
    plt.title('Impact load at Obstacle Locations for Arrays of Mixed Obstacles.')
    
    # Create custom legend handles
    row1_patch = mpatches.Patch(color='skyblue', label='Row 1-Thin Cuboid')
    row2_patch = mpatches.Patch(color='salmon', label='Row 2-Diamond Cuboid')
    row3_patch = mpatches.Patch(color='green', label='Row 3-Cylinder')

    # Display the legend
    plt.legend(handles=[row1_patch, row2_patch, row3_patch])

    plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <hdf5_directory>")
        sys.exit(1)
    hdf5_directory = sys.argv[1]
    time_directory = sys.argv[2]
    main(hdf5_directory, time_directory)

