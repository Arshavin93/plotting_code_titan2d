#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 09:41:26 2024

@author: arshavin
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os
import h5py
from joblib import Parallel, delayed
from scipy.integrate import quad

# Gravitational constant and density
g = 9.81  # Gravitational acceleration
rho = 2890  # Density (kg/m³)
H = 0.06 # height of the obstacles

# Function to read HDF5 file and extract required datasets
def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        cord = hdf5_file['Mesh/Points'][:]
        vertex = hdf5_file['Mesh/Connections'][:]
        height = hdf5_file['Properties/PILE_HEIGHT'][:]
        #mom_x = hdf5_file['Properties/XMOMENTUM'][:]
        mom_y = hdf5_file['Properties/YMOMENTUM'][:]
    return cord, vertex, height, mom_y

# Function to extract required data from coordinates and vertex arrays
def extract_data(cord, vertex, height, mom_y):
    ds1 = np.hstack((vertex, height, mom_y))
    xnew, ynew, znew, pile1, y_mom = [], [], [], [], []
    c1 = list(range(0, 5))
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
                ym = ds1[index, 5]
                y_mom.append(ym)
    return np.array(xnew), np.array(ynew), np.array(znew), np.array(pile1), np.array(y_mom)

# Function to process HDF5 file for specific target y range
def process_hdf5_file(hdf5_file, target_x, target_y_range, tol):
    cord, vertex, height, mom_y = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1, y_mom = extract_data(cord, vertex, height, mom_y)

    y_velocity_dict = {}

    for i in range(len(xnew)):
        if abs(xnew[i] - target_x) < tol:
            if target_y_range[0] <= ynew[i] <= target_y_range[1]:
                y_value = ynew[i]
                if y_value not in y_velocity_dict:
                    y_velocity_dict[y_value] = {'height_info': [], 'y_momentum_info': []}
                y_velocity_dict[y_value]['height_info'].append(pile1[i])
                y_velocity_dict[y_value]['y_momentum_info'].append(y_mom[i])

    # For each y_value, keep only the maximum height and maximum momentum
    for y_value, data in y_velocity_dict.items():
        height_info = data['height_info']
        y_momentum_info = data['y_momentum_info']

        if height_info and y_momentum_info:
            # Get the maximum values of height and momentum
            max_height = np.max(height_info)
            max_momentum = np.max(y_momentum_info)

            # Store only the maximum height and momentum
            y_velocity_dict[y_value] = {
                'height_info': [max_height],
                'y_momentum_info': [max_momentum]
            }

    return y_velocity_dict

# Function to square a polynomial and integrate it
def integrate_polynomial(coefficients, lower_limit, upper_limit):
    polynomial = np.poly1d(coefficients)
    # Define the square of the polynomial function
    square_polynomial = lambda y: (polynomial(y))**2
    # Perform the integration
    integral, _ = quad(square_polynomial, lower_limit, upper_limit)
    return integral


# Main function
def main(hdf5_directory): 
    target_x = 0.690078  # Specify the target x coordinate
    tol = 1e-5  # Tolerance for checking x coordinate
    target_y_range = (0.49, 0.50)  # Define the range of target_y values

    # Search for HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
    hdf5_files.sort()  # Sort the list of file paths

    # Dictionary to store y-velocities for each unique y-value
    all_y_velocities = {}

    # Process HDF5 files in parallel
    results = Parallel(n_jobs=6)(
        delayed(process_hdf5_file)(hdf5_file, target_x, target_y_range, tol) for hdf5_file in hdf5_files
    )

    # Combine results
    for result in results:
        for y_value, data in result.items():
            if y_value not in all_y_velocities:
                all_y_velocities[y_value] = {'height_info': [], 'y_momentum_info': []}
            all_y_velocities[y_value]['height_info'].extend(data['height_info'])
            all_y_velocities[y_value]['y_momentum_info'].extend(data['y_momentum_info'])

    # Calculate velocities for each unique y-value
    y_velocity_results = {}
    max_elements = 0  # Track the maximum number of elements in any velocity array

    for y_value, data in all_y_velocities.items():
        height_info = data['height_info']
        y_momentum_info = data['y_momentum_info']

        if height_info and y_momentum_info:
            y_velocity = [
                ym / h if h > 0 else 0.00 for h, ym in zip(height_info, y_momentum_info)
            ]
            y_velocity_results[y_value] = y_velocity
            max_elements = max(max_elements, len(y_velocity))

    # Create arrays to store velocity elements based on their position
    velocity_arrays = [[] for _ in range(max_elements)]
    y_values = sorted(y_velocity_results.keys())

    # Populate velocity_arrays with elements from y_velocity_results
    for idx, velocities in enumerate(y_velocity_results.values()):
        for i, velocity in enumerate(velocities):
            velocity_arrays[i].append(velocity)

    # Fit polynomials to each velocity array
    polynomial_fits = {}
    degree = 3  # Degree of the polynomial

    for idx, velocity_array in enumerate(velocity_arrays):
        if velocity_array:  # Only fit if the array is not empty
            coefficients = np.polyfit(y_values[:len(velocity_array)], velocity_array, degree)
            polynomial_fits[idx] = coefficients
            print(f"Polynomial fit for array {idx+1}: {np.poly1d(coefficients)}")
    
    # Integration limits
    lower_limit = 0
    upper_limit = 0.0334
    
    # # Integrate each polynomial
    # integrals = {}
    # for idx, coefficients in polynomial_fits.items():
    #     integral = integrate_polynomial(coefficients, lower_limit, upper_limit)
    #     integrals[idx] = integral
    #     print(f"Integral of the square of polynomial fit for array {idx+1} from {lower_limit} to {upper_limit}: {integral}")
    
    # Calculate impact load and impact pressure for each polynomial fit
    impact_loads = {}
    impact_pressures = {}
    for idx, coefficients in polynomial_fits.items():
        # Integrate the squared polynomial
        integral = integrate_polynomial(coefficients, lower_limit, upper_limit)
        # Calculate impact load
        impact_load = (0.5 * rho * H * integral)/1000
        impact_loads[idx] = impact_load
        impact_pressure = impact_load/(0.0334 * H)
        impact_pressures[idx] = impact_pressure
        print(f"Impact load for array {idx+1}: {impact_load} (KN)")
        print(f"Impact pressure for array {idx+1}: {impact_pressure} (KPa)")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <hdf5_directory_1> <hdf5_directory_2> ...")
        sys.exit(1)
    
    # Pass directories as arguments
    hdf5_directories = sys.argv[1:]
    
    for directory in hdf5_directories:
        print(f"Processing directory: {directory}")
        main(directory)
