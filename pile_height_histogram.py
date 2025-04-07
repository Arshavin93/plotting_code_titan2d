import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from joblib import Parallel, delayed
import matplotlib.patches as mpatches

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

# Modified function to collect maximum pile height at a specific location
def collect_max_pile_height_at_location(hdf5_file, target_x, target_y, tol):
    cord, vertex, height = read_hdf5(hdf5_file)
    xnew, ynew, znew, pile1 = extract_data(cord, vertex, height)
    heights = [pile1[i] for i in range(len(xnew)) if abs(ynew[i] - target_y) < tol and abs(xnew[i] - target_x) < tol]
    return max(heights) if heights else 0  # Return 0 if no heights found

def main(hdf5_directory):
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
    tol = 1e-5  # Tolerance for checking coordinates

    # Collect maximum pile height for each obstacle position in both rows
    hdf5_files = glob.glob(os.path.join(hdf5_directory, '*.h5'))
    hdf5_files.sort()

    for target_x, target_y in row_1_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_1.append(np.max(max_heights))

    for target_x, target_y in row_2_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_2.append(np.max(max_heights))
    
    for target_x, target_y in row_3_positions:
        max_heights = Parallel(n_jobs=6)(delayed(collect_max_pile_height_at_location)(hdf5_file, target_x, target_y, tol) for hdf5_file in hdf5_files)
        max_heights_row_3.append(np.max(max_heights))

    # 3D Bar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    y_offset_row_1 = 0  # Offset for first row along y-axis
    y_offset_row_2 = -0.5  # Offset for second row along y-axis
    y_offset_row_3 = -1

    # Plot row 1
    ax.bar3d(x_labels_row_1, [y_offset_row_1] * len(x_labels_row_1), np.zeros(len(max_heights_row_1)),
         dx=0.03, dy=0.05, dz=max_heights_row_1, color='skyblue', label="Row 1")
    
    # Plot row 2
    ax.bar3d(x_labels_row_2, [y_offset_row_2] * len(x_labels_row_2), np.zeros(len(max_heights_row_2)),
         dx=0.03, dy=0.05, dz=max_heights_row_2, color='salmon', label="Row 2")
    
    ax.bar3d(x_labels_row_3, [y_offset_row_3] * len(x_labels_row_3), np.zeros(len(max_heights_row_3)),
         dx=0.03, dy=0.05, dz=max_heights_row_3, color='Green', label="Row 3")

    ax.set_xlabel('Obstacle Location')
    #ax.set_ylabel('Row')
    ax.set_zlabel('Pile Height')
    ax.set_yticks([y_offset_row_1, y_offset_row_2, y_offset_row_3])
    ax.set_yticklabels(['Row 1', 'Row 2', 'Row 3'])
    plt.title('Pile Heights at Obstacle Locations for Arrays of Mixed Obstacles.')
    
    # Create custom legend handles
    row1_patch = mpatches.Patch(color='skyblue', label='Row 1-Diamond Cuboid')
    row2_patch = mpatches.Patch(color='salmon', label='Row 2-Thin Cuboid')
    row3_patch = mpatches.Patch(color='green', label='Row 3-Cylindrical')

    # Display the legend
    plt.legend(handles=[row1_patch, row2_patch, row3_patch])

    plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <hdf5_directory>")
        sys.exit(1)
    hdf5_directory = sys.argv[1]
    main(hdf5_directory)

