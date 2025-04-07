import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os
#from latexify import latexify

# Check if at least two arguments are provided (script name and directory paths)
if len(sys.argv) < 4:
    print("Error: Please provide at least one directory path as an argument.")
    exit(1)

# Get the list of directories from the command-line arguments
directories = sys.argv[1:]

# File pattern to search for in each directory
file_pattern = 'output_summary.-00001'

# Arrays to hold data from each directory
time = []
velocity = []
distance = []
height = []
run_efficiency = []

# Define markers and colors for different directories
markers = ["^", "o", "s", "D", "v", "*", "x", "P", "h", "+"]
colors = ["b", "g", "r", "c", "m", "y", "k"]
labels = ["$\delta$ = 15$^\circ$", "$\delta$ = 20$^\circ$", "$\delta$ = 25$^\circ$"]

# Configure Matplotlib for clean and professional plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (10, 8),  # Adjust figure size as needed
    "lines.linewidth": 1.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})
# Loop through each directory
for dir_idx, directory in enumerate(directories):
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    
    if len(file_paths) == 0:
        print(f"No files found in directory: {directory}")
        continue  # Skip this directory if no matching files are found

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Read and process each dataset from the file
        # Read and sort time data
        datafile0 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[0], delimiter="=", engine='python')
        datafile0[0] = datafile0[0].str.replace('[^\d.]', '', regex=True)
        datafile0[0] = pd.to_numeric(datafile0[0], errors='coerce')
        time_sorted = np.sort(datafile0[0].to_numpy().astype(float))
        time.append(time_sorted)
        
        # Read velocity data
        datafile = pd.read_csv(file_path, header=None, skiprows=2, usecols=[2], delimiter="=", engine='python')
        datafile[2] = datafile[2].str.replace('[^\d.]', '', regex=True)
        datafile[2] = pd.to_numeric(datafile[2], errors='coerce')
        velocity.append(datafile.iloc[:, 0].to_numpy().astype(float))
        
        # Read height data
        datafile1 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[3], delimiter="=", engine='python')
        datafile1[3] = datafile1[3].str.replace('[^\d.]', '', regex=True)
        datafile1[3] = pd.to_numeric(datafile1[3], errors='coerce')
        height.append(datafile1.iloc[:, 0].to_numpy().astype(float))
        
        # Read X and Y center data to calculate distance
        datafile3 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[4], delimiter="=", engine='python')
        datafile3[4] = datafile3[4].str.replace('[^\d.]', '', regex=True)
        datafile3[4] = pd.to_numeric(datafile3[4], errors='coerce')
        x = datafile3.iloc[:, 0].to_numpy().astype(float)
        
        datafile2 = pd.read_csv(file_path, header=None, skiprows=2, usecols=[5], delimiter="=", engine='python')
        datafile2[5] = datafile2[5].str.replace('[^\d.]', '', regex=True)
        datafile2[5] = pd.to_numeric(datafile2[5], errors='coerce')
        y = datafile2.iloc[:, 0].to_numpy().astype(float)
        
        # Calculate relative and cumulative distances
        xdiff = np.diff(x)
        ydiff = np.diff(y * 1000000)  # Scaled Y values
        rel_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))
        rel_dist = np.insert(rel_dist, 0, 0)
        dist = np.cumsum(rel_dist)
        distance.append(dist)
        
        # Calculate run-out efficiency
        run_out_efficiency = (x[-1] - x[0]) / height[0]  # Avoid division by zero
        run_efficiency.append(run_out_efficiency)

# Plotting all data on a single plot
fig, ax = plt.subplots()

# Iterate over each directory's data to plot with different markers and labels
for i in range(len(velocity)):
    marker_style = markers[i % len(markers)]
    color_style = colors[i % len(colors)]
    label_style = labels[i % len(labels)]
    ax.plot(distance[i], velocity[i], marker=marker_style, color=color_style, linestyle='-', label=label_style)

# Customize and show the plot
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Mean Velocity (m/s)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("plot.pdf", transparent=True, bbox_inches="tight", pad_inches=0.01)  # Save as a PDF with transparency
plt.show()
