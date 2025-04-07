import numpy as np

# Define the polynomial coefficients
coefficients = [0.1086, -0.5179, 1.3434, -1.6513, 0.89230]

# Define the polynomial function
def polynomial(x):
    return np.polyval(coefficients, x)

# Input: Desired x-values
x_values = [0.1, 0.13441, 0.21703, 0.321735, 0.434702, 0.538733, 0.623178, 0.683975, 0.719948, 0.733201, 0.734952, 0.735585, 0.736178, 0.736761, 0.737325, 0.737876, 0.738414, 0.738937]  # Example x-values; modify as needed

# Calculate the corresponding z-values
z_values = [polynomial(x) for x in x_values]

# Calculate delta_x and delta_z
delta_x = np.diff(x_values)  # Difference between successive x-values
delta_z = np.diff(z_values)  # Difference between successive z-values

# Calculate delta_z / delta_x (slope)
slopes = delta_z / delta_x

# Calculate the inverse tangent of the slopes to get the angle in radians
angles = np.arctan(slopes)

# Convert angles to degrees if desired
angles_degrees = np.degrees(angles)

# Print results
print("x values:", x_values)
print("z values:", z_values)
print("Delta x values:", delta_x)
print("Delta z values:", delta_z)
print("Slopes (Delta z / Delta x):", slopes)
print("Angles (radians):", angles)
print("Angles (degrees):", angles_degrees)

# Optional: Display results in a tabular format
print("\nResults:")
print("Index\tDelta x\tDelta z\tSlope (dz/dx)\tAngle (rad)\tAngle (deg)")
for i, (dx, dz, slope, angle_rad, angle_deg) in enumerate(zip(delta_x, delta_z, slopes, angles, angles_degrees)):
    print(f"{i} -> {i+1}\t{dx:.5f}\t{dz:.5f}\t{slope:.5f}\t{angle_rad:.5f}\t{angle_deg:.2f}")
