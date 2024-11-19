import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the original and processed datasets
original_file_path = 'data/mav0/state_groundtruth_estimate0/data.csv'
processed_file_path = 'processed_data_mav0_interpolated.csv'
# processed_file_path = 'processed_data_mav0_with_predictions.csv'

original_data = pd.read_csv(original_file_path)
processed_data = pd.read_csv(processed_file_path)

# Extract positions for analysis
original_positions = original_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
processed_positions = processed_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values

# Function to calculate Euclidean distance between x, y, z of prediction and ground-truth
def calculate_3d_distance(pred, truth):
    x_pred, y_pred, z_pred = pred[:3]
    x_true, y_true, z_true = truth[:3]
    return math.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2 + (z_pred - z_true) ** 2)

# Calculate positional differences using calculate_3d_distance
positional_differences = [
    calculate_3d_distance(pred, true)
    for pred, true in zip(processed_positions, original_positions[:len(processed_positions)])
]

# Calculate the average positional difference
avg_positional_difference = sum(positional_differences[5400:5600]) / len(positional_differences[5000:5600])

# Plot the data
plt.figure(figsize=(10, 6))

# Original data
# plt.plot(original_positions[5000:5600, 0], original_positions[5000:5600, 1], 'o-', label='Original Data', markersize=5)
plt.plot(original_positions[:, 0], original_positions[:, 1], 'o-', label='Original Data', markersize=5)

# Processed (Interpolated) data
# plt.plot(processed_positions[5000:5600, 0], processed_positions[5000:5600, 1], 'x-', label='Processed Data (Interpolated)', markersize=5)
plt.plot(processed_positions[:, 0], processed_positions[:, 1], 'x-', label='Processed Data (Interpolated)', markersize=5)

# Add labels, title, and legend for data comparison
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Comparison of Original and Processed (Interpolated) Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot positional differences
plt.figure(figsize=(10, 6))

# Positional differences
# plt.plot(positional_differences[5400:5600], label='Positional Difference', marker='o')
plt.plot(positional_differences[:], label='Positional Difference', marker='o')

# Add labels, title, and legend for positional differences
plt.xlabel('Index')
plt.ylabel('Positional Difference (Euclidean)')
plt.title('Positional Difference Between Original and Processed Data')
plt.legend()
plt.grid(True)
plt.show()

# Print the average positional difference
print(f"Average Positional Difference (Processed to Original): {avg_positional_difference:.4f}")
