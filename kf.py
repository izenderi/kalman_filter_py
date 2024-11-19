import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

# Load the dataset
file_path = 'data/mav0/state_groundtruth_estimate0/data.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

t1 = time.time()

# Extract relevant columns
positions = data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
velocities = data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].values
accelerations = data[[' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']].values

# Use a subset of the data for Kalman filter
observed_data = positions[5000:5400]
observed_velocity = velocities[5000:5400]
observed_acceleration = accelerations[5000:5400]

# Ground-truth data (next 200 observations from the dataset)
ground_truth = positions[5400:5600]

# EKF Parameters
dt = 1.0  # time step (assuming uniform time step)
state_dim = 9  # State vector size: 3 position + 3 velocity + 3 acceleration

# Non-linear state transition function
def f(x):
    new_x = np.zeros(state_dim)
    new_x[:3] = x[:3] + x[3:6] * dt + 0.5 * x[6:] * dt**2  # Position update
    new_x[3:6] = x[3:6] + x[6:] * dt  # Velocity update
    new_x[6:] = x[6:]  # Keep acceleration constant
    return new_x

# Jacobian of the state transition function
def F_jacobian():
    F = np.eye(state_dim)
    F[:3, 3:6] = np.eye(3) * dt  # Derivative of position wrt velocity
    F[:3, 6:] = 0.5 * np.eye(3) * dt**2  # Derivative of position wrt acceleration
    F[3:6, 6:] = np.eye(3) * dt  # Derivative of velocity wrt acceleration
    return F

# Observation function (we observe position only)
def h(x):
    return x[:3]

# Jacobian of the observation function
def H_jacobian():
    H = np.zeros((3, state_dim))
    H[:3, :3] = np.eye(3)
    return H

# Process noise covariance
Q = np.eye(state_dim) * 0.01

# Measurement noise covariance
R = np.eye(3) * 0.1

# Initial state estimate
x = np.zeros(state_dim)
x[:3] = observed_data[0]  # Initial position
x[3:6] = observed_velocity[0]  # Initial velocity
x[6:] = observed_acceleration[0]  # Initial acceleration

# Initial covariance estimate
P = np.eye(state_dim)

# Multi-State Update: Sequentially update the EKF with all 100 observations
for i in range(1, len(observed_data)):
    # Prediction step
    F = F_jacobian()
    x = f(x)  # Predict the next state
    P = F @ P @ F.T + Q  # Update the covariance

    # Measurement update
    z = observed_data[i]
    H = H_jacobian()
    y = z - h(x)  # Residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

    x = x + K @ y  # Update the state estimate
    P = (np.eye(state_dim) - K @ H) @ P  # Update the covariance

# Store predictions
predictions = []

# Predict the next 100 steps
for _ in range(50):
    F = F_jacobian()
    x = f(x)
    P = F @ P @ F.T + Q
    predictions.append(x[:3].copy())

# Convert predictions to numpy array
predictions = np.array(predictions)

print(f"Duration of the MSEKF: {time.time() - t1:.6f} seconds")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(predictions[:, 0], predictions[:, 1], 'o-', label='Predictions (MSEKF)', markersize=5)
plt.plot(observed_data[:, 0], observed_data[:, 1], 'x-', label='Observed Data (Last 100)', markersize=5)
plt.plot(ground_truth[:, 0], ground_truth[:, 1], 's-', label='Ground Truth (Next 100)', markersize=5)

# Add labels, title, and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Visualization of MSEKF Predictions, Observed Data, and Ground Truth')
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate Euclidean distance between x, y, z of prediction and ground-truth
def calculate_3d_distance(pred, truth):
    x_pred, y_pred, z_pred = pred[:3]
    x_true, y_true, z_true = truth[:3]
    return math.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2 + (z_pred - z_true) ** 2)

# Calculate average error over all predictions
distances = [calculate_3d_distance(pred, true) for pred, true in zip(predictions, ground_truth)]
avg_error = sum(distances) / len(distances)

print("Average 3D Error between predictions and ground truth:", avg_error)