import numpy as np
import pandas as pd
import math
import time

# Load the dataset
file_path = 'data/mav0/state_groundtruth_estimate0/data.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Parameters
n = 15  # Number of rows for observed and predicted data
start = 1000  # Starting row for prediction
dt = 1.0  # time step (assuming uniform time step)
state_dim = 9  # State vector size: 3 position + 3 velocity + 3 acceleration

# Extract relevant columns
positions = data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
velocities = data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].values
accelerations = data[[' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']].values

# Kalman filter functions
def f(x):
    new_x = np.zeros(state_dim)
    new_x[:3] = x[:3] + x[3:6] * dt + 0.5 * x[6:] * dt**2  # Position update
    new_x[3:6] = x[3:6] + x[6:] * dt  # Velocity update
    new_x[6:] = x[6:]  # Keep acceleration constant
    return new_x

def F_jacobian():
    F = np.eye(state_dim)
    F[:3, 3:6] = np.eye(3) * dt  # Derivative of position wrt velocity
    F[:3, 6:] = 0.5 * np.eye(3) * dt**2  # Derivative of position wrt acceleration
    F[3:6, 6:] = np.eye(3) * dt  # Derivative of velocity wrt acceleration
    return F

def H_jacobian():
    H = np.zeros((3, state_dim))
    H[:3, :3] = np.eye(3)
    return H

def calculate_3d_distance(pred, truth):
    return np.linalg.norm(pred - truth)

# Process noise covariance
Q = np.eye(state_dim) * 0.01

# Measurement noise covariance
R = np.eye(3) * 0.1

# Initialize error list
all_distances = []

# Iterate over the dataset in chunks of size `2*n`
for start_idx in range(start, len(positions) - 2 * n, n):
    end_idx = start_idx + n
    future_idx = end_idx + n

    # Observed and ground-truth data
    observed_data = positions[start_idx:end_idx]
    observed_velocity = velocities[start_idx:end_idx]
    observed_acceleration = accelerations[start_idx:end_idx]
    ground_truth = positions[end_idx:future_idx]

    # Initial state estimate
    x = np.zeros(state_dim)
    x[:3] = observed_data[0]  # Initial position
    x[3:6] = observed_velocity[0]  # Initial velocity
    x[6:] = observed_acceleration[0]  # Initial acceleration

    # Initial covariance estimate
    P = np.eye(state_dim)

    # Update Kalman filter with observed data
    for i in range(1, len(observed_data)):
        # Prediction step
        F = F_jacobian()
        x = f(x)  # Predict the next state
        P = F @ P @ F.T + Q  # Update the covariance

        # Measurement update
        z = observed_data[i]
        H = H_jacobian()
        y = z - x[:3]  # Residual
        S = H @ P @ H.T + R  # Residual covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x + K @ y  # Update the state estimate
        P = (np.eye(state_dim) - K @ H) @ P  # Update the covariance

    # Predict next `n` rows
    predictions = []
    for _ in range(n):
        F = F_jacobian()
        x = f(x)
        P = F @ P @ F.T + Q
        predictions.append(x[:3].copy())

    predictions = np.array(predictions)

    # Calculate distances for this chunk
    distances = [calculate_3d_distance(pred, true) for pred, true in zip(predictions, ground_truth)]
    all_distances.extend(distances)

# Calculate overall average error
overall_avg_error = np.mean(all_distances)
print(f"Overall Average 3D Error: {overall_avg_error:.4f}")
