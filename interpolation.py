import pandas as pd
import numpy as np

# Load the dataset
file_path = 'data/mav0/state_groundtruth_estimate0/data.csv'  # Original file
data = pd.read_csv(file_path)

# Parameters
n = 150  # Number of rows to miss
columns_to_interpolate = [
    ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', 
    ' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]', 
    ' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]'
]

# Function to interpolate between two points in a straight line
def interpolate_rows(start_row, end_row, num_points):
    interpolated_rows = []
    
    for i in range(1, num_points + 1):
        fraction = i / (num_points + 1)
        interpolated_row = {
            '#timestamp': int(start_row['#timestamp'] + fraction * (end_row['#timestamp'] - start_row['#timestamp']))
        }
        
        for col in columns_to_interpolate:
            interpolated_row[col] = start_row[col] + fraction * (end_row[col] - start_row[col])
        
        interpolated_rows.append(interpolated_row)
    
    return interpolated_rows

# Process the dataset
processed_data = []
for i in range(0, len(data), n + 1):
    if i + n + 1 < len(data):
        # Add the starting row
        processed_data.append(data.iloc[i].to_dict())
        # Interpolate rows
        interpolated = interpolate_rows(data.iloc[i], data.iloc[i + n + 1], n)
        processed_data.extend(interpolated)
    else:
        # Add remaining rows at the end
        processed_data.extend(data.iloc[i:].to_dict('records'))

# Convert to DataFrame
processed_data_df = pd.DataFrame(processed_data)

# Save the processed dataset
processed_file_path = 'processed_data_mav0_interpolated.csv'
processed_data_df.to_csv(processed_file_path, index=False)

print(f"Processed data saved to {processed_file_path}")
