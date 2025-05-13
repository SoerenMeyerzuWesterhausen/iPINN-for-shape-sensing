
# Import the needed libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from Postprocess import plot_strains
from Postprocess import plot_displacements
from Postprocess import erms_calculation

#######################################################################################################################
# Definition of required inputs

# Get the working directory to define relevant paths
directory = os.getcwd()
inp_filepath = directory + '\\Load_Case'
x_clamped = 120.0
x_load_app_coord = 1880.0
e_scaler = [1.1e-3, 3.2e-4, 7.1e-4]
x_scaler = [2000, 120, 60]

# Load the trained iPINN
train_results_path = directory + '\\Training_Data\\01_Training_Inputs_Results'
# Define correct model name here, depending on which to choose
model = tf.keras.models.load_model(train_results_path + '\\Trained_iPINN_Bending_50Sensors.keras', compile=False)

# Define the sensor position file
sens_file = directory + '\\Sensor_Positions_Alu_Tube_SCALE_50_Sensors.txt'


#######################################################################################################################
# Definition of required functions


# Load the sensor locations
def read_sensor_locations(fe_data, sens_file):
    # Read the sensor locations
    sensor_location = pd.read_csv(sens_file, delimiter=',', header=None)
    sensor_location.columns = ['ID', 'X', 'Y', 'Z']
    sensor_location = sensor_location.astype({"ID": int, "X": float, "Y": float, "Z": float})
    # Find the closest elements
    elem_ids = []
    for i in range(len(sensor_location['ID'])):
        x_sens_i = float(sensor_location['X'].iloc[i])
        y_sens_i = float(sensor_location['Y'].iloc[i])
        z_sens_i = float(sensor_location['Z'].iloc[i])
        possib_elems = fe_data[(fe_data['X'] >= (x_sens_i - 10)) & (fe_data['X'] <= (x_sens_i + 10)) &
                               (fe_data['Y'] >= (y_sens_i - 10)) & (fe_data['Y'] <= (y_sens_i + 10)) &
                               (fe_data['Z'] >= (z_sens_i - 10)) & (fe_data['Z'] <= (z_sens_i + 10))]
        closest_value_x = min(possib_elems['X'], key=lambda x: abs(x - x_sens_i))
        closest_value_y = min(possib_elems['Y'], key=lambda x: abs(x - y_sens_i))
        closest_value_z = min(possib_elems['Z'], key=lambda x: abs(x - z_sens_i))
        closest_fine = possib_elems[possib_elems['X'] == closest_value_x]
        if len(closest_fine['ID']) > 1:
            closest_fine = closest_fine[closest_fine['Y'] == closest_value_y]

        if len(closest_fine['ID']) > 1:
            closest_fine = closest_fine[closest_fine['Z'] == closest_value_z]

        # Find the ID of the closest element from the fine mesh
        elem_ids.append(int(closest_fine['ID']))

    # Make a DataFrame from the element-ID-List
    sensor_elem_ids = pd.DataFrame(elem_ids, columns=['ID'])
    sensor_elem_ids['ID'] = sensor_elem_ids['ID'] - 1
    # Return the IDs of the elements with sensors
    return sensor_elem_ids


# Get the boundary condition indices
def find_boundary_condition_indices(df_data, clamped_coord):
    # Filter for indices in the clamped part
    df_bcs = df_data[(df_data['X'] <= clamped_coord)]
    bc_list = df_bcs.index.values
    return bc_list


# Get the boundary condition indices
def find_load_application_indices(df_data, load_coord):
    # Filter all indices below x_load_app_coord and find the max. value as the last
    # points before application
    filtered_indices = fe_data[df_data['X'] < load_coord].index
    max_near_value = fe_data.loc[filtered_indices, 'X'].max()
    closest_indices = fe_data[(fe_data['X'] == max_near_value) & (fe_data.index.isin(filtered_indices))].index.tolist()

    # Create a dictionary to store all the indices in the load application part with
    # the same coordinates as in closest_indices
    result_dict = {}

    for idx in closest_indices:
        # Values for y and z at the current index
        y_value = df_data.loc[idx, 'Y']
        z_value = df_data.loc[idx, 'Z']

        # Find indices with the same Y- and Z-values and a different X-value which is above x_load_app_coord
        matching_indices = df_data[(df_data['Y'] == y_value) &
                                   (df_data['Z'] == z_value) &
                                   (df_data['X'] > load_coord)].index.tolist()

        # Save found indices iin the Dictionary with the current index as key
        result_dict[idx] = matching_indices

    return result_dict


# Adapt displacements for boundary conditions
def displacement_adaption(df_adapt, bc_ids, load_app_ids):
    # Set displacements at the clamped part to Zero
    df_adapt.loc[bc_ids, ['U1', 'U2', 'U3']] = 0.0
    # Set the displacements at the load application part to the last predicted ones before this part
    for key_index, indices_list in load_app_ids.items():
        # Hole die Werte von 'U1', 'U2', 'U3' an der Position des Key-Index
        displacement_values = df_adapt.loc[key_index, ['U1', 'U2', 'U3']]

        # Weisen Sie diese Werte allen Indices in indices_list zu
        df_adapt.loc[indices_list, ['U1', 'U2', 'U3']] = displacement_values.values

    return df_adapt


#######################################################################################################################
# Make calculations/ predictions with the trained iPINN

# Get the FE data
os.chdir(inp_filepath)
fe_data = pd.read_csv('Elements_centroids_displacements_strains_top_2025-04-22_Alu-Tube-SCALE-'
                      'FineMesh_Eigenmodes300Hz_Fz800N_T0Nm.txt', delimiter=',', header=None)
fe_data.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4', 'X', 'Y', 'Z', 'U1', 'U2', 'U3', 'E11', 'E22', 'E12']
fe_data = fe_data.astype({"ID": int, "Node1": int, "Node2": int, "Node3": int, "Node4": int,
                          "X": float, "Y": float, "Z": float, "U1": float, "U2": float, "U3": float,
                          "E11": float, "E22": float, "E12": float})
fe_data['original_index'] = fe_data.index
fe_data = fe_data.sort_values(by=['X', 'Y', 'Z'])
fe_data.reset_index(drop=True, inplace=True)
# Get the sensor element IDs and coordinates and strains there
sensor_elem_ids = read_sensor_locations(fe_data, sens_file)


# Functions to prepare the inputs for the iPINN by scaling them 0 to 1
def scale_column_strain(column, scaler):
    return (column + scaler) / (2 * scaler)


def scale_column_coords(column, scaler):
    return column / scaler


# Scale the coordinates and strains for the inputs
inp_strains = fe_data[['E11', 'E22', 'E12']]
inp_coords = fe_data[['X', 'Y', 'Z']]
inp_coords_scaled = inp_coords.copy()
inp_strains_scaled = inp_strains.copy()

inp_coords_scaled['X'] = scale_column_coords(inp_coords['X'], x_scaler[0])
inp_coords_scaled['Y'] = scale_column_coords(inp_coords['Y'], x_scaler[1])
inp_coords_scaled['Z'] = scale_column_coords(inp_coords['Z'], x_scaler[2])
inp_strains_scaled['E11'] = scale_column_strain(inp_strains['E11'], e_scaler[0])
inp_strains_scaled['E22'] = scale_column_strain(inp_strains['E22'], e_scaler[1])
inp_strains_scaled['E12'] = scale_column_strain(inp_strains['E12'], e_scaler[2])

# Combine scaled strains and coordinates as inputs
inp_sensor = pd.concat([inp_coords_scaled, inp_strains_scaled], axis=1)
inp_sensor = inp_sensor.iloc[sensor_elem_ids['ID']]
inp_sensor.reset_index(drop=True, inplace=True)

# Get the indices for applied boundary conditions and loads
boundary_conditions_ids = find_boundary_condition_indices(fe_data, x_clamped)
load_application_indices = find_load_application_indices(fe_data, x_load_app_coord)

# Prepare the inputs for the iPINN input layer
inp_coords = inp_coords_scaled.values.astype(np.float32).T
inp_coords_tensors = [tf.convert_to_tensor([inp_coords[i]], dtype=tf.float32) for i in range(inp_coords.shape[0])]

inp_sensor = inp_sensor.values.astype(np.float32)  # Arrays of the scaled coordinates and strains for each sensor neuron
inp_sensor_tensors = [tf.convert_to_tensor([inp_sensor[i, :]], dtype=tf.float32) for i in range(inp_sensor.shape[0])]

# Combine all inputs (coordinates + sensors) for the model
model_inputs = inp_coords_tensors + inp_sensor_tensors

# Calculate the displacements
pred_displacements = model(model_inputs, training=False)


pred_displacements = np.column_stack([pred_displacements[0][0], pred_displacements[1][0], pred_displacements[2][0]])

df_predictions = pd.DataFrame(pred_displacements, columns=['U1', 'U2', 'U3'])
df_predictions = df_predictions

# Reset indices after mapping
df_predictions['original_index'] = fe_data['original_index']
df_predictions = df_predictions.sort_values(by='original_index').reset_index(drop=True)
df_predictions.drop(columns='original_index', inplace=True)
fe_data = fe_data.sort_values(by='original_index').reset_index(drop=True)
fe_data.drop(columns='original_index', inplace=True)

# Calculate the model %ERMS
erms_x = erms_calculation(fe_data['U1'].values, df_predictions['U1'].values)
erms_y = erms_calculation(fe_data['U2'].values, df_predictions['U2'].values)
erms_z = erms_calculation(fe_data['U3'].values, df_predictions['U3'].values)
print(f"%ERMS_x: {erms_x} %; %ERMS_y: {erms_y} %; %ERMS_z: {erms_z} %")

# Plot the calculated against the reference solution
plot_displacements(train_results_path, fe_data, df_predictions, filename="Predictions_800N_Bending_Training.svg")
