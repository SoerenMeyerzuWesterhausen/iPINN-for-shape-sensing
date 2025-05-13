import os
import numpy as np
from Preprocess import prepare_all_data, preprocess, input_preparation
from Train_iPINN import train_model

"""
This is the main function for training the iPINN. It performs a preprocessing of data if necessary and continues
with the training then. After training an iPINN, the script iPINN_Calculation allows to perform calculations for 
shape sensing tasks.
"""

# Define relevant paths
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'Training_Data')
results_dir = os.path.join(data_dir, '01_Training_Inputs_Results')
inputs_dir = os.path.join(data_dir, '00_Training_Inputs')

# Define the coarse mesh file, where the strains will be assigned to for processing
coarse_mesh_file = os.path.join(cwd, 'Coarse_Mesh', '2024-09-27_Alu-Tube-SCALE_CoarseMesh_15mm_Disp_Load_Eigenmodes_V2.inp')

# Define the sensor file and get the number of sensors included there
sensor_file = os.path.join(cwd, 'Sensor_Positions_Alu_Tube_SCALE_50_Sensors.txt')
with open(sensor_file, 'r') as f:
    number_of_sensors = sum(1 for _ in f)

# Coordinates for boundary condition area and load application area along the x-axis
bc_coord, load_coord = 120.0, 1880.0

# Scalers for the inputs to be in range (0, 1) for the iPINN
strain_scaler = [1.1e-3, 3.2e-4, 7.1e-4]
geom_scaler = [2000, 120, 60]

# Check if preprocessing is needed
res_files = os.listdir(results_dir)
needs_preprocessing = (
    not any(f.startswith("U_") for f in res_files)
    or not any(f.startswith("E_") for f in res_files)
    or not any(f.startswith("Elements_centroids_displacements_strains_top_") for f in res_files)
)

if needs_preprocessing:
    print("Start to prepare inputs from Abaqus files...")
    prepare_all_data(inputs_dir, results_dir, coarse_mesh_file)

# Filter relevant training files
files = [f for f in os.listdir(results_dir) if f.startswith("Elements_centroids_displacements_strains_top_")]

# Predefine variables before the loop
X_coords, y_disps, X_sensors, strain_fields = [], [], [], []
bc_ids_all, load_ids_all, sensor_ids_all = [], [], []
sensors_grouped = None
index_offset = 0

print("Start to process files for iPINN-Input ...")
for f in files:
    full_path = os.path.join(results_dir, f)
    fe_data, bc_ids, load_ids, y, y_disp, sensor_ids, map_indices = preprocess(full_path, sensor_file, bc_coord, load_coord)
    coords, strains, _, full_strain, sensor_data = input_preparation(fe_data, sensor_ids, geom_scaler, strain_scaler)

    if sensors_grouped is None:
        sensors_grouped = [[] for _ in range(sensor_data.shape[0])]
    for i, sensor in enumerate(sensor_data):
        sensors_grouped[i].append(sensor)

    X_coords.append(np.hstack((coords, strains)))
    y_disps.append(y_disp)
    X_sensors.append(sensor_data)
    strain_fields.append(full_strain)

    bc_ids_all.extend([i + index_offset for i in bc_ids])
    load_ids_all.extend([i + index_offset for i in load_ids])
    sensor_ids_all.extend([i + index_offset for i in sensor_ids.values])

    index_offset += coords.shape[0]

# Prepare input and output arrays
X_train = np.concatenate(X_coords, axis=0)
X_sensor_train = np.concatenate(X_sensors, axis=0)
y_train = np.concatenate(y_disps, axis=0)

X_all_sensors = {str(i+1): np.stack(s, axis=0) for i, s in enumerate(sensors_grouped)}
X_all_coords = {axis: X_train[:, i::3][:, 0] for i, axis in enumerate(['x', 'y', 'z'])}
y_all_disps = {f'u{i+1}': y_train[:, i] for i in range(3)}
strain_dict = {f'E{ij}': np.stack([f[:, k] for f in strain_fields], axis=0)
               for k, ij in enumerate(['11', '22', '12'])}

# Train the model
train_model(
    bc_ids_all, load_ids_all, X_all_sensors, X_all_coords, y_all_disps,
    geom_scaler, strain_scaler, results_dir, strain_dict, cwd, map_indices, sensor_ids, number_of_sensors
)
