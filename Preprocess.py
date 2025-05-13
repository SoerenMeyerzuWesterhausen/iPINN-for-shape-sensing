
import os
import glob
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path


# Function to call the data preparation for all the training data files
def prepare_all_data(inp_folder, result_folder, coarse_direct):

    # Search for the *.inp-file from the coarse mesh in the current directory
    with open(coarse_direct, 'r') as file_coarse:
        lines_coarse = file_coarse.readlines()

    os.chdir(inp_folder)

    # Export the data from the *.odb-files
    # Call the Abaqus function for the strain export in cmd

    files = os.listdir(result_folder)
    filtered_files = [f for f in files if f.startswith("U_") or f.startswith("E_")]
    u_files = [f for f in files if f.startswith("U_")]
    e_files = [f for f in files if f.startswith("E_")]

    if filtered_files == [] or len(u_files) != len(e_files):
        os.system('cmd /C "abaqus cae noGUI=01-1_Strain_Displacement_Export.py"')
        os.system('exit cmd')
        os.unlink('abaqus.rpy')

        # Move the files to the results directory
        result_files = glob.glob('*.txt')
        for i in range(0, len(result_files)):
            file = result_files[i]
            os.replace(inp_folder + '\\' + file, result_folder + '\\' + file)

    files = os.listdir(inp_folder)

    # Filter for .inp and .odb files
    inp_files = set(f for f in files if f.endswith('.inp'))
    odb_files = set(f for f in files if f.endswith('.odb'))

    # Find pairs of *.inp and *.odb-files
    for inp_file in inp_files:
        # Datei ohne Endung
        file_name_without_extension = inp_file[:-4]

        # Prüfen, ob es eine entsprechende .odb Datei gibt
        if f"{file_name_without_extension}.odb" in odb_files:
            print(file_name_without_extension)
            # Ruf die data_preparation Funktion mit dem Dateinamen ohne Endung auf
            data_preparation(lines_coarse, file_name_without_extension, result_folder, inp_folder)


# Function for the initial data preparation
def data_preparation(coarse_data, inp_filename, result_folder, inp_folder):
    os.chdir(inp_folder)
    # Data from the coarse mesh
    lines_coarse = coarse_data
    # Open the *.inp files and read the data
    with open(inp_filename+'.inp', 'r') as file:
        lines_fine = file.readlines()

    ###################################################################################################################
    # Start the preprocessing by extracting the relevant data from Abaqus, if not already done

    # Search for the displacement and strain data in the directory and if they don't exist, run the script
    disp_file_name = 'U_'+inp_filename+'.txt'
    strain_file_name = 'E_'+inp_filename+'.txt'

    # Get the information on the meshes

    # Function to read the node information from an *.inp-file
    def node_input_generation(lines):
        node_coordinates = []
        # Extract node coordinates from the lines of the *.inp file
        for line in lines:
            # Search for lines with node coordinates
            if '*Node' in line:
                next_line = lines[lines.index(line) + 1]
                while ',' in next_line and not next_line.startswith('*'):
                    # Extract values for the nodes
                    if len(next_line) < 50:
                        break
                    else:
                        node_id, x, y, z = next_line.strip().split(',')[0:4]
                        node_coordinates.append([node_id, x, y, z])
                        next_line = lines[lines.index(next_line) + 1]

        # Write node data into a data frame
        df_node_coordinates = pd.DataFrame(node_coordinates)
        df_node_coordinates.columns = ['Node', 'X', 'Y', 'Z']
        df_node_coordinates = df_node_coordinates.astype({"Node": int, "X": float, "Y": float, "Z": float})
        df_node_coordinates.drop_duplicates(inplace=True, ignore_index=True)

        return df_node_coordinates

    # Function to read the element information from an *.inp-file
    def element_input_generation(lines):
        element_definition = []
        found_elements = False

        for line in lines:
            # Find starting point in *.inp file for the element connectivity
            if '*Element, type=S4R' in line or '*Element, type=S4' in line:
                found_elements = True
                continue
            # Find the correct end of the element connectivity in the *.inp file
            elif found_elements and line.startswith('*'):
                found_elements = False
            # Read the elements and nodes from the file
            elif found_elements:
                data = line.strip().split(',')
                data = [d.strip() for d in data]  # Delete Whitespace
                element_id = data[0]
                node_ids = data[1:]
                element_definition.append([element_id] + node_ids)

        # Generate dataframe for further processing
        df_element_definition = pd.DataFrame(element_definition)

        if not df_element_definition.empty:
            df_element_definition.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4']
            df_element_definition = df_element_definition.astype(int)
        else:
            print("⚠️ No elements found!.")

        return df_element_definition

    # Function to create a mesh for post-processing from the data
    def create_and_save_mesh(df_elem, df_nodes, save_path):
        # Get the points in the desired form for PyVista
        points = df_nodes[['X', 'Y', 'Z']]
        points = points.values
        points = np.array(points, dtype=np.float32)
        
        # Create the cells with the node information
        num_col = pd.DataFrame([4] * len(df_elem['ID']), columns=['Num'])
        df = pd.DataFrame([num_col['Num'].values, df_elem['Node1'].values - 1, df_elem['Node2'].values - 1,
                           df_elem['Node3'].values - 1, df_elem['Node4'].values - 1])
        cells = np.array(df.values, dtype=int)
        cells = cells.T
        
        # Define the cell-type --> Here we use quadrilateral shell elements
        cell_type = np.array([pv.CellType.QUAD] * len(df_elem['ID']), np.int8)
        # Create a grid and save it for further processing
        grid = pv.UnstructuredGrid(cells, cell_type, points)
        os.chdir(save_path)
        grid.save('mesh_alu_tube.vtk')

    # Call the functions to get the FE-Mesh data
    df_fine_nodes = node_input_generation(lines_fine)
    df_coarse_nodes = node_input_generation(lines_coarse)
    df_fine_elements = element_input_generation(lines_fine)
    df_coarse_elements = element_input_generation(lines_coarse)

    # Call the function to create and save the coarse mesh
    create_and_save_mesh(df_coarse_elements, df_coarse_nodes, result_folder)

    ###################################################################################################################
    # Prepare the data with the strains and displacements

    # Get and load the displacement data file and write its data into a data frame
    os.chdir(result_folder)
    df_disp_data = pd.read_csv(disp_file_name, delimiter=',', header=None)
    df_disp_data.columns = ['Node', 'U1', 'U2', 'U3']
    df_disp_data = df_disp_data.iloc[2:]  # Delete first two rows with the data of the reference points from the FE model

    # Get and load the strain file and write its data into a data frame
    df_strains = pd.read_csv(strain_file_name, delimiter=',', header=None)
    df_strains.columns = ['ID', 'SEC_PT', 'E11', 'E22', 'E33', 'E12']
    df_strains = df_strains.astype({"ID": int, "SEC_PT": int, "E11": float, "E22": float, "E33": float, "E12": float})
    # Get the data for the top and bottom surface of the part
    unique_SEC = df_strains['SEC_PT'].values.flatten()
    unique_SEC = np.unique(unique_SEC)
    indices_sec_top = df_strains['SEC_PT'] == unique_SEC[0]
    indices_sec_bottom = df_strains['SEC_PT'] == unique_SEC[1]
    # Make two dataframes for the two surfaces
    df_strains_top = df_strains[indices_sec_top]
    df_strains_bottom = df_strains[indices_sec_bottom]

    # Function to calculate the centroids of an elements
    def centroid_calculation(df_nodes, df_element_definition):
        # Initialise the variables for the calculation
        elem = []
        x = []
        y = []
        z = []
        node1 = []
        node2 = []
        node3 = []
        node4 = []

        # Loop over every element and calculate the centroids
        for el in range(0, len(df_element_definition)):
            # Get the corresponding nodes coordinates for each element
            element = df_element_definition.iloc[el]
            n1 = int(element[1])
            n2 = int(element[2])
            n3 = int(element[3])
            n4 = int(element[4])
            # Get the node coordinates
            n1_coord = df_nodes.iloc[n1 - 1]
            n2_coord = df_nodes.iloc[n2 - 1]
            n3_coord = df_nodes.iloc[n3 - 1]
            n4_coord = df_nodes.iloc[n4 - 1]
            # Calculate element midpoints and write the information needed into the list
            x.append((n1_coord[1] + n2_coord[1] + n3_coord[1] + n4_coord[1]) / 4)
            y.append((n1_coord[2] + n2_coord[2] + n3_coord[2] + n4_coord[2]) / 4)
            z.append((n1_coord[3] + n2_coord[3] + n3_coord[3] + n4_coord[3]) / 4)
            elem.append(element[0])
            node1.append(n1)
            node2.append(n2)
            node3.append(n3)
            node4.append(n4)

        # Combine the data into a list
        centroid_list = np.column_stack((elem, node1, node2, node3, node4, x, y, z))
        # Make a data frame from the list and save it
        df_element_centroids = pd.DataFrame(centroid_list)
        df_element_centroids.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4', 'X', 'Y', 'Z']

        return df_element_centroids

    # Function to calculate the centroids of elements and the average displacements
    def centroid_and_average_displacement_calculation(df_nodes, df_element, df_disp, df_strain):
        # Initialise the variables for the calculation
        elem = []
        x = []
        y = []
        z = []
        node1 = []
        node2 = []
        node3 = []
        node4 = []
        u1_average = []
        u2_average = []
        u3_average = []
        e11 = []
        e22 = []
        e12 = []

        # Loop over every element and calculate the centroids
        for el in range(0, len(df_element)):
            # Get the corresponding nodes coordinates for each element
            element = df_element.iloc[el]
            n1 = int(element[1])
            n2 = int(element[2])
            n3 = int(element[3])
            n4 = int(element[4])
            # Get the node coordinates
            n1_coord = df_nodes.iloc[n1 - 1]
            n2_coord = df_nodes.iloc[n2 - 1]
            n3_coord = df_nodes.iloc[n3 - 1]
            n4_coord = df_nodes.iloc[n4 - 1]
            # Calculate element midpoints and write the information needed into the list
            x.append((n1_coord[1] + n2_coord[1] + n3_coord[1] + n4_coord[1]) / 4)
            y.append((n1_coord[2] + n2_coord[2] + n3_coord[2] + n4_coord[2]) / 4)
            z.append((n1_coord[3] + n2_coord[3] + n3_coord[3] + n4_coord[3]) / 4)
            # Get the data of the U1 displacement of all four nodes and calculate the average value
            u_n1 = df_disp.iloc[n1 - 1]
            u_n2 = df_disp.iloc[n2 - 1]
            u_n3 = df_disp.iloc[n3 - 1]
            u_n4 = df_disp.iloc[n4 - 1]
            u1_average.append((u_n1[1] + u_n2[1] + u_n3[1] + u_n4[1]) / 4)
            u2_average.append((u_n1[2] + u_n2[2] + u_n3[2] + u_n4[2]) / 4)
            u3_average.append((u_n1[3] + u_n2[3] + u_n3[3] + u_n4[3]) / 4)
            # Append the furthermore needed data
            elem.append(element[0])
            node1.append(n1)
            node2.append(n2)
            node3.append(n3)
            node4.append(n4)
            e11.append(df_strain['E11'].iloc[el])
            e22.append(df_strain['E22'].iloc[el])
            e12.append(df_strain['E12'].iloc[el])

        # Combine the data into a list
        centroid_list = np.column_stack((elem, node1, node2, node3, node4, x, y, z,
                                         u1_average, u2_average, u3_average, e11, e22, e12))
        # Make a data frame from the list and save it
        df_element_centroids = pd.DataFrame(centroid_list)
        df_element_centroids.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4', 'X', 'Y', 'Z',
                                        'U1', 'U2', 'U3', 'E11', 'E22', 'E12']
        
        del elem, node1, node2, node3, node4, u1_average, u2_average, u3_average, e11, e22, e12

        return df_element_centroids

    # Call the function to get the data 
    df_elem_centroid_coarse = centroid_calculation(df_coarse_nodes, df_coarse_elements)
    df_elem_centroids_disp_strain_top_fine = centroid_and_average_displacement_calculation(
        df_fine_nodes, df_fine_elements, df_disp_data, df_strains_top)
    df_elem_centroids_disp_strain_bottom_fine = centroid_and_average_displacement_calculation(
        df_fine_nodes, df_fine_elements, df_disp_data, df_strains_bottom)

    ###################################################################################################################

    # Assign the data from the fine to the coarse mesh
    def assign_strains_displacements_to_coarse(df_coarse, df_fine):
        
        # Initialise variables for assignment
        elem_id = []
        n1_id = []
        n2_id = []
        n3_id = []
        n4_id = []
        x = []
        y = []
        z = []
        u1 = []
        u2 = []
        u3 = []
        e11 = []
        e22 = []
        e12 = []
        
        # Loop over each element of the coarse mesh, find the closest from the fine mesh and assign the strains
        for i in range(len(df_coarse['ID'])):
            element = int(df_coarse['ID'].iloc[i])
            x_i = float(df_coarse['X'].iloc[i])
            y_i = float(df_coarse['Y'].iloc[i])
            z_i = float(df_coarse['Z'].iloc[i])

            # Filter the centroids from the finer mesh by the x-coordinate of the inverse element
            possib_fine_elems = df_fine[(df_fine['X'] >= (x_i-10)) & (df_fine['X'] <= (x_i+10)) &
                                        (df_fine['Y'] >= (y_i-10)) & (df_fine['Y'] <= (y_i+10)) &
                                        (df_fine['Z'] >= (z_i-10)) & (df_fine['Z'] <= (z_i+10))]
            closest_value_x = min(possib_fine_elems['X'], key=lambda x: abs(x - x_i))
            closest_value_y = min(possib_fine_elems['Y'], key=lambda x: abs(x - y_i))
            closest_value_z = min(possib_fine_elems['Z'], key=lambda x: abs(x - z_i))
            closest_fine = possib_fine_elems[possib_fine_elems['X'] == closest_value_x]
            closest_fine = closest_fine[closest_fine['Y'] == closest_value_y]
            closest_fine = closest_fine[closest_fine['Z'] == closest_value_z]

            # Find the strains of the closest element from the fine mesh
            elem_closest = int(closest_fine['ID'])
            closest = df_fine[df_fine['ID'] == elem_closest]
            u1.append(float(closest['U1']))
            u2.append(float(closest['U2']))
            u3.append(float(closest['U3']))
            e11.append(float(closest['E11']))
            e22.append(float(closest['E22']))
            e12.append(float(closest['E12']))
            
            # Append all the other data needed
            elem_id.append(element)
            n1_id.append(int(df_coarse['Node1'].iloc[i]))
            n2_id.append(int(df_coarse['Node2'].iloc[i]))
            n3_id.append(int(df_coarse['Node3'].iloc[i]))
            n4_id.append(int(df_coarse['Node4'].iloc[i]))
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)

        # Combine the data and store it
        df_coarse_assigned = np.column_stack((elem_id, n1_id, n2_id, n3_id, n4_id, x, y, z, u1, u2, u3, e11, e22, e12))
        df_coarse_assigned = pd.DataFrame(df_coarse_assigned)
        df_coarse_assigned.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4', 'X', 'Y', 'Z', 'U1', 'U2', 'U3', 'E11', 'E22', 'E12']
        
        del elem_id, n1_id, n2_id, n3_id, n4_id, x, y, z, u1, u2, u3, e11, e22, e12
        
        return df_coarse_assigned

    df_coarse_data_top = assign_strains_displacements_to_coarse(df_elem_centroid_coarse,
                                                                df_elem_centroids_disp_strain_top_fine)
    df_coarse_data_bottom = assign_strains_displacements_to_coarse(df_elem_centroid_coarse,
                                                                   df_elem_centroids_disp_strain_bottom_fine)
    
    # Save the created files
    os.chdir(result_folder)
    np.savetxt('Elements_centroids_displacements_strains_top_'+inp_filename+'.txt', df_coarse_data_top, delimiter=',')
    np.savetxt('Elements_centroids_displacements_strains_bottom_'+inp_filename+'.txt', df_coarse_data_bottom, delimiter=',')


# Function to prepare the data for calculation
def preprocess(fe_file, sens_filepath, x_clamped, x_load_app):
    
    # Load the simulation data
    def read_fe_data(fe_file):
        map_indices = pd.DataFrame()
        df_fe_sim_data = pd.read_csv(fe_file, delimiter=',', header=None)
        df_fe_sim_data.columns = ['ID', 'Node1', 'Node2', 'Node3', 'Node4', 'X', 'Y', 'Z',
                                  'U1', 'U2', 'U3', 'E11', 'E22', 'E12']
        df_fe_sim_data = df_fe_sim_data.astype({"ID": int, "Node1": int, "Node2": int, "Node3": int,
                                                "Node4": int, "X": float, "Y": float, "Z": float,
                                                "U1": float, "U2": float, "U3": float,
                                                "E11": float, "E22": float, "E12": float})
        map_indices['original_index'] = df_fe_sim_data.index
        df_fe_sim_data = df_fe_sim_data.sort_values(by=['X', 'Y', 'Z'])
        df_fe_sim_data.reset_index(drop=True, inplace=True)
        return df_fe_sim_data, map_indices
        
    # Load the sensor locations
    def read_sensor_locations(fe_data, sens_filepath):
        # Read the sensor locations
        sensor_location = pd.read_csv(sens_filepath, delimiter=',', header=None)
        sensor_location.columns = ['ID', 'X', 'Y', 'Z']
        print(sensor_location)
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

            # Find the strains of the closest element from the fine mesh
            elem_ids.append(int(closest_fine['ID']))
        # Make a DataFrame from the element-ID-List
        sensor_elem_ids = pd.DataFrame(elem_ids, columns=['ID'])
        # Return the IDs of the elements with sensors
        return sensor_elem_ids
    
    # Get the boundary condition indices
    def find_boundary_condition_indices(df_data, clamped_coord):
        # Filter for indices in the clamped part
        df_bcs = df_data[(df_data['X'] <= clamped_coord)]
        bc_list = df_bcs.index.values
        return bc_list

    # Get the boundary condition indices
    def find_load_application_indices(df_data, clamped_coord):
        # Filter for indices in the clamped part
        df_bcs = df_data[(df_data['X'] >= clamped_coord)]
        bc_list = df_bcs.index.values
        return bc_list

    # Adapt displacements for boundary conditions
    def displacement_adaption(df_fe, bc_ids):
        df_adapt = df_fe
        df_adapt.loc[bc_ids, ['U1', 'U2', 'U3']] = 0.0
        df_adapt.loc[bc_ids, ['E11', 'E22', 'E12']] = 0.0
        return df_adapt
    
    # Call the functions to get the data
    fe_data, map_indices = read_fe_data(fe_file)
    boundary_conditions_ids = find_boundary_condition_indices(fe_data, x_clamped)
    load_application_indices = find_load_application_indices(fe_data, x_load_app)
    fe_data_adapt = displacement_adaption(fe_data, boundary_conditions_ids)
    sensor_ids = read_sensor_locations(fe_data, sens_filepath)
    sensor_ids = pd.DataFrame(sensor_ids, columns=['ID'])
    
    # Prepare the output data for the neural network
    in_train_data = fe_data_adapt[['X', 'Y', 'Z', 'E11', 'E22', 'E12']]
    # Set the strain values with missing sensor positions to Zero
    out_train_data = fe_data_adapt[['U1', 'U2', 'U3']]
    out_train_data = out_train_data.values.astype(np.float32)
    
    return (fe_data, boundary_conditions_ids, load_application_indices, in_train_data, out_train_data,
            sensor_ids, map_indices)


# Prepare the input for training/ prediction
def input_preparation(inp_data, sensor_ids, x_scaler, e_scaler):
    # Prepare the sensor IDs for indexing
    sensor_ids['ID'] = sensor_ids['ID'] - 1

    # Functions to prepare the inputs for the iPINN by scaling them 0 to 1
    def scale_column_strain(column, scaler):
        return (column + scaler) / (2 * scaler)

    def scale_column_coords(column, scaler):
        return column / scaler

    # Prepare the coordinates by deleting the one at not-sensorised positions and scale them
    inp_coords = inp_data[['X', 'Y', 'Z']]
    out_coords = inp_coords.copy()  # Output coordinates in case it is needed as unscaled collocation points
    out_coords = out_coords.values.astype(np.float32)
    # Scaling the coordinates
    inp_coords_scaled = inp_coords.copy()
    inp_coords_scaled['X'] = scale_column_coords(inp_coords['X'], x_scaler[0])
    inp_coords_scaled['Y'] = scale_column_coords(inp_coords['Y'], x_scaler[1])
    inp_coords_scaled['Z'] = scale_column_coords(inp_coords['Z'], x_scaler[2])

    # Prepare the strains by deleting the one at not-sensorised positions and scale them
    inp_strains = inp_data[['E11', 'E22', 'E12']]
    inp_all_strains = inp_strains.values.astype(np.float32)

    # Scale the strains to the ranges
    inp_strains_scaled = inp_strains.copy()
    inp_strains_scaled['E11'] = scale_column_strain(inp_strains['E11'], e_scaler[0])
    inp_strains_scaled['E22'] = scale_column_strain(inp_strains['E22'], e_scaler[1])
    inp_strains_scaled['E12'] = scale_column_strain(inp_strains['E12'], e_scaler[2])

    # Combine scaled strains and coordinates as inputs
    inp_sensor = pd.concat([inp_coords_scaled, inp_strains_scaled], axis=1)
    inp_sensor = inp_sensor.iloc[sensor_ids['ID']]
    inp_sensor.reset_index(drop=True, inplace=True)
    inp_sensor = inp_sensor.values.astype(np.float32)

    # Set the strains to 0, where no value is measured
    inp_strains.loc[~inp_strains.index.isin(sensor_ids['ID']), ['E11', 'E22', 'E12']] = 0.0
    inp_strains = inp_strains.values.astype(np.float32)

    inp_coords = inp_coords_scaled.values.astype(np.float32)

    return inp_coords, inp_strains, out_coords, inp_all_strains, inp_sensor
