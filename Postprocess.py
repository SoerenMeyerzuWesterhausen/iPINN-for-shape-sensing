import os
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata


# Calculate the %ERMS
def erms_calculation(df_ref, df_disp):
    max_val = max(abs(df_ref))
    weighted_squared_diff = ((df_disp - df_ref) ** 2) / max_val
    erms = 100 * np.sqrt((sum(weighted_squared_diff) / len(df_disp)))
    return erms


# Function to plot the predicted strains against the FE reference strains
def plot_strains(mesh_path, df_ref, df_strain):
    # Go to the directory with the mesh
    os.chdir(mesh_path)
    # Get the magnitude values of the displacements
    magnitude = df_strain.max(axis=1).to_numpy()

    # Load the created mesh
    mesh = pv.read('mesh_alu_tube.vtk')
    # Create copies for the different plots
    mesh_e11_pred = mesh.copy()
    mesh_e22_pred = mesh.copy()
    mesh_e12_pred = mesh.copy()
    mesh_e11_ref = mesh.copy()
    mesh_e22_ref = mesh.copy()
    mesh_e12_ref = mesh.copy()
    '''
    # Calculate element centroids for mapping of strains to the nodes
    element_centers = np.array([mesh.cell_centers(i).center for i in range(mesh.n_cells)])

    # Interpolate element to node data
    def interpolate_to_nodes(df):
        return {
            col: griddata(element_centers, df[col].values, mesh.points, method='linear')
            for col in df.columns
        }

    node_strain = interpolate_to_nodes(df_strain)
    node_ref = interpolate_to_nodes(df_ref)

    plotter = pv.Plotter(shape=(2, 3))

    # Setze die Kameraposition
    plotter.camera_position = [
        (-690.6472292362095, -1640.18229140095, 1182.5880431379926),
        (855.9760919980398, 113.7949390370839, -58.44195586610785),
        (0, 0, 1)
    ]

    # Helferfunktion zum Hinzufügen von Meshes zu Plots
    def add_plot(row, col, data, label):
        plotter.subplot(row, col)
        mesh.point_data[label] = data  # Setze Knotendaten
        plotter.add_mesh(mesh, scalars=label, cmap='turbo')
        plotter.show_axes()

    # Vorhersage-Daten hinzufügen
    add_plot(0, 0, node_strain['E11'], 'Predicted Strain E11')
    add_plot(0, 1, node_strain['E22'], 'Predicted Strain E22')
    add_plot(0, 2, node_strain['E12'], 'Predicted Strain E12')

    # Referenz-Daten hinzufügen
    add_plot(1, 0, node_ref['E11'], 'Reference Strain E11')
    add_plot(1, 1, node_ref['E22'], 'Reference Strain E22')
    add_plot(1, 2, node_ref['E12'], 'Reference Strain E12')

    plotter.link_views()  # Verknüpfe die Ansichten
    plotter.show()
    '''
    # Calculate the deformed shape
    #points = mesh.points.copy()
    #points = points + df_disp.values
    #mesh.points = points
    # Get the displacement magnitudes into the mesh for the colorplot
    #mesh["Displacement [mm]"] = magnitude

    # Plot the calculations
    plotter = pv.Plotter(shape=(2, 3))
    # Set the camera position in a predefined way
    plotter.camera_position = [
        (-690.6472292362095, -1640.18229140095, 1182.5880431379926),
        (855.9760919980398, 113.7949390370839, -58.44195586610785),
        (0, 0, 1)
    ]
    # Plot of the predicted solutions
    plotter.subplot(0, 0)
    mesh_e11_pred["Predicted Strain E11"] = df_strain['E11'].values
    plotter.add_mesh(mesh_e11_pred, scalars='Predicted Strain E11',
                     cmap='turbo', clim=[mesh_e11_pred['Predicted Strain E11'].min(),
                                         mesh_e11_pred['Predicted Strain E11'].max()])
    plotter.show_axes()
    plotter.subplot(0, 1)
    mesh_e22_pred["Predicted Strain E22"] = df_strain['E22'].values
    plotter.add_mesh(mesh_e22_pred, scalars='Predicted Strain E22',
                     cmap='turbo', clim=[mesh_e22_pred['Predicted Strain E22'].min(),
                                         mesh_e22_pred['Predicted Strain E22'].max()])
    plotter.show_axes()
    plotter.subplot(0, 2)
    mesh_e12_pred["Predicted Strain E12"] = df_strain['E12'].values
    plotter.add_mesh(mesh_e12_pred, scalars='Predicted Strain E12',
                     cmap='turbo', clim=[mesh_e12_pred['Predicted Strain E12'].min(),
                                         mesh_e12_pred['Predicted Strain E12'].max()])
    plotter.show_axes()
    # Plot of the predicted solutions
    plotter.subplot(1, 0)
    mesh_e11_ref["Reference Strain E11"] = df_ref['E11'].values
    plotter.add_mesh(mesh_e11_ref, scalars='Reference Strain E11',
                     cmap='turbo', clim=[mesh_e11_ref['Reference Strain E11'].min(),
                                         mesh_e11_ref['Reference Strain E11'].max()])
    plotter.show_axes()
    plotter.subplot(1, 1)
    mesh_e22_ref["Reference Strain E22"] = df_ref['E22'].values
    plotter.add_mesh(mesh_e22_ref, scalars='Reference Strain E22',
                     cmap='turbo', clim=[mesh_e22_ref['Reference Strain E22'].min(),
                                         mesh_e22_ref['Reference Strain E22'].max()])
    plotter.show_axes()
    plotter.subplot(1, 2)
    mesh_e12_ref["Reference Strain E12"] = df_ref['E12'].values
    plotter.add_mesh(mesh_e12_ref, scalars='Reference Strain E12',
                     cmap='turbo', clim=[mesh_e12_ref['Reference Strain E12'].min(),
                                         mesh_e12_ref['Reference Strain E12'].max()])
    plotter.show_axes()
    plotter.link_views()  # Link the views so they adapt all, if one of them is moved in a specific way
    plotter.show()


def plot_displacements(mesh_path, df_ref, df_disp, filename):
    # Go to the directory with the mesh
    os.chdir(mesh_path)

    # Load the created mesh
    mesh = pv.read('mesh_alu_tube.vtk')

    # Calculate the deformed shape
    #points = mesh.points.copy()
    #points = points + df_disp.values
    #mesh.points = points

    # Create copies for the different plots
    mesh_u1_pred = mesh.copy()
    mesh_u2_pred = mesh.copy()
    mesh_u3_pred = mesh.copy()
    mesh_u1_ref = mesh.copy()
    mesh_u2_ref = mesh.copy()
    mesh_u3_ref = mesh.copy()

    # Plot the calculations
    plotter = pv.Plotter(shape=(2, 3))
    # Set the camera position in a predefined way
    plotter.camera_position = [
        (-690.6472292362095, -1640.18229140095, 1182.5880431379926),
        (855.9760919980398, 113.7949390370839, -58.44195586610785),
        (0, 0, 1)
    ]
    # Plot of the predicted solutions
    plotter.subplot(0, 0)
    mesh_u1_pred["Predicted Displacement U1"] = df_disp['U1'].values
    plotter.add_mesh(mesh_u1_pred, scalars='Predicted Displacement U1',
                     cmap='turbo', clim=[mesh_u1_pred['Predicted Displacement U1'].min(),
                                         mesh_u1_pred['Predicted Displacement U1'].max()])
    plotter.show_axes()
    plotter.subplot(0, 1)
    mesh_u2_pred["Predicted Displacement U2"] = df_disp['U2'].values
    plotter.add_mesh(mesh_u2_pred, scalars='Predicted Displacement U2',
                     cmap='turbo', clim=[mesh_u2_pred['Predicted Displacement U2'].min(),
                                         mesh_u2_pred['Predicted Displacement U2'].max()])
    plotter.show_axes()
    plotter.subplot(0, 2)
    mesh_u3_pred["Predicted Displacement U3"] = df_disp['U3'].values
    plotter.add_mesh(mesh_u3_pred, scalars='Predicted Displacement U3',
                     cmap='turbo', clim=[mesh_u3_pred['Predicted Displacement U3'].min(),
                                         mesh_u3_pred['Predicted Displacement U3'].max()])
    plotter.show_axes()
    # Plot of the predicted solutions
    plotter.subplot(1, 0)
    mesh_u1_ref["Reference Displacement U1"] = df_ref['U1'].values
    plotter.add_mesh(mesh_u1_ref, scalars='Reference Displacement U1',
                     cmap='turbo', clim=[mesh_u1_ref['Reference Displacement U1'].min(),
                                         mesh_u1_ref['Reference Displacement U1'].max()])
    plotter.show_axes()
    plotter.subplot(1, 1)
    mesh_u2_ref["Reference Displacement U2"] = df_ref['U2'].values
    plotter.add_mesh(mesh_u2_ref, scalars='Reference Displacement U2',
                     cmap='turbo', clim=[mesh_u2_ref['Reference Displacement U2'].min(),
                                         mesh_u2_ref['Reference Displacement U2'].max()])
    plotter.show_axes()
    plotter.subplot(1, 2)
    mesh_u3_ref["Reference Displacement U3"] = df_ref['U3'].values
    plotter.add_mesh(mesh_u3_ref, scalars='Reference Displacement U3',
                     cmap='turbo', clim=[mesh_u3_ref['Reference Displacement U3'].min(),
                                         mesh_u3_ref['Reference Displacement U3'].max()])
    plotter.show_axes()
    plotter.link_views()  # Link the views, so they adapt all, if one of them is moved in a specific way
    pv.global_theme.full_screen = True
    #plotter.save_graphic(filename, raster=True, painter=True)  # Save the plot as *.svg
    plotter.show()
