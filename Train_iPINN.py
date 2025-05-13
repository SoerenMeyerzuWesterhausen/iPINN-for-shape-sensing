
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from scipy.optimize import minimize
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Concatenate, Layer
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras import Model
from Postprocess import plot_displacements

tf.config.run_functions_eagerly(True)


# Definition of the training function using the Adam optimizer
def train_with_adam(bc_ids, load_ids, X_all_sensors, X_all_coordinates, y_all_displacements, x_scaler, e_scaler,
                    strain_field, model, max_iterations, tol_loss, sensor_ids):
    optimizer = Adam(learning_rate=0.0005)
    total_loss_history = []
    data_loss_history = []
    strain_loss_history = []
    bc_loss_history = []
    best_loss = float('inf')
    best_model_weights = None

    for iteration in range(max_iterations):
        with tf.GradientTape(persistent=True) as tape:
            x_coord = tf.convert_to_tensor(X_all_coordinates['x'], dtype=tf.float32)
            y_coord = tf.convert_to_tensor(X_all_coordinates['y'], dtype=tf.float32)
            z_coord = tf.convert_to_tensor(X_all_coordinates['z'], dtype=tf.float32)

            # Tell the tape to watch the coordinate tensors explicitly
            tape.watch(x_coord)
            tape.watch(y_coord)
            tape.watch(z_coord)

            sensor_inputs = [tf.convert_to_tensor(X_all_sensors[key], dtype=tf.float32) for key in
                             sorted(X_all_sensors.keys())]

            y_pred = model([x_coord, y_coord, z_coord] + sensor_inputs, training=True)

            # Calculate the gradients for the predicted outputs with respect to the coordinates
            grad_u_x = tape.gradient(y_pred[0], [x_coord])
            grad_u_y = tape.gradient(y_pred[0], [y_coord])
            grad_u_z = tape.gradient(y_pred[0], [z_coord])
            grad_v_x = tape.gradient(y_pred[1], [x_coord])
            grad_v_y = tape.gradient(y_pred[1], [y_coord])
            grad_v_z = tape.gradient(y_pred[1], [z_coord])
            grad_w_x = tape.gradient(y_pred[2], [x_coord])
            grad_w_y = tape.gradient(y_pred[2], [y_coord])
            grad_w_z = tape.gradient(y_pred[2], [z_coord])

            # Compute strain components
            scaling_factors = tf.convert_to_tensor(x_scaler, dtype=tf.float32)
            scaling_factors = tf.reshape(scaling_factors, (-1, 1, 1))
            epsilon_xx = grad_u_x[0] / x_scaler[0]
            epsilon_yy = grad_v_y[0] / x_scaler[1]
            epsilon_zz = grad_w_z[0] / x_scaler[2]
            epsilon_xy = 0.5 * ((grad_u_y[0] / x_scaler[1]) + (grad_v_x[0] / x_scaler[0]))
            epsilon_xz = 0.5 * ((grad_u_z[0] / x_scaler[2]) + (grad_w_x[0] / x_scaler[0]))

            # Get indices of side faces of the tube at y = 0 or 120
            mask_side_faces = tf.logical_or(
                tf.math.equal(y_coord, 0.0),
                tf.math.equal(y_coord, 120.0)
            )

            # Exchange the strains e_yy and e_xy for the side faces with e_zz and e_xz
            epsilon_yy = tf.where(mask_side_faces, epsilon_zz, epsilon_yy)
            epsilon_xy = tf.where(mask_side_faces, epsilon_xz, epsilon_xy)

            # Loss computation
            total_loss, data_loss, strain_loss, bc_loss, _ = custom_loss(bc_ids, load_ids, y_all_displacements,
                                                                         y_pred[0], y_pred[1], y_pred[2],
                                                                         epsilon_xx, epsilon_yy, epsilon_xy,
                                                                         strain_field)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Save model weights to store the best settings for later use
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_weights = model.get_weights()
        
        # Store the losses for plots later
        total_loss_history.append(total_loss.numpy())
        data_loss_history.append(data_loss.numpy())
        strain_loss_history.append(strain_loss.numpy())
        bc_loss_history.append(bc_loss.numpy())

        if iteration % 100 == 0 or total_loss < tol_loss:
            print(f"Iteration {iteration}: Total Loss = {total_loss.numpy()}, Data Loss = {data_loss.numpy()}, "
                  f"Strain Loss = {strain_loss.numpy()}, BC Loss = {bc_loss.numpy()}")

        if total_loss < tol_loss:
            print("Stopping criteria reached.")
            break

    model.set_weights(best_model_weights)
    return {
        'total_loss': total_loss_history,
        'data_loss': data_loss_history,
        'strain_loss': strain_loss_history,
        'bc_loss': bc_loss_history
    }


def lbfgs_optimization(model, loss_function, X_all_coordinates, X_all_sensors, y_all_displacements, bc_ids, load_ids,
                       strain_field, x_scaler, e_scaler):
    """
    Fine-tunes the model using the L-BFGS optimizer and tracks loss history.
    """

    def loss_and_grads(flat_weights):
        """
        Computes the loss and gradients for the current model weights.
        """
        reshaped_weights = unflatten_weights(flat_weights, weight_shapes)
        model.set_weights(reshaped_weights)

        with tf.GradientTape(persistent=True) as tape:
            x_coord = tf.convert_to_tensor(X_all_coordinates['x'], dtype=tf.float32)
            y_coord = tf.convert_to_tensor(X_all_coordinates['y'], dtype=tf.float32)
            z_coord = tf.convert_to_tensor(X_all_coordinates['z'], dtype=tf.float32)

            # Tell the tape to watch the coordinate tensors explicitly
            tape.watch(x_coord)
            tape.watch(y_coord)
            tape.watch(z_coord)

            sensor_inputs = [tf.convert_to_tensor(X_all_sensors[key], dtype=tf.float32) for key in
                             sorted(X_all_sensors.keys())]

            y_pred = model([x_coord, y_coord, z_coord] + sensor_inputs, training=True)

            # Calculate the gradients for the predicted outputs with respect to the coordinates
            grad_u_x = tape.gradient(y_pred[0], [x_coord])
            grad_u_y = tape.gradient(y_pred[0], [y_coord])
            grad_u_z = tape.gradient(y_pred[0], [z_coord])
            grad_v_x = tape.gradient(y_pred[1], [x_coord])
            grad_v_y = tape.gradient(y_pred[1], [y_coord])
            grad_v_z = tape.gradient(y_pred[1], [z_coord])
            grad_w_x = tape.gradient(y_pred[2], [x_coord])
            grad_w_y = tape.gradient(y_pred[2], [y_coord])
            grad_w_z = tape.gradient(y_pred[2], [z_coord])

            # Compute strain components
            scaling_factors = tf.convert_to_tensor(x_scaler, dtype=tf.float32)
            scaling_factors = tf.reshape(scaling_factors, (-1, 1, 1))
            epsilon_xx = grad_u_x[0] / x_scaler[0]
            epsilon_yy = grad_v_y[0] / x_scaler[1]
            epsilon_zz = grad_w_z[0] / x_scaler[2]
            epsilon_xy = 0.5 * ((grad_u_y[0] / x_scaler[1]) + (grad_v_x[0] / x_scaler[0]))
            epsilon_xz = 0.5 * ((grad_u_z[0] / x_scaler[2]) + (grad_w_x[0] / x_scaler[0]))

            # Get indices of side faces of the tube at y = 0 or 120
            mask_side_faces = tf.logical_or(
                tf.math.equal(y_coord, 0.0),
                tf.math.equal(y_coord, 120.0)
            )

            # Exchange the strains e_yy and e_xy for the side faces with e_zz and e_xz
            epsilon_yy = tf.where(mask_side_faces, epsilon_zz, epsilon_yy)
            epsilon_xy = tf.where(mask_side_faces, epsilon_xz, epsilon_xy)

            # Loss computation
            total_loss, data_loss, strain_loss, bc_loss, _ = custom_loss(bc_ids, load_ids, y_all_displacements,
                                                                         y_pred[0], y_pred[1], y_pred[2],
                                                                         epsilon_xx, epsilon_yy, epsilon_xy,
                                                                         strain_field)

        #gradients = tape.gradient(total_loss, model.trainable_variables)
        grads = tape.gradient(total_loss, model.trainable_variables)
        grad_flat = np.concatenate([g.numpy().flatten() for g in grads])

        # Append to loss history
        lbfgs_loss_history['total_loss'].append(((data_loss) + (strain_loss) + (bc_loss)).numpy())
        lbfgs_loss_history['data_loss'].append(data_loss.numpy())
        lbfgs_loss_history['strain_loss'].append(strain_loss.numpy())
        lbfgs_loss_history['bc_loss'].append(bc_loss.numpy())

        return total_loss.numpy(), grad_flat

    # Helper function to flatten weights
    def flatten_weights(weights):
        return np.concatenate([w.flatten() for w in weights])

    # Helper function to unflatten weights
    def unflatten_weights(flat_weights, shapes):
        reshaped = []
        index = 0
        for shape in shapes:
            size = np.prod(shape)
            reshaped.append(flat_weights[index:index + size].reshape(shape))
            index += size
        return reshaped

    # Flatten model weights and save shapes
    initial_weights = model.get_weights()
    weight_shapes = [w.shape for w in initial_weights]
    flat_weights = flatten_weights(initial_weights)

    # Loss history for L-BFGS
    lbfgs_loss_history = {
        'total_loss': [],
        'data_loss': [],
        'strain_loss': [],
        'bc_loss': []
    }

    # Optimize using L-BFGS
    res = minimize(
        fun=lambda w: loss_and_grads(w)[0],
        jac=lambda w: loss_and_grads(w)[1],
        x0=flat_weights,
        method='L-BFGS-B',
        options={'maxiter': 50000, 'disp': True}
    )

    # Set the optimized weights back to the model
    optimized_weights = unflatten_weights(res.x, weight_shapes)
    model.set_weights(optimized_weights)

    return model, lbfgs_loss_history


def train_with_adam_and_lbfgs(bc_ids, load_ids, X_all_sensors, X_all_coordinates, y_all_displacements, x_scaler,
                              e_scaler, strain_field, model, sensor_ids, max_iterations=10000, tol_loss=6e-4):
    """
    Train the model with Adam first and then fine-tune with L-BFGS.
    """
    # Adam Optimization
    print("Starting training with Adam...")
    history_adam = train_with_adam(bc_ids, load_ids, X_all_sensors, X_all_coordinates, y_all_displacements, x_scaler,
                                   e_scaler, strain_field, model, max_iterations, tol_loss, sensor_ids)

    # L-BFGS Optimization
    print("Switching to L-BFGS for fine-tuning...")
    model, history_lbfgs = lbfgs_optimization(model, custom_loss, X_all_coordinates, X_all_sensors, y_all_displacements,
                                              bc_ids, load_ids, strain_field, x_scaler, e_scaler)

    # Combine histories
    combined_history = {
        'total_loss': history_adam['total_loss'] + history_lbfgs['total_loss'],
        'data_loss': history_adam['data_loss'] + history_lbfgs['data_loss'],
        'strain_loss': history_adam['strain_loss'] + history_lbfgs['strain_loss'],
        'bc_loss': history_adam['bc_loss'] + history_lbfgs['bc_loss']
    }
    trained_model = model

    return combined_history, trained_model


# Definition of the custom loss-function
def custom_loss(bc_ids, load_ids, y_all_displacements, y_pred_u1, y_pred_u2, y_pred_u3,
                epsilon_xx, epsilon_yy, epsilon_xy, strain_field):

    # Calculate strain loss with strains
    strain_e11_loss = tf.reduce_mean(tf.reduce_mean(tf.square((epsilon_xx * 10e3) - (strain_field['E11'] * 10e3)), axis=0))
    strain_e22_loss = tf.reduce_mean(tf.reduce_mean(tf.square((epsilon_yy * 10e4) - (strain_field['E22'] * 10e4)), axis=0))
    strain_e12_loss = tf.reduce_mean(tf.reduce_mean(tf.square((epsilon_xy * 10e4) - (strain_field['E12'] * 10e4)), axis=0))

    strain_loss = ((1.0 * strain_e11_loss + 1.0 * strain_e22_loss + 1.0 * strain_e12_loss) / 3)

    # Data loss as MSE
    data_loss_x = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred_u1 - y_all_displacements['u1']), axis=1))
    data_loss_y = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred_u2 - y_all_displacements['u2']), axis=1))
    data_loss_z = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred_u3 - y_all_displacements['u3']), axis=1))
    # Combine the different displacement losses with different weights
    data_loss = ((1.0 * data_loss_x + 10.0 * data_loss_y + 20.0 * data_loss_z) / 3)
    #print("Predictions U3: \n", y_pred_u3)
    #print("Reference U3: \n", y_all_displacements['u3'])

    # Boundary-condition-loss as MSE for clamped end
    bc_ids = tf.convert_to_tensor(bc_ids, dtype=tf.int32)
    disp_u1_filtered_bc = tf.gather(y_pred_u1, bc_ids, axis=1)  # Get the displacements only at the indices with boundary conditions
    disp_u2_filtered_bc = tf.gather(y_pred_u2, bc_ids, axis=1)
    disp_u3_filtered_bc = tf.gather(y_pred_u3, bc_ids, axis=1)

    bc_loss_u1 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u1_filtered_bc), axis=0))
    bc_loss_u2 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u2_filtered_bc), axis=0))
    bc_loss_u3 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u3_filtered_bc), axis=0))
    bc_loss = ((bc_loss_u1 + bc_loss_u2 + bc_loss_u3) / 3)

    # Boundary-condition-loss as MSE for the load application part
    load_ids = tf.convert_to_tensor(load_ids, dtype=tf.int32)
    disp_u1_filtered_load_app = tf.gather(y_pred_u1, load_ids, axis=1)  # Get the displacements only at the indices with load application
    disp_u2_filtered_load_app = tf.gather(y_pred_u2, load_ids, axis=1)
    disp_u3_filtered_load_app = tf.gather(y_pred_u3, load_ids, axis=1)

    disp_u1_ref_filtered_load_app = tf.gather(tf.convert_to_tensor(y_all_displacements['u1'], dtype=tf.float32),
                                              load_ids, axis=1)  # Get the displacements only at the indices with load application
    disp_u2_ref_filtered_load_app = tf.gather(tf.convert_to_tensor(y_all_displacements['u2'], dtype=tf.float32),
                                              load_ids, axis=1)
    disp_u3_ref_filtered_load_app = tf.gather(tf.convert_to_tensor(y_all_displacements['u3'], dtype=tf.float32),
                                              load_ids, axis=1)

    bc_load_app_loss_u1 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u1_filtered_load_app -
                                                                  disp_u1_ref_filtered_load_app), axis=0))
    bc_load_app_loss_u2 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u2_filtered_load_app -
                                                                  disp_u2_ref_filtered_load_app), axis=0))
    bc_load_app_loss_u3 = tf.reduce_mean(tf.reduce_mean(tf.square(disp_u3_filtered_load_app -
                                                                  disp_u3_ref_filtered_load_app), axis=0))
    bc_load_loss = ((bc_load_app_loss_u1 + bc_load_app_loss_u2 + bc_load_app_loss_u3) / 3)

    # Calculate the total loss with weights
    total_loss = (100.0 * data_loss + 0.1 * strain_loss + 10.0 * bc_loss + 10.0 * bc_load_loss)
    print(f"Total Loss: {total_loss}; Data Loss: {data_loss}; Strain Loss: {strain_loss}; BC Loss: {bc_loss};"
          f"Load application Loss: {bc_load_loss}")

    return total_loss, data_loss, strain_loss, bc_loss, bc_load_loss


# To plot the loss history
def plot_loss_history(total_loss_history, data_loss_history, physics_loss_history, bc_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss', color='red', linestyle='--')
    plt.plot(data_loss_history, label='Data Loss', color='orange')
    plt.plot(physics_loss_history, label='Physics-Informed Loss', color='green')
    plt.plot(bc_loss_history, label='Boundary Condition Loss', color='blue')

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--")
    plt.title('Losses during Training for pure Bending')
    plt.show()


# Function to train the neural network
def train_model(bc_ids, load_ids, X_all_sensors, X_all_coordinates, y_all_displacements, x_scaler, e_scaler,
                save_path, strain_field, main_path, map_indices, sensor_ids, number_of_sensors):

    # Defines loss function to call the custom loss
    def loss_wrapper(bc_ids, load_ids, y_all_displacements, X_all_coordinates, X_all_sensors,
                     model, x_scaler, e_scaler, strain_field):

        def loss_function(y_all_displacements, y_pred_u1, y_pred_u2, y_pred_u3, epsilon_xx, epsilon_yy, epsilon_xy):
            total_loss = custom_loss(bc_ids, load_ids, y_all_displacements, y_pred_u1, y_pred_u2, y_pred_u3,
                                     epsilon_xx, epsilon_yy, epsilon_xy, strain_field)
            return total_loss

        return loss_function

    # Extract collocation point arrays
    X_all_x = X_all_coordinates['x']
    X_all_y = X_all_coordinates['y']
    X_all_z = X_all_coordinates['z']

    # Dynamisch alle Sensor-Arrays extrahieren
    X_sensors = [X_all_sensors[sensor_id] for sensor_id in sorted(X_all_sensors.keys(), key=lambda x: int(x))]

    # Build the iPINN
    i_pinn = IPINN(X_all_x, X_all_y, X_all_z, X_sensors)
    model = i_pinn.model
    start_time = time.time()

    # Training
    model.compile(optimizer='adam', loss=loss_wrapper(bc_ids, load_ids, y_all_displacements, X_all_coordinates,
                                                      X_all_sensors, model, x_scaler, e_scaler, strain_field))
    history, trained_model = train_with_adam_and_lbfgs(bc_ids, load_ids, X_all_sensors, X_all_coordinates,
                                                       y_all_displacements, x_scaler, e_scaler, strain_field, model, sensor_ids)

    end_time = time.time()
    training_time = (end_time - start_time) / 60

    # Save the trained iPINN
    os.chdir(save_path)
    trained_model.save("Trained_iPINN_Bending_"+str(int(number_of_sensors))+"Sensors.keras")
    os.chdir(main_path)

    print(f"Time needed for training:  {training_time:.2f} minutes")
    
    # Plot the loss history from training
    plot_loss_history(history['total_loss'], history['data_loss'], history['strain_loss'], history['bc_loss'])


# Variant as a 'normal' FNN with inputs x, y, z, sensor_1, sensor_2, ... sensor_n
class IPINN:
    def __init__(self, X_all_x, X_all_y, X_all_z, X_sensors):
        self.model = self._build_model(
            input_shapes={
                'x': X_all_x.shape[1],
                'y': X_all_y.shape[1],
                'z': X_all_z.shape[1],
                'sensors': [sensor.shape[1] for sensor in X_sensors]
            }
        )

    def _build_model(self, input_shapes):
        # Inputs
        input_x = layers.Input(shape=(input_shapes['x'],), name='x_coordinates')
        input_y = layers.Input(shape=(input_shapes['y'],), name='y_coordinates')
        input_z = layers.Input(shape=(input_shapes['z'],), name='z_coordinates')
        input_sensors = [
            layers.Input(shape=(s,), name=f'sensor_{i+1}')
            for i, s in enumerate(input_shapes['sensors'])
        ]

        # Coordinate Branch
        coords = [self._build_coord_branch(inp) for inp in [input_x, input_y, input_z]]

        # Sensor Branches
        sensor_branches = [self._build_sensor_branch(sensor_inp) for sensor_inp in input_sensors]

        # Merge all
        merged = layers.Concatenate()(coords + sensor_branches)
        hidden = Dense(64, activation='relu')(merged)

        # Outputs
        output_u1 = Dense(input_shapes['x'], activation='linear', name='u1')(hidden)
        output_u2 = Dense(input_shapes['y'], activation='linear', name='u2')(hidden)
        output_u3 = Dense(input_shapes['z'], activation='linear', name='u3')(hidden)

        return models.Model(inputs=[input_x, input_y, input_z] + input_sensors, outputs=[output_u1, output_u2, output_u3])

    def _build_coord_branch(self, input_layer):
        x = Dense(32, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        return x

    def _build_sensor_branch(self, input_layer):
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2())(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        return x
