import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import gc 

def load_data_chunk(file_path, group_name, start_idx, end_idx, selected_channels=None):
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        beamE = group['beamE'][start_idx:end_idx] / 1000
        hist2d_data = group['hist2d_data'][start_idx:end_idx]
        
        if selected_channels is not None:
            hist2d_data = hist2d_data[:, :, :, selected_channels]
        
        esum = np.sum(hist2d_data[:, :, :, -1], axis=(1, 2))
        
        n_events = end_idx - start_idx
        _, height, width, n_channels = hist2d_data.shape
        
    return beamE, hist2d_data, esum, (height, width, n_channels)

def get_dataset_size(file_path, group_name):
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        return group['beamE'].shape[0]

def build_model(input_shape):
    input_img = layers.Input(shape=input_shape, name='input_img')
    input_esum = layers.Input(shape=(1,), name='input_esum')
    
    x = layers.Conv2D(64, (5, 5), activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Concatenate()([x, input_esum])
    x = layers.Dense(32, activation='relu')(x)

    output_particle = layers.Dense(2, activation='softmax', name='output_particle')(x)
    output_energy = layers.Dense(1, name='output_energy')(x)

    model = models.Model(inputs=[input_img, input_esum], outputs=[output_particle, output_energy])
    return model

def plot_loss(history, save_file='loss_vs_epoch.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_file)
    plt.close()  # Close the plot to free memory

def train_batch(model, chunk_idx, files_info, chunk_size=500, val_ratio=0.2, test_ratio=0.1, epochs=15, selected_channels=None):
    """Train the model on a batch of data from each particle type"""
    all_beamE = []
    all_hist2d_data = []
    all_esum = []
    all_particle_types = []
    data_shape = None
    
    for i, (particle, file_paths) in enumerate(files_info.items()):
        for file_path, group_name in file_paths:
            total_size = get_dataset_size(file_path, group_name)
            
            if chunk_idx * chunk_size >= total_size:
                continue
                
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_size)
            
            if start_idx >= end_idx:
                continue
                
            beamE, hist2d_data, esum, shape = load_data_chunk(file_path, group_name, start_idx, end_idx, selected_channels)
            
            all_beamE.append(beamE)
            all_hist2d_data.append(hist2d_data)
            all_esum.append(esum)
            all_particle_types.append(np.full(beamE.shape[0], i))
            
            if data_shape is None:
                data_shape = shape
                
            print(f"Loaded chunk {chunk_idx} from {file_path}: {start_idx}-{end_idx}, shape {hist2d_data.shape}")
    
    if len(all_beamE) == 0:
        return False, None
        
    beamE = np.concatenate(all_beamE, axis=0)
    hist2d_data = np.concatenate(all_hist2d_data, axis=0)
    esum = np.concatenate(all_esum, axis=0)
    particles_type = np.concatenate(all_particle_types, axis=0)
    
    print(f"Combined chunk data shapes: beamE: {beamE.shape}, hist2d_data: {hist2d_data.shape}")
    
    scaler = StandardScaler()
    esum = scaler.fit_transform(esum.reshape(-1, 1)).flatten()
    
    labels_particle = tf.keras.utils.to_categorical(particles_type, num_classes=2)
    
    total_samples = beamE.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_end = int((1 - val_ratio - test_ratio) * total_samples)
    val_end = train_end + int(val_ratio * total_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = (
        [hist2d_data[train_indices], esum[train_indices].reshape(-1, 1)],
        [labels_particle[train_indices], beamE[train_indices].reshape(-1, 1)]
    )
    
    val_data = (
        [hist2d_data[val_indices], esum[val_indices].reshape(-1, 1)],
        [labels_particle[val_indices], beamE[val_indices].reshape(-1, 1)]
    )
    
    test_data = (
        [hist2d_data[test_indices], esum[test_indices].reshape(-1, 1)],
        [labels_particle[test_indices], beamE[test_indices].reshape(-1, 1)]
    )
    
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=(val_data[0], val_data[1]),
        epochs=epochs,
        batch_size=64,  
        verbose=1
    )
    
    del all_beamE, all_hist2d_data, all_esum, all_particle_types
    del beamE, hist2d_data, esum, particles_type, labels_particle
    del train_data, val_data
    gc.collect()
    
    return True, test_data

def main():

    selected_channels = None  
    
    print(f"Selected channels: {selected_channels}")
    
    particle_types = ['e-', 'pi-']
    
    files_info = {}
    
    for particle in particle_types:
        files_info[particle] = []
        
        for pattern in [
            {'energy_range': 'E1-100', 'suffix': '0.1'},
            {'energy_range': 'E1-100', 'suffix': '0.1_0'}
        ]:
            energy_range = pattern['energy_range']
            suffix = pattern['suffix']
            
            file_path = f'../dataset/{particle}_{energy_range}_2000_time_sliced{suffix}.h5'
            group_name = f'{particle}_{energy_range}_2000'
            
            files_info[particle].append((file_path, group_name))
    
    temp_beamE, temp_hist2d_data, temp_esum, input_shape = load_data_chunk(
        files_info['e-'][0][0], files_info['e-'][0][1], 0, 1, selected_channels
    )
    height, width, n_channels = input_shape
    
    model = build_model(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'output_particle': 'categorical_crossentropy', 
              'output_energy': 'mean_squared_logarithmic_error'},
        loss_weights={'output_particle': 1.0, 'output_energy': 10.0},
        metrics={'output_particle': 'accuracy', 'output_energy': 'mae'}
    )
    model.summary()
    
    all_test_data = [[], [], [], []] 
    
    chunk_idx = 0
    max_chunks = 10  
    
    while chunk_idx < max_chunks:
        print(f"\n--- Training on chunk {chunk_idx} ---")
        has_data, test_data = train_batch(
            model, chunk_idx, files_info, 
            chunk_size=250,  
            epochs=15,  
            selected_channels=selected_channels
        )
        
        if not has_data:
            print(f"No more data to process after chunk {chunk_idx}")
            break
            
        channel_str = 'all' if selected_channels is None else '_'.join(map(str, selected_channels))
        model.save(f'model_chunk_{chunk_idx}_channels_{channel_str}.keras')
        
        if test_data is not None:
            all_test_data[0].append(test_data[0][0])  # hist2d_data
            all_test_data[1].append(test_data[0][1])  # esum
            all_test_data[2].append(test_data[1][0])  # labels_particle
            all_test_data[3].append(test_data[1][1])  # beamE
        
        chunk_idx += 1
    
    if len(all_test_data[0]) > 0:
        combined_test_hist2d_data = np.concatenate(all_test_data[0], axis=0)
        combined_test_esum = np.concatenate(all_test_data[1], axis=0)
        combined_test_labels_particle = np.concatenate(all_test_data[2], axis=0)
        combined_test_beamE = np.concatenate(all_test_data[3], axis=0)
        
        combined_test_data = (
            [combined_test_hist2d_data, combined_test_esum],
            [combined_test_labels_particle, combined_test_beamE]
        )
        
        loss, particle_loss, energy_loss, acc_particle, mae_energy = model.evaluate(
            combined_test_data[0], combined_test_data[1]
        )
        print(f'Final Test Loss: {loss}, Particle Loss: {particle_loss}, Energy Loss: {energy_loss}')
        print(f'Final Test Particle Accuracy: {acc_particle}, Test Energy MAE: {mae_energy}')
        
        predictions_particle, predicted_energy = model.predict(combined_test_data[0])
        predictions_particle_classes = np.argmax(predictions_particle, axis=1)
        
        particle_accuracy = np.mean(predictions_particle_classes == np.argmax(combined_test_data[1][0], axis=1))
        print(f'Final Test Particle Type Accuracy: {particle_accuracy}')
        
        particle_labels = ['e-', 'pi-']
        true_labels = np.argmax(combined_test_data[1][0], axis=1)
        
        channel_str = 'all' if selected_channels is None else '_'.join(map(str, selected_channels))
        
        plt.figure(figsize=(12, 6))
        plt.hist(true_labels, bins=np.arange(len(particle_labels) + 1) - 0.5, alpha=0.5, label='True Particle Types')
        plt.hist(predictions_particle_classes, bins=np.arange(len(particle_labels) + 1) - 0.5, alpha=0.5, label='Predicted Particle Types')
        plt.xlabel('Particle Types')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(len(particle_labels)), particle_labels)
        plt.legend()
        plt.title(f'True vs Predicted Particle Types (Channels: {channel_str})')
        plt.savefig(f'particle_types_histogram_channels_{channel_str}.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(combined_test_data[1][1], predicted_energy, alpha=0.5, label='Predicted vs True Energy')
        plt.xlabel('True Energy (GeV)')
        plt.ylabel('Predicted Energy (GeV)')
        plt.legend()
        plt.title(f'True vs Predicted Energy (Channels: {channel_str})')
        plt.savefig(f'energy_scatter_channels_{channel_str}.png')
        plt.close()
    
    model.save(f'final_model_channels_{channel_str}.keras')

if __name__ == "__main__":
    main()
