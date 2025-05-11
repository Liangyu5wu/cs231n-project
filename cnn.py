import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

def load_data(file_path, group_name):
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        beamE = group['beamE'][:] / 1000
        hist2d_data = group['hist2d_data'][:]
        
        esum = np.sum(hist2d_data, axis=(1, 2))

        n_events, height, width = hist2d_data.shape
        
    return beamE, hist2d_data, esum, (height, width)

def concatenate_data(data_list):
    concatenated_beamE = np.concatenate([data[0] for data in data_list], axis=0)
    concatenated_hist2d_data = np.concatenate([data[1] for data in data_list], axis=0)
    concatenated_esum = np.concatenate([data[2] for data in data_list], axis=0)
    return concatenated_beamE, concatenated_hist2d_data, concatenated_esum, data_list[0][3]

def split_data(beamE, hist2d_data, esum, labels_particle, split_ratio=(0.7, 0.2, 0.1)):
    total_samples = beamE.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_end = int(split_ratio[0] * total_samples)
    val_end = train_end + int(split_ratio[1] * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_beamE, val_beamE, test_beamE = beamE[train_indices], beamE[val_indices], beamE[test_indices]
    train_hist2d_data, val_hist2d_data, test_hist2d_data = hist2d_data[train_indices], hist2d_data[val_indices], hist2d_data[test_indices]
    train_esum, val_esum, test_esum = esum[train_indices], esum[val_indices], esum[test_indices]
    train_labels_particle, val_labels_particle, test_labels_particle = labels_particle[train_indices], labels_particle[val_indices], labels_particle[test_indices]

    return (train_beamE, train_hist2d_data, train_esum, train_labels_particle), \
           (val_beamE, val_hist2d_data, val_esum, val_labels_particle), \
           (test_beamE, test_hist2d_data, test_esum, test_labels_particle)

def build_model(input_shape):
    input_img = layers.Input(shape=input_shape, name='input_img')
    input_esum = layers.Input(shape=(1,), name='input_esum')

    x = layers.Conv2D(64, (5, 5), activation='relu')(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(6, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)  
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x) 
    x = layers.Dense(32, activation='relu')(x)

    x = layers.Concatenate()([x, input_esum])

    output_particle = layers.Dense(4, activation='softmax', name='output_particle')(x)

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

def train_until_convergence(train_data, val_data, test_data, input_shape, max_rounds=1, epochs_per_round=20, patience=10):
    best_val_loss = np.inf
    round_counter = 0
    
    while round_counter < max_rounds:
        model = build_model(input_shape)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
                      loss={'output_particle': 'categorical_crossentropy', 
                            'output_energy': 'mean_squared_logarithmic_error'},
                      loss_weights={'output_particle': 0.5, 'output_energy': 2.0},  
                      metrics={'output_particle': 'accuracy', 'output_energy': 'mae'})

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience) 
        checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

        history = model.fit(train_data[0],
                            train_data[1],
                            validation_data=(val_data[0], val_data[1]),
                            epochs=epochs_per_round,
                            batch_size=128,
                            callbacks=[early_stopping, checkpoint])

        plot_loss(history, save_file=f'loss_vs_epoch_round_{round_counter}.png')

        model = tf.keras.models.load_model('best_model.keras')
        val_loss = model.evaluate(val_data[0], val_data[1])[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            round_counter += 1
        else:
            print(f"Validation loss did not improve after {round_counter} rounds.")
            break
    
    return model
