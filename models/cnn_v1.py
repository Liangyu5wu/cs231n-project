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

# particle_types = ['e+', 'e-', 'pi+', 'pi-']
particle_types = ['e-', 'pi-']
energy_ranges = ['E1-100', 'E30-70']
# energy_ranges = ['E1-100']

beamE_list = []
hist2d_data_list = []
esum_list = []
particles_type_list = []
data_shape = None

for i, particle in enumerate(particle_types):
    for energy_range in energy_ranges:
        file_path = f'{particle}_{energy_range}_2000.h5'
        group_name = f'{particle}_{energy_range}_2000'
        beamE, hist2d_data, esum, shape = load_data(file_path, group_name)
  
        beamE_list.append(beamE)
        hist2d_data_list.append(hist2d_data)
        esum_list.append(esum)
        
        particles_type_list.append(np.full(beamE.shape[0], i))
        data_shape = shape  

beamE = np.concatenate(beamE_list, axis=0)
hist2d_data = np.concatenate(hist2d_data_list, axis=0)
esum = np.concatenate(esum_list, axis=0)
particles_type = np.concatenate(particles_type_list, axis=0)

scaler = StandardScaler()
esum = scaler.fit_transform(esum.reshape(-1, 1)).flatten()

labels_particle = tf.keras.utils.to_categorical(particles_type, num_classes=4)

height, width = data_shape
hist2d_data = hist2d_data.reshape((-1, height, width, 1))

(train_beamE, train_hist2d_data, train_esum, train_labels_particle), \
(val_beamE, val_hist2d_data, val_esum, val_labels_particle), \
(test_beamE, test_hist2d_data, test_esum, test_labels_particle) = split_data(beamE, hist2d_data, esum, labels_particle)

input_shape = (height, width, 1)

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

train_data = ([train_hist2d_data, train_esum], [train_labels_particle, train_beamE.reshape((-1, 1))])
val_data = ([val_hist2d_data, val_esum], [val_labels_particle, val_beamE.reshape((-1, 1))])
test_data = ([test_hist2d_data, test_esum], [test_labels_particle, test_beamE.reshape((-1, 1))])

best_model = train_until_convergence(train_data, val_data, test_data, input_shape)

# predicted_energy_scaled = best_model.predict(test_data[0])[1]
# predicted_energy = scaler.inverse_transform(predicted_energy_scaled)

predicted_energy = best_model.predict(test_data[0])[1]

loss, acc_particle, mae_energy = best_model.evaluate(test_data[0], test_data[1])
print(f'Test Loss: {loss}, Test Particle Accuracy: {acc_particle}, Test Energy MAE: {mae_energy}')

predictions_particle, _ = best_model.predict(test_data[0])
predictions_particle_classes = np.argmax(predictions_particle, axis=1)

particle_accuracy = np.mean(predictions_particle_classes == np.argmax(test_data[1][0], axis=1))
print(f'Test Particle Type Accuracy: {particle_accuracy}')

# particle_labels = ['e+', 'e-', 'pi+', 'pi-']
particle_labels = ['e-', 'pi-']
true_labels = np.argmax(test_data[1][0], axis=1)
plt.figure(figsize=(12, 6))
plt.hist(true_labels, bins=np.arange(len(particle_labels) + 1) - 0.5, alpha=0.5, label='True Particle Types')
plt.hist(predictions_particle_classes, bins=np.arange(len(particle_labels) + 1) - 0.5, alpha=0.5, label='Predicted Particle Types')
plt.xlabel('Particle Types')
plt.ylabel('Frequency')
plt.xticks(np.arange(len(particle_labels)), particle_labels)
plt.legend()
plt.title('True vs Predicted Particle Types')
plt.savefig('particle_types_histogram.png')

plt.figure(figsize=(12, 6))
plt.scatter(test_data[1][1], predicted_energy, alpha=0.5, label='Predicted vs True Energy')
plt.xlabel('True Energy')
plt.ylabel('Predicted Energy')
plt.legend()
plt.title('True vs Predicted Energy')
plt.savefig('energy_scatter.png')
