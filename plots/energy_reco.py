import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time

energy_values = [10, 30, 50, 90]
particle_types = ['e-', 'pi-']
particle_to_idx = {'e-': 0, 'pi-': 1}

selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]
print(f"Using channels: {selected_channels}")

def load_data(file_path, group_name, selected_channels):
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        beamE = group['beamE'][:] / 1000
        hist2d_data = group['hist2d_data'][:]
        
        if any(idx < 0 for idx in selected_channels):
            n_channels_total = hist2d_data.shape[3]
            selected_channels = [idx if idx >= 0 else n_channels_total + idx for idx in selected_channels]
        
        hist2d_data = hist2d_data[:, :, :, selected_channels]
        
        esum = np.sum(hist2d_data[:, :, :, -1], axis=(1, 2))
        
        n_events, height, width, n_channels = hist2d_data.shape
        
    return beamE, hist2d_data, esum, (height, width, n_channels)

scaler_file = 'esum_scaler.pkl'
if os.path.exists(scaler_file):
    print(f"Loading StandardScaler from {scaler_file}")
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
else:
    print("Rebuilding StandardScaler from training data...")
    
    train_data_paths = [
        ('dataset/e-_E1-100_2000_time_sliced.h5', 'e-_E1-100_2000'),
        ('dataset/e-_E1-100_2000_1_time_sliced.h5', 'e-_E1-100_2000'),
        ('dataset/pi-_E1-100_2000_time_sliced.h5', 'pi-_E1-100_2000'),
        ('dataset/pi-_E1-100_2000_1_time_sliced.h5', 'pi-_E1-100_2000')
    ]
    
    all_esum = []
    for file_path, group_name in train_data_paths:
        try:
            _, _, esum, _ = load_data(file_path, group_name, selected_channels)
            all_esum.append(esum)
            print(f"Loaded esum from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if all_esum:
        combined_esum = np.concatenate(all_esum)
        
        scaler = StandardScaler()
        scaler.fit(combined_esum.reshape(-1, 1))
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"StandardScaler saved to {scaler_file}")
    else:
        print("Warning: Could not load any training data. Using a new scaler.")
        scaler = StandardScaler()

model_path = 'best_model_time_sliced_channels_0_1_2_3_4_5_6_7_8_9_10_11.keras'
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from: {model_path}")

results = {
    'e-': {
        'accuracy': [],
        'resolution': [],
        'inverse_sqrt_energy': [],
        'energy_ratio': []
    },
    'pi-': {
        'accuracy': [],
        'resolution': [],
        'inverse_sqrt_energy': [],
        'energy_ratio': []
    }
}

for particle_type in particle_types:
    print(f"\n{'-'*50}\nEvaluating {particle_type} particles\n{'-'*50}")
    
    for energy_value in energy_values:
        print(f"\nTesting {particle_type} at {energy_value} GeV")
        
        file_path = f'dataset/{particle_type}_E{energy_value}-{energy_value}_1000_time_sliced.h5'
        group_name = f'{particle_type}_E{energy_value}-{energy_value}_1000'
        
        try:
            true_beamE, hist2d_data, esum, data_shape = load_data(file_path, group_name, selected_channels)
            esum_scaled = scaler.transform(esum.reshape(-1, 1)).flatten()
            
            expected_particle_idx = particle_to_idx[particle_type]
            labels = np.zeros((len(true_beamE), len(particle_to_idx)))
            labels[:, expected_particle_idx] = 1
            
            model_input = [hist2d_data, esum_scaled]
            predictions_particle, predicted_energy = model.predict(model_input, verbose=0)
            predictions_particle_classes = np.argmax(predictions_particle, axis=1)
            
            accuracy = np.mean(predictions_particle_classes == expected_particle_idx) * 100
            energy_mean = np.mean(predicted_energy)
            energy_std = np.std(predicted_energy)
            resolution = energy_std / energy_mean
            inverse_sqrt_energy = 1 / np.sqrt(energy_value)
            energy_ratio = energy_mean / energy_value
            
            results[particle_type]['accuracy'].append(accuracy)
            results[particle_type]['resolution'].append(resolution)
            results[particle_type]['inverse_sqrt_energy'].append(inverse_sqrt_energy)
            results[particle_type]['energy_ratio'].append(energy_ratio)
            
            print(f"Particle classification accuracy: {accuracy:.2f}%")
            print(f"Energy resolution (std/mean): {resolution:.4f}")
            print(f"Energy ratio (predicted/true): {energy_ratio:.4f}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results[particle_type]['accuracy'].append(None)
            results[particle_type]['resolution'].append(None)
            results[particle_type]['inverse_sqrt_energy'].append(1 / np.sqrt(energy_value))
            results[particle_type]['energy_ratio'].append(None)

colors = ['#1f77b4', '#ff7f0e']
markers = ['o', 's']

plt.figure(figsize=(10, 8), dpi=300)
for i, particle_type in enumerate(particle_types):
    valid_indices = [j for j, x in enumerate(results[particle_type]['accuracy']) if x is not None]
    valid_energies = [energy_values[j] for j in valid_indices]
    valid_accuracies = [results[particle_type]['accuracy'][j] for j in valid_indices]
    
    plt.plot(valid_energies, valid_accuracies, 
             marker=markers[i], linestyle='-', linewidth=2, 
             color=colors[i],
             label=f'{particle_type} particles')

plt.xlabel('Energy (GeV)', fontsize=14)
plt.ylabel('Classification Accuracy (%)', fontsize=14)
plt.title('Particle Classification Accuracy vs Energy', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(90, 105)

for i, particle_type in enumerate(particle_types):
    valid_indices = [j for j, x in enumerate(results[particle_type]['accuracy']) if x is not None]
    valid_energies = [energy_values[j] for j in valid_indices]
    valid_accuracies = [results[particle_type]['accuracy'][j] for j in valid_indices]
    
    for x, y in zip(valid_energies, valid_accuracies):
        plt.annotate(f'{y:.1f}%', 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=12)

plt.tight_layout()
plt.savefig('particle_accuracy_vs_energy.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
for i, particle_type in enumerate(particle_types):
    valid_indices = [j for j, x in enumerate(results[particle_type]['resolution']) if x is not None]
    valid_inverse_sqrt = [results[particle_type]['inverse_sqrt_energy'][j] for j in valid_indices]
    valid_resolutions = [results[particle_type]['resolution'][j] for j in valid_indices]
    
    if len(valid_inverse_sqrt) > 1:
        coeffs = np.polyfit(valid_inverse_sqrt, valid_resolutions, 1)
        poly = np.poly1d(coeffs)
        fit_line = poly(valid_inverse_sqrt)
        
        a, b = coeffs
        eq_text = f'Resolution = {a:.3f}/√E + {b:.3f}'
        
        plt.plot(valid_inverse_sqrt, valid_resolutions, 
                 marker=markers[i], linestyle='', markersize=8, 
                 color=colors[i],
                 label=f'{particle_type} data')
        
        plt.plot(valid_inverse_sqrt, fit_line, 
                 linestyle='--', linewidth=2, 
                 color=colors[i],
                 label=f'{particle_type} fit: {eq_text}')
    else:
        plt.plot(valid_inverse_sqrt, valid_resolutions, 
                 marker=markers[i], linestyle='', markersize=8, 
                 color=colors[i],
                 label=f'{particle_type} data')

for i, particle_type in enumerate(particle_types):
    valid_indices = [j for j, x in enumerate(results[particle_type]['resolution']) if x is not None]
    valid_inverse_sqrt = [results[particle_type]['inverse_sqrt_energy'][j] for j in valid_indices]
    valid_resolutions = [results[particle_type]['resolution'][j] for j in valid_indices]
    valid_energies = [energy_values[j] for j in valid_indices]
    
    for x, y, e in zip(valid_inverse_sqrt, valid_resolutions, valid_energies):
        plt.annotate(f'{e} GeV', 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(5, 5), 
                     ha='left',
                     fontsize=12)

plt.xlabel('1/√E (1/√GeV)', fontsize=14)
plt.ylabel('Energy Resolution (σ/μ)', fontsize=14)
plt.title('Energy Resolution vs 1/√E', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

energy_ticks = np.array([10, 30, 50, 90, 150])
inverse_sqrt_ticks = 1/np.sqrt(energy_ticks)
plt.gca().set_xticks(inverse_sqrt_ticks, minor=True)
plt.gca().set_xticklabels([f"{e}" for e in energy_ticks], minor=True)

plt.tight_layout()
plt.savefig('energy_resolution_vs_inverse_sqrt_energy.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
for i, particle_type in enumerate(particle_types):
    valid_indices = [j for j, x in enumerate(results[particle_type]['energy_ratio']) if x is not None]
    valid_energies = [energy_values[j] for j in valid_indices]
    valid_ratios = [results[particle_type]['energy_ratio'][j] for j in valid_indices]
    
    plt.plot(valid_energies, valid_ratios, 
             marker=markers[i], linestyle='-', linewidth=2, 
             color=colors[i],
             label=f'{particle_type} particles')
    
    for x, y in zip(valid_energies, valid_ratios):
        plt.annotate(f'{y:.3f}', 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=12)

plt.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Energy (GeV)', fontsize=14)
plt.ylabel('Energy Response (Predicted/True)', fontsize=14)
plt.title('Energy Response vs True Energy', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('energy_response_vs_energy.png', dpi=300)
plt.show()

print("\nResults Summary:")
print("-" * 75)
print(f"{'Particle':<10}{'Energy (GeV)':<15}{'Accuracy (%)':<15}{'Resolution':<15}{'Energy Ratio':<15}")
print("-" * 75)

for particle_type in particle_types:
    for i, energy in enumerate(energy_values):
        acc = results[particle_type]['accuracy'][i]
        res = results[particle_type]['resolution'][i]
        ratio = results[particle_type]['energy_ratio'][i]
        
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        res_str = f"{res:.4f}" if res is not None else "N/A"
        ratio_str = f"{ratio:.4f}" if ratio is not None else "N/A"
        
        print(f"{particle_type:<10}{energy:<15}{acc_str:<15}{res_str:<15}{ratio_str:<15}")
