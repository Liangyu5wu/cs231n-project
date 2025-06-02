import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path, group_name, selected_channels=None):
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        beamE = group['beamE'][:] / 1000
        hist2d_data = group['hist2d_data'][:]
        
        if selected_channels is not None:
            hist2d_data = hist2d_data[:, :, :, selected_channels]
        
    return beamE, hist2d_data

def plot_particles_comparison(channels, energy_range=(0, 100)):
    particle_labels = ['e-', 'pi-']
    file_paths = [
        f'dataset/{particle}_E1-100_2000_time_sliced.h5' for particle in particle_labels
    ]
    group_names = [
        f'{particle}_E1-100_2000' for particle in particle_labels
    ]
    
    if -1 in channels:
        total_channels = 12
        channels = [ch if ch >= 0 else total_channels + ch for ch in channels]
    channels = sorted(channels)
    
    fig = plt.figure(figsize=(5*len(channels) + 1, 5*2))
    grid = plt.GridSpec(2, len(channels) + 1, width_ratios=[1]*len(channels) + [0.1])
    
    for p_idx, (file_path, group_name) in enumerate(zip(file_paths, group_names)):
        beamE, hist2d_data = load_data(file_path, group_name, channels)
        
        energy_mask = (beamE >= energy_range[0]) & (beamE <= energy_range[1])
        filtered_data = hist2d_data[energy_mask]
        filtered_energies = beamE[energy_mask]
        
        if len(filtered_data) == 0:
            continue
            
        sample_idx = np.random.randint(0, len(filtered_data))
        
        vmin = float('inf')
        vmax = float('-inf')
        
        for channel_idx in range(len(channels)):
            img = filtered_data[sample_idx, :, :, channel_idx]
            vmin = min(vmin, np.min(img))
            vmax = max(vmax, np.max(img))
        
        vmin = vmin * 0.95 if vmin > 0 else vmin * 1.05
        vmax = vmax * 1.05 if vmax > 0 else vmax * 0.95
        
        cbar_ax = fig.add_subplot(grid[p_idx, -1])
        
        for j in range(len(channels)):
            ax = fig.add_subplot(grid[p_idx, j])
            
            img = filtered_data[sample_idx, :, :, j]
            im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            
            if j == 0:
                ax.set_ylabel(f"{particle_labels[p_idx]} ({filtered_energies[sample_idx]:.1f} GeV)")
            
            ax.set_title(f"Channel {channels[j]}")
        
        plt.colorbar(im, cax=cbar_ax)
    
    plt.suptitle(f"Particle Comparison ({energy_range[0]}-{energy_range[1]} GeV)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    channel_str = '_'.join(map(str, channels))
    plt.savefig(f'particle_comparison_{energy_range[0]}-{energy_range[1]}GeV_channels_{channel_str}.png', 
                bbox_inches='tight')
    plt.show()

# selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]
selected_channels = [0, 4,  8, 9, 10, -1]
plot_particles_comparison(selected_channels, energy_range=(5, 10))
