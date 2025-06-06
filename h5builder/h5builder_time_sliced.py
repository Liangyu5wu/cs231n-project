import ROOT
import h5py
import numpy as np
import os

def save_root_to_h5(particle_type, energy_min, energy_max, event_read=2000, event_count=2000, timestep=0.1):


    energy_range = f"{energy_min}-{energy_max}"
    output_file = f"{particle_type}_E{energy_range}_{event_count}_time_sliced{timestep}.h5"
    
    print(f"Processing {particle_type} with energy {energy_range} GeV")
    print(f"Output file: {output_file}")
    
    x_bins = np.arange(-25, 15, 0.25)
    y_bins = np.arange(-14, 21, 0.25)
    
    time_cuts = np.arange(7, 20.1, timestep).tolist()

    time_cuts_all = time_cuts + [float("inf")]
    num_time_slices = len(time_cuts_all)
    
    with h5py.File(output_file, "w") as h5_file:
        group_name = f"{particle_type}_E{energy_range}_{event_count}"
        group = h5_file.create_group(group_name)
        
        beamE_dataset = group.create_dataset("beamE", (0,), maxshape=(None,), dtype='f8')
        
        hist2d_dataset = group.create_dataset(
            "hist2d_data", 
            (0, len(y_bins) - 1, len(x_bins) - 1, num_time_slices), 
            maxshape=(None, len(y_bins) - 1, len(x_bins) - 1, num_time_slices), 
            dtype='f8', 
            compression="gzip"
        )
        
        time_info_dataset = group.create_dataset(
            "time_cuts", 
            data=np.array(time_cuts_all),
            dtype='f8'
        )

        def append_to_dataset(dataset, new_data):
            dataset.resize(dataset.shape[0] + new_data.shape[0], axis=0)
            dataset[-new_data.shape[0]:] = new_data

        for i in range(1, event_read // 10 + 1):
            file_name = f"{particle_type}_E{energy_range}_{event_count}_0/mc_{particle_type}_job_run1_{i}_Test_10evt_{particle_type}_{energy_min}_{energy_max}.root"
            if not os.path.exists(file_name):
                print(f"File {file_name} doesn't exist, skipping.")
                continue

            print(f"Processing file: {file_name}")
            root_file = ROOT.TFile.Open(file_name)
            tree = root_file.Get("tree")

            beamE_list = []
            hist2d_data_list = []

            for event in range(tree.GetEntries()):
                tree.GetEntry(event)
                
                beamE = tree.GetLeaf("beamE").GetValue()
                beamE_list.append(beamE)

                hist2d_list = []
                for t_idx, t_cut in enumerate(time_cuts_all):
                    hist_name = f"hist2d_{event}_{t_idx}"
                    hist2d = ROOT.TH2F(
                        hist_name, hist_name,
                        len(x_bins) - 1, x_bins, 
                        len(y_bins) - 1, y_bins
                    )
                    hist2d_list.append(hist2d)

                nentries = tree.OP_pos_final_x.size()
                print(f"Event {event} has {nentries} entries")
                
                for n in range(nentries):
                    OP_pos_final_x = tree.OP_pos_final_x[n]
                    OP_pos_final_y = tree.OP_pos_final_y[n]
                    OP_pos_final_z = tree.OP_pos_final_z[n]
                    OP_isCoreC = tree.OP_isCoreC[n]
                    OP_time_final = tree.OP_time_final[n]
                    
                    if OP_pos_final_z > 80 and OP_isCoreC == 1:

                        for t_idx, t_cut in enumerate(time_cuts_all):
                            if OP_time_final < t_cut:
                                hist2d_list[t_idx].Fill(OP_pos_final_x, OP_pos_final_y)

                event_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1, num_time_slices))
                for t_idx in range(num_time_slices):
                    hist2d = hist2d_list[t_idx]
                    for y in range(len(y_bins) - 1):
                        for x in range(len(x_bins) - 1):
                            event_data[y, x, t_idx] = hist2d.GetBinContent(x + 1, y + 1)
                    hist2d.Delete()
                
                hist2d_data_list.append(event_data)

            beamE_data = np.array(beamE_list, dtype='f8')
            hist2d_data = np.array(hist2d_data_list, dtype='f8')
            
            append_to_dataset(beamE_dataset, beamE_data)
            append_to_dataset(hist2d_dataset, hist2d_data)

            print(f"Data from {file_name} have been saved.")

    print(f"All data have been saved to {output_file}.")
particle_type = "e+"
energy_min = 1
energy_max = 100
event_read = 4000
event_count = 4000
timestep = 0.1

save_root_to_h5(particle_type, energy_min, energy_max, event_read, event_count, timestep)
