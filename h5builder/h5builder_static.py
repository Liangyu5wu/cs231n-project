import ROOT
import h5py
import numpy as np
import os

def save_root_to_h5(particle_type, energy_range, event_read, event_count, energy_min, energy_max, output_file):

    x_bins = np.arange(-15, 5, 0.4)
    y_bins = np.arange(-8, 15, 0.4)
    
    with h5py.File(output_file, "w") as h5_file:
        group_name = f"{particle_type}_E{energy_range}_{event_count}"
        group = h5_file.create_group(group_name)
        
        beamE_dataset = group.create_dataset("beamE", (0,), maxshape=(None,), dtype='f8')
        hist2d_data_dataset = group.create_dataset("hist2d_data", (0, len(y_bins) - 1, len(x_bins) - 1), maxshape=(None, len(y_bins) - 1, len(x_bins) - 1), dtype='f8', compression="gzip")

        def append_to_dataset(dataset, new_data):
            dataset.resize(dataset.shape[0] + new_data.shape[0], axis=0)
            dataset[-new_data.shape[0]:] = new_data

        for i in range(1, event_read // 10 + 1):
            file_name = f"/fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/cs231n/{particle_type}_E{energy_range}_{event_count}/mc_{particle_type}_job_run1_{i}_Test_10evt_{particle_type}_{energy_min}_{energy_max}.root"
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

                hist2d = ROOT.TH2F("hist2d", f"hist2d_{i}_{event}", len(x_bins) - 1, x_bins, len(y_bins) - 1, y_bins)

                nentries = tree.OP_pos_final_x.size()
                print(f"Event {event} has {nentries} entries")
                
                for n in range(nentries):
                    OP_pos_final_x = tree.OP_pos_final_x[n]
                    OP_pos_final_y = tree.OP_pos_final_y[n]
                    OP_pos_final_z = tree.OP_pos_final_z[n]
                    OP_isCoreC = tree.OP_isCoreC[n]
                    
                    if OP_pos_final_z > 80 and OP_isCoreC == 1:
                        hist2d.Fill(OP_pos_final_x, OP_pos_final_y)

                hist2d_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
                for y in range(len(y_bins) - 1):
                    for x in range(len(x_bins) - 1):
                        hist2d_data[y, x] = hist2d.GetBinContent(x + 1, y + 1)

                hist2d_data_list.append(hist2d_data)
                hist2d.Delete()

            beamE_data = np.array(beamE_list, dtype='f8')
            hist2d_data = np.array(hist2d_data_list, dtype='f8')

            append_to_dataset(beamE_dataset, beamE_data)
            append_to_dataset(hist2d_data_dataset, hist2d_data)

            print(f"Data from {file_name} have been saved.")

    print(f"All data have been saved to {output_file}.")


particle_type = "e-"
energy_range = "30-70"
event_read = 2000
event_count = 2000
energy_min = 30
energy_max = 70
output_file = "e-_E30-70_2000.h5"

save_root_to_h5(particle_type, energy_range, event_read, event_count, energy_min, energy_max, output_file)
