


import csv 
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt
import mne
mne.set_log_level('WARNING')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEG():
    def __init__(self):
        pass
        
    @staticmethod
    def load_csv_data(subdir_path, chunk_size=640, step_size=128):
        epochs = []
        for file_name in Path(subdir_path).iterdir():
            # print(file_name)
            if file_name.suffix == ".csv":
                df = pd.read_csv(file_name)
                for start_row in range(0, len(df) - chunk_size + 1, step_size):
                    chunk = df.iloc[start_row:start_row + chunk_size]
                    epochs.append(chunk.values.tolist())
        return epochs[5:-5]
    
    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a
    
    @staticmethod
    def apply_filter(data, lowcut, highcut, fs, order=4):
        b, a = EEG.butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, data, axis=0)
    
    @staticmethod
    def create_raw_from_numpy(data, sfreq=128):
        ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        raw.filter(l_freq=4, h_freq=25)
        # print(raw.get_data())
        return raw.get_data()

if __name__ == "__main__":
    path = Path(r"/media/sumit/dd6174bf-2d05-4a68-9324-d66b0a8e63762/EEG/TEST")
    epochs_dict = []
    
    pr = EEG()

    if path.exists() and path.is_dir():
        subdirectories = sorted(
            [subdir for subdir in path.iterdir() if subdir.is_dir()],
            key=lambda x: int(x.name)  # Sorting numerically
        )
        
        for subdir in subdirectories:
            print(subdir)
            epochs_dict.append(pr.load_csv_data(subdir))
        else:
            path.mkdir(parents=True, exist_ok=True)
    
    for i, epoch in enumerate(epochs_dict):
        print(f"Subject_wise_Epochs {i} shape: {np.array(epoch).shape}")

    fs = 200  
    lowcut = 5   
    highcut = 20 
    filt_epoch_dict = []

    filt_epoch_dict = []

    outer_pbar = tqdm(total=len(epochs_dict), desc="Filtering Epoch Groups", position=0, leave=True)

    for i in range(len(epochs_dict)):
        final_epochs_dict = []

        total_inner = sum(len(epochs_dict[i][j]) for j in range(len(epochs_dict[i])))
        inner_pbar = tqdm(total=total_inner, desc=f"Filtering Epochs in Group {i+1}", position=1, leave=False)

        for j in range(len(epochs_dict[i])): 
            small = []
            for k in range(len(epochs_dict[i][j])):
                np_data = np.array(epochs_dict[i][j])
                # print(len(np_data))
                # print(len(np_data[0]))
                np_data = np_data.T
                raw_data = pr.create_raw_from_numpy(np_data)
                inner_pbar.update(1)

        inner_pbar.close()
        outer_pbar.update(1)
        filt_epoch_dict.append(final_epochs_dict)

    outer_pbar.close()
    
    for i, epoch in enumerate(filt_epoch_dict):
        print(f"Subject_wise_Epochs {i} shape: {np.array(epoch).shape}")
        
    final_epoch_label = [[i for _ in epochs] for i, epochs in enumerate(filt_epoch_dict)]
    data_list = [item for sublist in filt_epoch_dict for item in sublist]
    label_list = [item for sublist in final_epoch_label for item in sublist]
    
    # Convert data to torch tensors and move to GPU
    # data_tensor = torch.tensor(np.array(data_list),  device=device)
    # label_tensor = torch.tensor(np.array(label_list),  device=device)
    
    # print("Data tensor shape:", data_tensor.shape)
    # print("Label tensor shape:", label_tensor.shape)
