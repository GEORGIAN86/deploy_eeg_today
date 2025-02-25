import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

class EEG:
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
        return filtfilt(b, a, data, axis=1)

def load_csv_data(subdir_path, chunk_size=640, step_size=128, sfreq=128):
    epochs = []
    for file_name in Path(subdir_path).iterdir():
        if file_name.suffix == ".csv":
            df = pd.read_csv(file_name)
            data = df.to_numpy().T  
            
            filtered_data = EEG.apply_filter(data, lowcut=4, highcut=25, fs=sfreq).T

            for start_row in range(0, filtered_data.shape[0] - chunk_size + 1, step_size):
                chunk = filtered_data[start_row:start_row + chunk_size]
                epochs.append(chunk.tolist())
    return epochs[5:-5]

if __name__ == "__main__":
    path = Path(r"/media/sumit/dd6174bf-2d05-4a68-9324-d66b0a8e63762/EEG/TEST")
    epochs_dict = []

    if path.exists() and path.is_dir():
        subdirectories = sorted(
            [subdir for subdir in path.iterdir() if subdir.is_dir()],
            key=lambda x: int(x.name)  # Sorting numerically
        )
        
        for subdir in subdirectories:
            print(subdir)
            epochs_dict.append(load_csv_data(subdir))
    
    for i, epoch in enumerate(epochs_dict):
        print(f"Subject_wise_Epochs {i} shape: {np.array(epoch).shape}")
