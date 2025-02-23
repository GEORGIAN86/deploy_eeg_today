# import csv 
# from tqdm import tqdm
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import numpy as np
# from scipy.signal import butter, filtfilt

# class EEG():
    
#     def __init__(self):
#         pass
        
        
#     @staticmethod
#     def load_csv_data(subdir_path,chunk_size=640, step_size=128):
#         epochs = []

#         for file_name in Path(subdir_path).iterdir():
#             print(file_name)
#             if file_name.suffix == ".csv":
#                 df = pd.read_csv(file_name)
#                 for start_row in range(0, len(df) - chunk_size + 1, step_size):
#                     chunk = df.iloc[start_row:start_row + chunk_size]
#                     epochs.append(chunk.values.tolist())
                
#         # return epochs
#         return epochs[5:-5]
    
#     @staticmethod
#     def butter_bandpass(lowcut, highcut, fs, order=4):
#         nyquist = 0.5 * fs
#         low = lowcut / nyquist
#         high = highcut / nyquist
#         b, a = butter(order, [low, high], btype="band")
#         return b, a
    
#     @staticmethod
#     def apply_filter(data, lowcut, highcut, fs, order=4):
#         b, a = pr.butter_bandpass(lowcut, highcut, fs, order)
#         return filtfilt(b, a, data, axis=0)


        
            
            
    
    

        



# if __name__ == "__main__":
#     path = Path(r"C:\EEG_MODEL_NEW\Pipeline\TEST")
#     epochs_dict = []
    
#     pr = EEG()

#     if path.exists() and path.is_dir():
#         subdirectories = [subdir for subdir in path.iterdir() if subdir.is_dir()]
#         for subdir in subdirectories:
#             epochs_dict.append(pr.load_csv_data(subdir))
#     else:
#         path.mkdir(parents=True, exist_ok=True)
        
#     # print(len(epochs_dict))
#     # print(len(epochs_dict[0]))
#     # print(len(epochs_dict[0][0]))
#     # print(len(epochs_dict[0][0][0]))

#     for i, epoch in enumerate(epochs_dict):
#         print(f"Subject_wise_Epochs {i} shape: {np.array(epoch).shape}")

#     fs = 200  
#     t = np.linspace(0, 1, fs, endpoint=False)
    
#     lowcut = 5   
#     highcut = 20 
#     filt_epoch_dict = []
#     outer_pbar = tqdm(total=len(epochs_dict), desc="Filtering Epoch Groups", position=0, leave=True)

#     for i in range(len(epochs_dict)):
#         final_epochs_dict = []
#         inner_pbar = tqdm(total=len(epochs_dict[i]), desc=f"Filtering Epochs in Group {i+1}", position=1, leave=False)
#         for j in range(len(epochs_dict[i])): 
#             temp_dict = []

#             inner_inner_pbar = tqdm(total=len(epochs_dict[i][j]), desc=f"Filtering Data in Epoch {j+1}", position=2, leave=False)

#             for k in range(len(epochs_dict[i][j])):
#                 df = pd.DataFrame(epochs_dict[i][j])           
#                 df_filtered = df.apply(lambda col: pr.apply_filter(col, lowcut, highcut, fs))
#                 temp_dict.append(df_filtered)

#                 inner_inner_pbar.update(1)

#             inner_inner_pbar.close()

#             final_epochs_dict.append(temp_dict)

#             inner_pbar.update(1)

#         inner_pbar.close()

#         filt_epoch_dict.append(final_epochs_dict)

#         outer_pbar.update(1)

#     outer_pbar.close()
    
    
            
#     for i, epoch in enumerate(filt_epoch_dict):
#         print(f"Subject_wise_Epochs {i} shape: {np.array(epoch).shape}")
        
#     final_epoch_label = [[i for _ in epochs] for i, epochs in enumerate(filt_epoch_dict)]
#     data_list = [item for sublist in filt_epoch_dict for item in sublist]
#     label_list = [item for sublist in final_epoch_label for item in sublist]
    
#     # return data_list,label_list


import csv 
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt

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

if __name__ == "__main__":
    path = Path(r"/media/sumit/dd6174bf-2d05-4a68-9324-d66b0a8e63762/EEG/deploy/TEST")
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
                df = pd.DataFrame(epochs_dict[i][j])
                df_tensor = torch.tensor(df.values, dtype=torch.float32, device=device)
                df_filtered_numpy = pr.apply_filter(df_tensor.cpu().numpy(), lowcut, highcut, fs)
                df_filtered = torch.tensor(df_filtered_numpy.copy(), dtype=torch.float32, device=device)
                small.append(df_filtered.to(device))
                inner_pbar.update(1)

            final_epochs_dict.append(small)

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
