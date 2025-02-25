import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path

def apply_mne_filter(data, sfreq, l_freq=20, h_freq=63):
    ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw.get_data()

def plot_eeg_colored(file_path, duration=15, sfreq=128):
    df = pd.read_csv(file_path)
    data = df.to_numpy().T
    print(data[0])
    
    num_samples = duration * sfreq
    time_axis = np.arange(0, duration, 1/sfreq)
    
    filtered_data = apply_mne_filter(data, sfreq)
    
    palette = sns.color_palette("husl", data.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    print(data[0])
    print(filtered_data[0])

    for ch, color in enumerate(palette):
        axes[0].plot(time_axis, data[ch, :num_samples], color=color, alpha=0.8, label=f"Ch {ch+1}")
    axes[0].set_title("Raw EEG Signal")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(ncol=4, fontsize=8, loc='upper right', frameon=True)

    for ch, color in enumerate(palette):
        axes[1].plot(time_axis, filtered_data[ch, :num_samples], color=color, alpha=0.8, label=f"Ch {ch+1}")
    axes[1].set_title("Filtered EEG Signal (4-25 Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].legend(ncol=4, fontsize=8, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()

sample_csv = Path("/media/sumit/dd6174bf-2d05-4a68-9324-d66b0a8e63762/EEG/TEST/1/data1.csv")
plot_eeg_colored(sample_csv)    