import pandas as pd
import numpy as np
from mne.filter import filter_data

# 1. Load the CSV data into a Pandas DataFrame
df = pd.read_csv(r"C:\Users\pawasthi\Desktop\TEST")

# 2. Convert DataFrame to a NumPy array
#    Assuming each column is a channel and each row is a time sample,
#    we transpose the array to match MNE's shape requirement: (n_channels, n_samples)
data = df.values.T

# 3. Define your sampling frequency (Hz)
sfreq = 128.0  # e.g., 250 Hz

# 4. Apply a bandpass filter (1–40 Hz in this example)
filtered_data = filter_data(data,
                            sfreq=sfreq,
                            l_freq=20.0,  # lower cutoff
                            h_freq=60.0, # upper cutoff
                            method='fir')  # or 'iir'

