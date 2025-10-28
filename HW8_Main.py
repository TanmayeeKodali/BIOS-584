# HW8_main.py
print("TANMAYEE KODALI")

# Main script: set globals, load data, call functions from HW8Fun.py

import os
import numpy as np
import scipy.io as sio

from HW8FUN import (
    produce_trunc_mean_cov,
    plot_trunc_mean,
    plot_trunc_cov,
)

# -----------------------------
# Global constants (copy here)
# -----------------------------
bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4',
                     'T7', 'C3', 'Cz', 'C4', 'T8',
                     'CP3', 'CP4', 'P3', 'Pz',
                     'P4', 'PO7', 'PO8', 'Oz']
subject_name = 'K114'
session_name = '001_BCI_TRN'
time_index = np.linspace(0, 800, 25)  # 25 time points across 0–800 ms

# Parent directories (current working dir + data)
parent_dir = os.getcwd()
parent_data_dir = f"{parent_dir}/data"

# Construct full data file path:
#   <subject>_<session>_Truncated_Data_<bp_low>_<bp_upp>.mat
data_file = f"{parent_data_dir}/{subject_name}_{session_name}_Truncated_Data_{bp_low}_{bp_upp}.mat"

# -----------------------------
# Load .mat file
# -----------------------------
if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Data file not found:\n  {data_file}\n"
        f"Please place the .mat file in the 'data' folder."
    )

eeg_trunc_obj = sio.loadmat(data_file)

# Extract arrays
eeg_trunc_signal = eeg_trunc_obj['Signal']              # shape (N, 400)
eeg_trunc_type = np.squeeze(eeg_trunc_obj['Type'])      # shape (N,)

# -----------------------------
# Compute means & covariances
# -----------------------------
(signal_tar_mean,
 signal_ntar_mean,
 signal_tar_cov,
 signal_ntar_cov,
 signal_all_cov) = produce_trunc_mean_cov(
    eeg_trunc_signal, eeg_trunc_type, electrode_num
)

# -----------------------------
# Plot & SAVE figures (inside HW8Fun)
# -----------------------------
# Mean figure → ./K114/Mean.png
plot_trunc_mean(
    eeg_tar_mean=signal_tar_mean,
    eeg_ntar_mean=signal_ntar_mean,
    subject_name=subject_name,
    time_index=time_index,
    E_val=electrode_num,
    electrode_name_ls=electrode_name_ls
)

# Covariance figures → ./K114/Covariance_Target.png, etc.
plot_trunc_cov(
    eeg_cov=signal_tar_cov,
    cov_type='Target',
    time_index=time_index,
    subject_name=subject_name,
    E_val=electrode_num,
    electrode_name_ls=electrode_name_ls
)

plot_trunc_cov(
    eeg_cov=signal_ntar_cov,
    cov_type='Non-Target',
    time_index=time_index,
    subject_name=subject_name,
    E_val=electrode_num,
    electrode_name_ls=electrode_name_ls
)

plot_trunc_cov(
    eeg_cov=signal_all_cov,
    cov_type='All',
    time_index=time_index,
    subject_name=subject_name,
    E_val=electrode_num,
    electrode_name_ls=electrode_name_ls
)

print(f"Done. Figures saved under ./{subject_name}/ :")
print(" - Mean.png")
print(" - Covariance_Target.png")
print(" - Covariance_Non-Target.png")
print(" - Covariance_All.png")