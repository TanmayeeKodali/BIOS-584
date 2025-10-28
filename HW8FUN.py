# HW8Fun.py
print("TANMAYEE KODALI")

# Functions for computing and plotting truncated EEG means and covariances
# Saves output figures into ./<subject_name>/ as required.

import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _ensure_output_dir(subject_name: str) -> str:
    """
    Ensure the subject-specific output directory exists under the current working directory.
    Returns the absolute path to the directory (e.g., ./K114).
    """
    out_dir = os.path.join(os.getcwd(), subject_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def produce_trunc_mean_cov(
    input_signal: np.ndarray,
    input_type: np.ndarray,
    E_val: int
) -> List[np.ndarray]:
    """
    Compute per-electrode sample means and covariances for target, non-target, and all trials.

    Parameters
    ----------
    input_signal : (N, F) array
        Truncated EEG signals (N trials × F features). F = E_val * T, where T = # time points per electrode.
    input_type : (N,) array
        Trial labels. Supports either {1, -1} or {1, 0} for {target, non-target}.
    E_val : int
        Number of electrodes, e.g., 16.

    Returns
    -------
    [tar_mean, ntar_mean, tar_cov, ntar_cov, all_cov]
      tar_mean : (E_val, T)
      ntar_mean : (E_val, T)
      tar_cov : (E_val, T, T)
      ntar_cov : (E_val, T, T)
      all_cov : (E_val, T, T)
    """
    # Determine label used for non-target (robust to 0 or -1)
    if np.any(input_type == 0):
        ntar_val = 0
    elif np.any(input_type == -1):
        ntar_val = -1
    else:
        raise ValueError("input_type must contain non-target labels as 0 or -1, and target as 1.")

    feature_len = input_signal.shape[1]
    length_per_electrode = feature_len // E_val  # e.g., 400 // 16 = 25

    # Separate trials by type
    signal_tar = input_signal[input_type == 1, :]
    signal_ntar = input_signal[input_type == ntar_val, :]

    # Prepare outputs
    tar_mean = np.zeros((E_val, length_per_electrode))
    ntar_mean = np.zeros((E_val, length_per_electrode))
    tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    # Compute per-electrode mean/cov
    for e in range(E_val):
        start = e * length_per_electrode
        end = (e + 1) * length_per_electrode

        tar_e = signal_tar[:, start:end]    # (n_tar, T)
        ntar_e = signal_ntar[:, start:end]  # (n_ntar, T)
        all_e = input_signal[:, start:end]  # (N, T)

        # Means across trials
        tar_mean[e, :] = np.mean(tar_e, axis=0)
        ntar_mean[e, :] = np.mean(ntar_e, axis=0)

        # Covariances (variables are columns → transpose first)
        # np.cov uses ddof=1 by default → sample covariance
        tar_cov[e, :, :] = np.cov(tar_e.T)
        ntar_cov[e, :, :] = np.cov(ntar_e.T)
        all_cov[e, :, :] = np.cov(all_e.T)

    return [tar_mean, ntar_mean, tar_cov, ntar_cov, all_cov]


def plot_trunc_mean(
    eeg_tar_mean: np.ndarray,
    eeg_ntar_mean: np.ndarray,
    subject_name: str,
    time_index: np.ndarray,
    E_val: int,
    electrode_name_ls: List[str],
    y_limit: np.ndarray = np.array([-5, 8]),
    fig_size: Tuple[int, int] = (12, 12)
) -> None:
    """
    Plot mean ERP signals (target vs non-target) for each electrode in a 4x4 grid,
    and SAVE the figure as ./<subject_name>/Mean.png
    """
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"Subject: {subject_name} - Mean ERP Signals", fontsize=16, fontweight="bold")

    for e in range(E_val):
        row, col = divmod(e, 4)
        ax = axes[row, col]

        ax.plot(time_index, eeg_tar_mean[e, :], "r-", label="Target", linewidth=2)
        ax.plot(time_index, eeg_ntar_mean[e, :], "b-", label="Non-Target", linewidth=2)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (muV)")
        ax.set_title(electrode_name_ls[e])
        ax.set_ylim(y_limit)
        ax.grid(True, alpha=0.3)

        # Put legend only on the first subplot to reduce clutter (optional)
        if e == 0:
            ax.legend()

    plt.tight_layout()
    # Save (inside this function as required)
    out_dir = _ensure_output_dir(subject_name)
    out_path = os.path.join(out_dir, "Mean.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trunc_cov(
    eeg_cov: np.ndarray,
    cov_type: str,
    time_index: np.ndarray,
    subject_name: str,
    E_val: int,
    electrode_name_ls: List[str],
    fig_size: Tuple[int, int] = (14, 12)
) -> None:
    """
    Plot covariance matrices for each electrode (contour plots) in a 4x4 grid,
    and SAVE the figure as:
      - Covariance_Target.png
      - Covariance_Non-Target.png
      - Covariance_All.png
    under ./<subject_name>/

    Parameters
    ----------
    eeg_cov : (E_val, T, T) array
    cov_type : {'Target', 'Non-Target', 'All'}
    """
    # Normalize cov_type for output naming
    ct = cov_type.strip()
    if ct not in {"Target", "Non-Target", "All"}:
        # Allow a few common variants
        ct = {"NonTarget": "Non-Target", "Non Target": "Non-Target"}.get(ct, ct)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"Subject: {subject_name} - {ct} Covariance", fontsize=16, fontweight="bold")

    X, Y = np.meshgrid(time_index, time_index)  # (T, T)

    for e in range(E_val):
        row, col = divmod(e, 4)
        ax = axes[row, col]
        contour = ax.contourf(X, Y, eeg_cov[e, :, :], levels=20, cmap="viridis")
        ax.invert_yaxis()  # time increases top->bottom as requested

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(electrode_name_ls[e])
        plt.colorbar(contour, ax=ax)

    plt.tight_layout()
    # Save (inside this function as required)
    out_dir = _ensure_output_dir(subject_name)
    file_name = f"Covariance_{ct}.png"
    out_path = os.path.join(out_dir, file_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)