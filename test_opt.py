import numpy as np
import mne
from gedai import Gedai
from gedai.sensai.sensai import _sensai_gridsearch_fast, _sensai_optimize_fast, subspace_angles, _sensai_to_eigen
from scipy.linalg import eigh
from gedai.gedai.covariances import _compute_refcov
import os

data_path = r'c:\Users\Utilisateur\Documents\GitHub\gedai\gedai\data\CAUEEG.set'
raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose='error')
raw.set_eeg_reference("average", ch_type="eeg")
gedai = Gedai(wavelet_level=0, epoch_size_in_cycles=12)

# Extract epoch covariance
window_size = int(raw.info["sfreq"] * 1.0)
n_times = raw.get_data().shape[1]
starts = np.arange(0, n_times - window_size, window_size) # 0-overlap for speed comparison
segments = [raw.get_data()[:, start:start+window_size] for start in starts]

# Compute Eigenvalues
epochs_evals = []
epochs_evecs = []
for seg in segments:
    cov = np.cov(seg)
    evals, evecs = eigh(cov)
    # Return descending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    epochs_evals.append(evals)
    epochs_evecs.append(evecs)
epochs_evals = np.array(epochs_evals)
epochs_evecs = np.array(epochs_evecs)

# Compute Reference
ref_cov = _compute_refcov(raw, os.path.join(os.path.dirname(__file__), 'gedai/data/fsavLEADFIELD_4_GEDAI.mat'))[0]
ref_cov_reg = 0.95 * ref_cov + 0.05 * (np.trace(ref_cov)/19) * np.eye(19)
evals_ref, evecs_ref = eigh(ref_cov_reg)
evecs_ref = evecs_ref[:, ::-1][:, :3]

eigen_thresholds = [_sensai_to_eigen(val, epochs_evals, 98) for val in np.arange(0, 12, 0.25)]

best_t_grid, runs_grid = _sensai_gridsearch_fast(
    epochs_evals, epochs_evecs, ref_cov_reg, evecs_ref, 3, 6.0, eigen_thresholds
)

best_e_opt, runs_opt = _sensai_optimize_fast(
    epochs_evals, epochs_evecs, ref_cov_reg, evecs_ref, 3, 6.0, (0, 12), 98
)

print(f"GRID  -> Eigen={best_t_grid:.2e} (Sensai Thresh aprox {np.log10(best_t_grid):.2f})  Score={max([r[1] for r in runs_grid]):.4f}")
print(f"BRENT -> Eigen={best_e_opt:.2e} (Sensai Thresh aprox {np.log10(best_e_opt):.2f})  Score={max([r[1] for r in runs_opt]):.4f}")

import matplotlib.pyplot as plt
sensais = np.arange(0, 12, 0.25)
scores = [r[1] for r in runs_grid]
for i in range(len(sensais)):
    print(f"Grid x={sensais[i]:.2f}, Score={scores[i]:.4f}, Sig={runs_grid[i][2]:.4f}, Noi={runs_grid[i][3]:.4f}")
