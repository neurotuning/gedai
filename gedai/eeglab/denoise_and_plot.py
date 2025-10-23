

import mne
import time
from gedai import Gedai
from gedai.viz.compare import plot_mne_style_overlay_interactive

# Parameters
n_jobs = 1
duration = 1  # seconds

# Load the data
raw = mne.io.read_raw_eeglab("mixed_NOISE_EOG_EMG.set", preload=True)

# Start timer
start_time = time.time()

# Denoise the data using the provided snippet
raw.set_eeg_reference("average", projection=False)

gedai = Gedai()
gedai.fit_raw(raw, noise_multiplier=6., duration=duration, n_jobs=n_jobs)
raw_corrected = gedai.transform_raw(raw, duration=duration)


gedai_spectral = Gedai(wavelet_type='haar', wavelet_level=6, low_cutoff=2.0)
gedai_spectral.fit_raw(raw_corrected, noise_multiplier=3., duration=duration, n_jobs=n_jobs)
raw_corrected_final = gedai_spectral.transform_raw(raw_corrected, duration=duration)

# End timer and print elapsed time
end_time = time.time()
print(f"Denoising process took: {end_time - start_time:.2f} seconds")

# Visualize the results
plot_mne_style_overlay_interactive(raw, raw_corrected_final, title="Original vs. Denoised")
