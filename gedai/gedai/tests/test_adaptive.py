"""Tests for Adaptive multiband GEDAI."""

import mne
from mne import make_fixed_length_epochs

from gedai.data import get_contaminated_eeg_set_path
from gedai.gedai.adaptive import AdaptiveMultibandGedai

raw_fname = get_contaminated_eeg_set_path()
raw = mne.io.read_raw(raw_fname, preload=True)
epochs_eeg = make_fixed_length_epochs(raw, duration=1.0, overlap=0)
wavelet_level = 8


def test_adaptive_fit_raw_():
    """Fit the adaptive model on the bundled raw sample."""
    model = AdaptiveMultibandGedai(
        wavelet_type="haar",
        wavelet_level=wavelet_level,
        cycles_per_wavelet=4,
    )
    model.fit_raw(raw, overlap=0.5, reference_cov="leadfield", n_jobs=1)
    band_samples = [fit["n_samples"] for fit in model._wavelets_fits]
    assert len(band_samples) == model.wavelet_level + 1
    assert len(set(band_samples)) > 1
