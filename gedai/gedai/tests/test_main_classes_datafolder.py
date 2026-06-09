"""Smoke tests for main GEDAI classes using bundled package data."""

from pathlib import Path

import mne
import numpy as np

from gedai import AdaptiveMultibandGedai, Gedai, MultibandGedai
from gedai.data import (
    get_contaminated_eeg_set_path,
    get_leadfield_cov_path,
    get_simulated_clean_eeg_set_path,
)


def _load_small_raw_segment(tmax=8.0):
    """Load a short EEG-only segment from the bundled sample dataset."""
    raw = mne.io.read_raw(get_contaminated_eeg_set_path(), preload=True)
    raw.pick("eeg")
    raw.crop(0, tmax)
    return raw


def test_bundled_data_paths_exist():
    """Bundled data helper paths should resolve to existing files."""
    assert Path(get_contaminated_eeg_set_path()).exists()
    assert Path(get_simulated_clean_eeg_set_path()).exists()
    assert Path(get_leadfield_cov_path()).exists()


def test_gedai_fit_transform_raw_with_bundled_data():
    """Gedai should fit and transform a short bundled raw segment."""
    raw = _load_small_raw_segment(tmax=8.0)

    model = Gedai()
    model.fit_raw(raw, duration=1.0, overlap=0.5, n_jobs=1)
    corrected = model.transform_raw(raw, overlap=0.5, n_jobs=1)

    assert model.fitted
    assert corrected.get_data().shape == raw.get_data().shape
    assert np.isfinite(corrected.get_data()).all()


def test_multiband_fit_transform_raw_with_bundled_data():
    """MultibandGedai should fit and transform bundled data."""
    raw = _load_small_raw_segment(tmax=10.0)

    model = MultibandGedai(wavelet_type="haar", wavelet_level=3)
    model.fit_raw(raw, duration=1.0, overlap=0.5, n_jobs=1)
    corrected = model.transform_raw(raw, overlap=0.5, n_jobs=1)

    assert model.fitted
    assert len(model._wavelets_fits) > 0
    assert corrected.get_data().shape == raw.get_data().shape
    assert np.isfinite(corrected.get_data()).all()


def test_adaptive_fit_transform_raw_with_bundled_data():
    """AdaptiveMultibandGedai should fit and transform bundled data."""
    raw = _load_small_raw_segment(tmax=8.0)

    model = AdaptiveMultibandGedai(
        wavelet_type="haar",
        wavelet_level=3,
        cycles_per_wavelet=4,
    )
    model.fit_raw(raw, overlap=0.5, n_jobs=1)
    corrected = model.transform_raw(raw, overlap=0.5, n_jobs=1)

    assert model.fitted
    assert len(model._wavelets_fits) == model.wavelet_level + 1
    non_ignored = [fit for fit in model._wavelets_fits if not fit["ignore"]]
    assert len(non_ignored) > 1
    assert len({fit["n_samples"] for fit in non_ignored}) > 1
    assert corrected.get_data().shape == raw.get_data().shape
    assert np.isfinite(corrected.get_data()).all()
