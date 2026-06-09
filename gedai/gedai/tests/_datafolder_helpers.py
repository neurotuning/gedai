"""Helpers for tests that use bundled package data."""

from pathlib import Path

import mne

from gedai.data import (
    get_contaminated_eeg_set_path,
    get_leadfield_cov_path,
    get_simulated_clean_eeg_set_path,
)


def assert_bundled_data_paths_exist():
    """Assert all expected bundled data files are available."""
    assert Path(get_contaminated_eeg_set_path()).exists()
    assert Path(get_simulated_clean_eeg_set_path()).exists()
    assert Path(get_leadfield_cov_path()).exists()


def load_small_raw_segment(tmax=8.0):
    """Load a short EEG-only segment from bundled sample data."""
    raw = mne.io.read_raw(get_contaminated_eeg_set_path(), preload=True)
    raw.pick("eeg")
    raw.crop(0, tmax)
    return raw
