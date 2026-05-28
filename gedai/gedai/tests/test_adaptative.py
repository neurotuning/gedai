"""Tests for adaptive multiband GEDAI."""

import mne
import pytest
from mne.datasets import testing
from mne.io import read_raw_brainvision

from gedai.gedai.adaptative import AdaptativeMultibandGedai


data_path = testing.data_path(download=False)
raw_fname = data_path / "antio" / "CA_208" / "test_CA_208_start_stop.vhdr"
with pytest.warns():
    raw = read_raw_brainvision(raw_fname, eog=["EOG"], preload=True)
raw.pick_types(meg=False, eeg=True)
raw.drop_channels([ch_name for ch_name in raw.ch_names if "BIP" in ch_name])

wavelet_level = 5

def test_adaptative_fit_raw_():

    model = AdaptativeMultibandGedai(
        wavelet_type="haar",
        wavelet_level=wavelet_level,
        min_cycles_per_wavelet=3,
    )
    model.fit_raw(raw, overlap=0.5, reference_cov="leadfield", n_jobs=1)
    band_samples = [fit["samples"] for fit in model._wavelets_fits]
    assert len(band_samples) == model.wavelet_level + 1
    assert len(set(band_samples)) > 1
    assert max(band_samples) > min(band_samples)
