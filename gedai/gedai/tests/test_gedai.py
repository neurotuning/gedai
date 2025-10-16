"""Test Gedai."""


from mne import Epochs, make_fixed_length_events
from mne.datasets import testing
from mne.io import read_raw_fif

from gedai import Gedai, logger, set_log_level

set_log_level("INFO")
logger.propagate = True


directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
# raw
raw = read_raw_fif(fname, preload=False)
raw.crop(0, 15)
raw.pick("eeg").load_data().apply_proj()

# epochs
events = make_fixed_length_events(raw, duration=1)
epochs_eeg = Epochs(raw, events, tmin=0, tmax=1, baseline=None, preload=True)


def test_gedai_fit_raw():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)

    gedai = Gedai(wavelet_level=5)
    gedai.fit_raw(raw)

def test_gedai_fit_epochs():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=5)
    gedai.fit_epochs(epochs_eeg)

def test_gedai_transform_raw():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)

    gedai = Gedai(wavelet_level=5)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)


def test_gedai_transform_epochs():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)
    gedai.transform_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=5)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)
