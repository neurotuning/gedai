"""Tests for Adaptive multiband GEDAI."""

import mne
import numpy as np
import pytest

import gedai.gedai.adaptive as adaptive_module
from gedai.data import get_contaminated_eeg_set_path
from gedai.gedai.adaptive import AdaptiveMultibandGedai

raw_fname = get_contaminated_eeg_set_path()
raw_eeg = mne.io.read_raw(raw_fname, preload=True)
raw_eeg.drop_channels([ch_name for ch_name in raw_eeg.ch_names if "BIP" in ch_name])

wavelet_level = 8


def test_gedai_multiband_adaptive_fit_transform_raw():
    """Test Gedai transform on raw data."""
    model = AdaptiveMultibandGedai()
    model.fit_raw(raw_eeg)
    band_samples = [fit["n_samples"] for fit in model._wavelets_fits]
    assert len(band_samples) == model.wavelet_level + 1
    assert len(set(band_samples)) > 1

    transformed_raw = model.transform_raw(raw_eeg)
    assert transformed_raw.info["ch_names"] == raw_eeg.info["ch_names"]
    assert transformed_raw.info["sfreq"] == raw_eeg.info["sfreq"]
    assert raw_eeg.annotations == transformed_raw.annotations


def test_gedai_multiband_adaptive_raw_picks():
    """Test Gedai fit on raw data."""
    model = AdaptiveMultibandGedai()
    model.fit_raw(raw_eeg, picks="all")
    assert model.ch_names == raw_eeg.ch_names

    model = AdaptiveMultibandGedai()
    model.fit_raw(raw_eeg, picks="data")
    assert model.ch_names == raw_eeg.ch_names

    model = AdaptiveMultibandGedai()
    model.fit_raw(raw_eeg, picks=raw_eeg.ch_names[:10])
    assert model.ch_names == raw_eeg.ch_names[:10]

    raw_transformed = model.transform_raw(raw_eeg)
    assert raw_transformed.ch_names == raw_eeg.ch_names[:10]

    raw_test = raw_eeg.copy()
    raw_test.load_data()
    raw_test.pick_channels(raw_eeg.ch_names[:5])
    with pytest.raises(
        ValueError,
        match="The following channels are missing in the input inst but were present",
    ):
        model.transform_raw(raw_test)


def test_gedai_multiband_adaptive_auto_level_and_metrics():
    """Test adaptive multiband with auto level and metrics."""
    model = AdaptiveMultibandGedai(wavelet_type="haar", wavelet_level="auto")
    model.fit_raw(raw_eeg, picks=raw_eeg.ch_names[:6], n_jobs=1)
    assert model._actual_wavelet_level is not None
    assert model._actual_wavelet_level >= 6
    assert model.fit_metrics_ is not None
    assert "sensai_score" in model.fit_metrics_
    assert isinstance(model.summary, str)
    assert model.fit_summary() == model.summary

    transformed_raw = model.transform_raw(raw_eeg, n_jobs=1)
    assert transformed_raw.get_data().shape[0] == 6


def test_gedai_multiband_adaptive_broadband_pass():
    """Test adaptive multiband with broadband pass."""
    model = AdaptiveMultibandGedai(
        wavelet_type="haar", wavelet_level=4, broadband_pass=True
    )
    model.fit_raw(raw_eeg, picks=raw_eeg.ch_names[:6], n_jobs=1)
    assert model._broadband_model is not None
    assert model.fit_metrics_ is not None

    transformed_raw = model.transform_raw(raw_eeg, n_jobs=1)
    assert transformed_raw.get_data().shape[0] == 6


def test_adaptive_fit_transform_cache_compatibility():
    """Test that transform_raw reuses cached fit_raw outputs with numerical parity."""
    model = AdaptiveMultibandGedai(
        wavelet_type="haar", wavelet_level=2, broadband_pass=True
    )
    raw_sub = raw_eeg.copy().pick(raw_eeg.ch_names[:6])
    model.fit_raw(raw_sub, n_jobs=1)

    assert hasattr(model, "_cached_broadband_data")
    assert model._cached_broadband_data is not None
    assert model._fitted_raw_id == id(raw_sub)

    # Transform with cache
    clean_cached = model.transform_raw(raw_sub, n_jobs=1)

    # Clear cache and transform again to verify numerical equivalence
    model.clear_cache()
    assert model._cached_broadband_data is None
    clean_uncached = model.transform_raw(raw_sub, n_jobs=1)

    np.testing.assert_allclose(
        clean_cached.get_data(), clean_uncached.get_data(), rtol=1e-10, atol=1e-10
    )


def test_adaptive_fit_raw_engine_override_reaches_prefilter(monkeypatch):
    """Explicit fit-time engine overrides should reach adaptive preprocessing."""
    calls = []
    original = adaptive_module._apply_wavelet_highpass_prefilter

    def spy(data, sfreq, lowcut_hz=0.5, engine="auto"):
        calls.append(engine)
        return original(data, sfreq, lowcut_hz=lowcut_hz, engine=engine)

    monkeypatch.setattr(adaptive_module, "_apply_wavelet_highpass_prefilter", spy)

    model = AdaptiveMultibandGedai(
        wavelet_type="haar", wavelet_level=1, broadband_pass=True
    )
    model.fit_raw(
        raw_eeg.copy().pick(raw_eeg.ch_names[:4]),
        picks="all",
        n_jobs=1,
        verbose=False,
        engine="numpy",
    )

    assert calls
    assert all(engine == "numpy" for engine in calls)


def test_adaptive_wavelet_low_cutoff_modes(monkeypatch):
    """Test 0.1 Hz default, 0.0 Hz (None) zero-leak mode, and 0.5 Hz drift mode."""
    calls = []
    original = adaptive_module._apply_wavelet_highpass_prefilter

    def spy(data, sfreq, lowcut_hz=0.1, engine="auto"):
        calls.append(lowcut_hz)
        return original(data, sfreq, lowcut_hz=lowcut_hz, engine=engine)

    monkeypatch.setattr(adaptive_module, "_apply_wavelet_highpass_prefilter", spy)

    names = raw_eeg.ch_names[:4]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(names), 1200)) * 1e-6
    raw_small = mne.io.RawArray(
        data.copy(), mne.create_info(names, 200.0, "eeg"), verbose=False
    )
    cov = mne.Covariance(np.eye(len(names)) * 1e-12, names, [], [], nfree=1)

    # 1. Default 0.1 Hz: single pre-filter call at 0.1 Hz, all spectral bands retained
    calls.clear()
    model_01 = AdaptiveMultibandGedai(wavelet_level=4, broadband_pass=True)
    model_01.fit_raw(raw_small, reference_cov=cov, n_jobs=1, verbose=False)
    assert model_01._wavelet_low_cutoff == 0.1
    assert calls == [0.1]
    assert not any(wf["ignore"] for wf in model_01._wavelets_fits)

    # 2. 0.0 Hz / None: zero pre-filter calls, all spectral bands retained
    calls.clear()
    model_00 = AdaptiveMultibandGedai(wavelet_level=4, broadband_pass=True)
    model_00.fit_raw(
        raw_small,
        reference_cov=cov,
        wavelet_low_cutoff=None,
        n_jobs=1,
        verbose=False,
    )
    assert model_00._wavelet_low_cutoff == 0.0
    assert calls == []
    assert not any(wf["ignore"] for wf in model_00._wavelets_fits)

    # 3. >= 0.5 Hz: single pre-filter call at 0.5 Hz, lowest band (0-0.39 Hz) ignored
    calls.clear()
    model_05 = AdaptiveMultibandGedai(wavelet_level=8, broadband_pass=True)
    model_05.fit_raw(
        raw_small,
        reference_cov=cov,
        wavelet_low_cutoff=0.5,
        n_jobs=1,
        verbose=False,
    )
    assert model_05._wavelet_low_cutoff == 0.5
    assert calls == [0.5]
    assert any(wf["ignore"] for wf in model_05._wavelets_fits)

