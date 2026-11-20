"""Tests for MNE bad-channel handling, auto-detection, and interpolation."""

import logging

import mne
import numpy as np
import pytest
from mne import make_fixed_length_epochs
from scipy.linalg import eigh

from gedai import AdaptiveMultibandGedai, Gedai, MultibandGedai
from gedai.covariance.covariance import _ensure_cov, _pick_cov
from gedai.data import get_contaminated_eeg_set_path
from gedai.gedai._utils import (
    _check_average_reference,
    _check_reference_channel,
    _detect_flat_channels,
)


def _load_sample_eeg(n_channels=10, duration_s=4.0):
    """Load a small preloaded EEG Raw subset with valid 3D coordinates."""
    raw = mne.io.read_raw(
        get_contaminated_eeg_set_path(), preload=False, verbose=False
    )
    raw.drop_channels([ch for ch in raw.ch_names if "BIP" in ch])
    raw.pick(raw.ch_names[:n_channels])
    raw.crop(0.0, duration_s)
    raw.load_data()
    return raw


def _make_synthetic_bad_channel_eeg(
    n_channels=19,
    sfreq=100.0,
    n_times=800,
    seed=42,
    include_flat=True,
    include_noisy=True,
    with_montage=True,
):
    """Build synthetic EEG with leadfield-aligned signal plus flat/noisy channels."""
    rng = np.random.default_rng(seed)
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ][:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    if with_montage:
        montage_name = (
            "colin27_1020"
            if "colin27_1020" in mne.channels.get_builtin_montages()
            else "standard_1020"
        )
        info.set_montage(mne.channels.make_standard_montage(montage_name))

    cov = _pick_cov(_ensure_cov("leadfield"), ch_names)
    _, evecs = eigh(cov.data)
    top_modes = evecs[:, ::-1][:, :3]
    brain_sources = rng.standard_normal((3, n_times)) * 1e-5
    data = top_modes @ brain_sources

    if include_noisy:
        # Inject strong spatially localized outlier noise on the last channel
        ch_std = float(np.median(np.std(data, axis=1)))
        noisy_idx = n_channels - 1
        data[noisy_idx] += rng.standard_normal(n_times) * (10.0 * ch_std)

    if include_flat:
        flat_idx = n_channels - 2 if include_noisy else n_channels - 1
        data[flat_idx] = 1e-4  # constant DC offset (zero temporal diff)

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def test_check_average_reference_and_ref_channel_ignore_bads(caplog):
    """Average-reference and flat-ref checks should ignore info['bads']."""
    rng = np.random.default_rng(0)
    ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4"]
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
    data = rng.standard_normal((6, 300)) * 1e-5 + 2e-4
    data[0] = 0.0  # Flat channel marked bad should not count as reference channel
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.info["bads"] = ["Fp1"]

    with caplog.at_level(logging.WARNING, logger="gedai"):
        _check_reference_channel(raw)
    assert "does not contain a flat reference channel" in caplog.text

    raw_avg = raw.copy()
    raw_avg.set_eeg_reference("average", projection=False, verbose=False)
    assert _check_average_reference(raw_avg) is True

    epochs_avg = make_fixed_length_epochs(
        raw_avg, duration=1.0, preload=True, verbose=False
    )
    assert _check_average_reference(epochs_avg) is True

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gedai"):
        _check_reference_channel(epochs_avg)
    assert "does not contain a flat reference channel" in caplog.text


def test_gedai_on_bad_channel_exclude_and_ignore():
    """Gedai should exclude or ignore info['bads'] without mutating inputs."""
    raw = _load_sample_eeg(n_channels=8, duration_s=4.0)
    bad_ch = raw.ch_names[0]
    raw_avg = raw.copy()
    raw_avg.info["bads"] = [bad_ch]
    raw_avg.set_eeg_reference("average", projection=False, verbose=False)
    # Deliberately corrupt the bad channel offset so global mean != 0
    raw_avg._data[0] += 5e-4
    orig_bads = list(raw_avg.info["bads"])

    # Default: exclude
    g_excl = Gedai(engine="numpy")
    g_excl.fit_raw(raw_avg, duration=1.0, verbose=False)
    assert raw_avg.info["bads"] == orig_bads
    assert bad_ch not in g_excl.ch_names
    assert g_excl.bad_channels_ == [bad_ch]
    assert bad_ch in g_excl.summary

    clean_excl = g_excl.transform_raw(raw_avg, verbose=False)
    assert bad_ch not in clean_excl.ch_names
    assert clean_excl.info["bads"] == []
    assert g_excl.metrics_ is not None
    assert len(g_excl.metrics_["enova_per_channel"]) == len(g_excl.ch_names)

    # Ignore: clears bads on internal copy and re-references ALL channels together
    g_ign = Gedai(engine="numpy")
    g_ign.fit_raw(raw_avg, duration=1.0, on_bad_channel="ignore", verbose=False)
    assert raw_avg.info["bads"] == orig_bads
    assert g_ign.ch_names == raw_avg.ch_names
    assert g_ign.bad_channels_ == []

    clean_ign = g_ign.transform_raw(raw_avg, verbose=False)
    assert clean_ign.ch_names == raw_avg.ch_names
    assert np.allclose(clean_ign.get_data().mean(axis=0), 0.0, atol=1e-6)

    # Epochs fit/transform parity
    epochs_avg = make_fixed_length_epochs(
        raw_avg, duration=1.0, preload=True, verbose=False
    )
    g_ep = Gedai(engine="numpy")
    g_ep.fit_epochs(epochs_avg, on_bad_channel="exclude", verbose=False)
    assert bad_ch not in g_ep.ch_names
    clean_ep = g_ep.transform_epochs(epochs_avg, verbose=False)
    assert clean_ep.ch_names == g_ep.ch_names


def test_explicit_picks_still_excludes_bads_and_validation_errors():
    """Explicit picks must still exclude info['bads']; invalid inputs raise."""
    raw = _load_sample_eeg(n_channels=6, duration_s=3.0)
    bad_ch = raw.ch_names[0]
    raw.info["bads"] = [bad_ch]

    g = Gedai(engine="numpy")
    g.fit_raw(raw, picks=raw.ch_names[:4], on_bad_channel="exclude", verbose=False)
    assert g.ch_names == raw.ch_names[1:4]
    assert g.bad_channels_ == [bad_ch]

    with pytest.raises(ValueError, match="No channels remain"):
        Gedai(engine="numpy").fit_raw(raw, picks=[bad_ch], verbose=False)

    with pytest.raises(ValueError, match="Invalid value for the 'on_bad_channel'"):
        Gedai(engine="numpy").fit_raw(raw, on_bad_channel="drop", verbose=False)

    # Transform-time bad channel mismatch under 'exclude'
    raw_other = raw.copy()
    raw_other.info["bads"] = [raw.ch_names[1]]
    with pytest.raises(ValueError, match="marked as bad in the input"):
        g.transform_raw(raw_other, verbose=False)

    # Under 'ignore', transform-time bad channel is cleared and processed cleanly
    g_ign = Gedai(engine="numpy")
    g_ign.fit_raw(raw, picks=raw.ch_names[:4], on_bad_channel="ignore", verbose=False)
    out_ign = g_ign.transform_raw(raw_other, verbose=False)
    assert out_ign.ch_names == raw.ch_names[:4]


def test_multiband_and_adaptive_on_bad_channel():
    """MultibandGedai and AdaptiveMultibandGedai respect on_bad_channel."""
    raw = _load_sample_eeg(n_channels=8, duration_s=4.0)
    bad_ch = raw.ch_names[1]
    raw.info["bads"] = [bad_ch]
    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)

    mb_ep = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_ep.fit_epochs(epochs, on_bad_channel="exclude", n_jobs=1, verbose=False)
    assert bad_ch not in mb_ep.ch_names
    assert mb_ep.bad_channels_ == [bad_ch]
    clean_mb_ep = mb_ep.transform_epochs(epochs, n_jobs=1, verbose=False)
    assert clean_mb_ep.ch_names == mb_ep.ch_names
    assert np.any(mb_ep.metrics_["enova_per_channel"] > 0)

    mb_raw = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_raw.fit_raw(raw, duration=1.0, on_bad_channel="exclude", n_jobs=1, verbose=False)
    assert bad_ch not in mb_raw.ch_names
    clean_mb_raw = mb_raw.transform_raw(raw, n_jobs=1, verbose=False)
    assert clean_mb_raw.ch_names == mb_raw.ch_names
    assert mb_raw.metrics_ is not None
    assert len(mb_raw.metrics_["enova_per_channel"]) == len(mb_raw.ch_names)

    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, on_bad_channel="exclude", n_jobs=1, verbose=False)
    assert bad_ch not in amb.ch_names
    assert amb.bad_channels_ == [bad_ch]
    clean_cached = amb.transform_raw(raw, n_jobs=1, verbose=False)
    amb.clear_cache()
    clean_uncached = amb.transform_raw(raw, n_jobs=1, verbose=False)
    np.testing.assert_allclose(
        clean_cached.get_data(), clean_uncached.get_data(), rtol=1e-10, atol=1e-10
    )


def test_detect_flat_channels_unit_and_dc_invariant():
    """Flat-channel detection should work across scales, DC offsets, and Epochs."""
    rng = np.random.default_rng(7)
    ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4"]
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
    base = rng.standard_normal((6, 400)) * 1e-6
    base[1] = 0.0  # zero flat
    base[4] = 3.5e-4  # non-zero DC flat

    raw_v = mne.io.RawArray(base.copy(), info, verbose=False)
    raw_uv = mne.io.RawArray(base.copy() * 1e6, info, verbose=False)
    assert _detect_flat_channels(raw_v) == ["Fp2", "C3"]
    assert _detect_flat_channels(raw_uv) == ["Fp2", "C3"]

    epochs = make_fixed_length_epochs(raw_v, duration=1.0, preload=True, verbose=False)
    assert _detect_flat_channels(epochs) == ["Fp2", "C3"]


@pytest.mark.parametrize("threshold", [0.95, "auto"])
def test_two_pass_bad_channel_detection_gedai(threshold):
    """Gedai should auto-detect flat and noisy channels on Raw and Epochs."""
    raw = _load_sample_eeg(n_channels=27, duration_s=6.0)
    rng = np.random.default_rng(42)
    ch_std = float(np.median(np.std(raw.get_data(), axis=1)))
    raw._data[-2] = 1e-4
    raw._data[-1] += rng.standard_normal(raw.n_times) * (10.0 * ch_std)
    flat_ch = raw.ch_names[-2]
    noisy_ch = raw.ch_names[-1]

    g_raw = Gedai(engine="numpy")
    g_raw.fit_raw(raw, duration=1.0, bad_channel_threshold=threshold, verbose=False)
    assert g_raw.flat_channels_ == [flat_ch]
    assert noisy_ch in g_raw.noisy_channels_
    assert flat_ch not in g_raw.ch_names
    assert noisy_ch not in g_raw.ch_names
    assert {flat_ch, noisy_ch}.issubset(g_raw.bad_channels_)
    assert g_raw.enova_per_channel_[flat_ch] == float("inf")
    assert g_raw.enova_per_channel_[noisy_ch] > 0.95

    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)
    g_ep = Gedai(engine="numpy")
    g_ep.fit_epochs(epochs, bad_channel_threshold=threshold, verbose=False)
    assert g_ep.flat_channels_ == [flat_ch]
    assert noisy_ch in g_ep.noisy_channels_
    assert flat_ch not in g_ep.ch_names
    assert noisy_ch not in g_ep.ch_names


def test_two_pass_multiband_adaptive_and_cache_parity():
    """Two-pass bad-channel detection works in Multiband and Adaptive models."""
    raw = _load_sample_eeg(n_channels=27, duration_s=6.0)
    rng = np.random.default_rng(42)
    ch_std = float(np.median(np.std(raw.get_data(), axis=1)))
    raw._data[-2] = 1e-4
    raw._data[-1] += rng.standard_normal(raw.n_times) * (10.0 * ch_std)
    flat_ch = raw.ch_names[-2]
    noisy_ch = raw.ch_names[-1]
    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)

    mb = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb.fit_epochs(epochs, bad_channel_threshold=0.95, n_jobs=1, verbose=False)
    assert mb.flat_channels_ == [flat_ch]
    assert noisy_ch in mb.noisy_channels_
    assert flat_ch not in mb.ch_names
    assert noisy_ch not in mb.ch_names

    mb_raw = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_raw.fit_raw(
        raw, duration=1.0, bad_channel_threshold=0.95, n_jobs=1, verbose=False
    )
    assert mb_raw.flat_channels_ == [flat_ch]
    assert noisy_ch in mb_raw.noisy_channels_

    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, bad_channel_threshold=0.95, n_jobs=1, verbose=False)
    assert amb.flat_channels_ == [flat_ch]
    assert noisy_ch in amb.noisy_channels_
    assert amb.bad_channels_ == amb.detected_bad_channels_

    clean_cached = amb.transform_raw(raw, n_jobs=1, verbose=False)
    amb.clear_cache()
    clean_uncached = amb.transform_raw(raw, n_jobs=1, verbose=False)
    np.testing.assert_allclose(
        clean_cached.get_data(), clean_uncached.get_data(), rtol=1e-10, atol=1e-10
    )


def test_two_pass_skips_pass2_when_clean_and_combines_bads(monkeypatch):
    """When Pass 1 finds no bad channels, Pass 2 must be skipped."""
    raw_clean = _make_synthetic_bad_channel_eeg(
        n_channels=10, sfreq=100.0, n_times=600, include_flat=False, include_noisy=False
    )
    calls = []
    orig_fit_epochs = Gedai.fit_epochs

    def spy_fit_epochs(self, *args, **kwargs):
        calls.append(kwargs.get("picks"))
        return orig_fit_epochs(self, *args, **kwargs)

    monkeypatch.setattr(Gedai, "fit_epochs", spy_fit_epochs)

    epochs = make_fixed_length_epochs(
        raw_clean, duration=1.0, preload=True, verbose=False
    )
    model = Gedai(engine="numpy")
    model.fit_epochs(epochs, bad_channel_threshold=0.95, verbose=False)
    # 1 outer call + 1 Pass 1 call (no Pass 2 call)
    assert len(calls) == 2
    assert model.fitted is True
    assert model.metrics_ is None
    assert model.bad_channels_ == []
    assert model.detected_bad_channels_ == []

    # Union of pre-existing info['bads'] and auto-detected bad channels
    raw_comb = _load_sample_eeg(n_channels=27, duration_s=6.0)
    rng = np.random.default_rng(42)
    ch_std = float(np.median(np.std(raw_comb.get_data(), axis=1)))
    raw_comb._data[-2] = 1e-4
    raw_comb._data[-1] += rng.standard_normal(raw_comb.n_times) * (10.0 * ch_std)
    pre_bad = raw_comb.ch_names[0]
    raw_comb.info["bads"] = [pre_bad]
    g_comb = Gedai(engine="numpy")
    g_comb.fit_raw(
        raw_comb,
        duration=1.0,
        on_bad_channel="exclude",
        bad_channel_threshold=0.95,
        verbose=False,
    )
    assert pre_bad not in g_comb.detected_bad_channels_
    assert pre_bad in g_comb.bad_channels_
    assert g_comb.flat_channels_[0] in g_comb.bad_channels_


def test_bad_channel_threshold_validation_and_guards():
    """Invalid thresholds or fewer than 2 remaining channels should raise."""
    raw = _make_synthetic_bad_channel_eeg(
        n_channels=6, sfreq=100.0, n_times=400, include_flat=False, include_noisy=False
    )
    g_inf = Gedai(engine="numpy")
    g_inf.fit_raw(raw, bad_channel_threshold=np.inf, verbose=False)
    assert g_inf.enova_per_channel_ is None
    assert g_inf.bad_channels_ == []

    for bad_val in (0, -0.5, np.nan, "bad"):
        with pytest.raises(ValueError, match="bad_channel_threshold"):
            Gedai(engine="numpy").fit_raw(
                raw, bad_channel_threshold=bad_val, verbose=False
            )

    for bad_type in (True, [0.95]):
        with pytest.raises(TypeError, match="bad_channel_threshold"):
            Gedai(engine="numpy").fit_raw(
                raw, bad_channel_threshold=bad_type, verbose=False
            )

    # Fewer than 2 non-flat channels should raise ValueError
    raw_all_flat = raw.copy()
    raw_all_flat._data[:-1] = 0.0
    with pytest.raises(ValueError, match="At least 2 non-flat channels"):
        Gedai(engine="numpy").fit_raw(
            raw_all_flat, bad_channel_threshold=0.95, verbose=False
        )


def test_interpolate_bads_raw_and_epochs():
    """interpolate_bads=True restores montage order and EEG average reference."""
    raw = _load_sample_eeg(n_channels=27, duration_s=6.0)
    rng = np.random.default_rng(42)
    ch_std = float(np.median(np.std(raw.get_data(), axis=1)))
    raw._data[-2] = 1e-4
    raw._data[-1] += rng.standard_normal(raw.n_times) * (10.0 * ch_std)
    orig_chs = list(raw.ch_names)
    raw.info["bads"] = [orig_chs[0]]

    # 1. Gedai Raw
    g = Gedai(engine="numpy")
    g.fit_raw(raw, duration=1.0, bad_channel_threshold=0.95, verbose=False)
    assert len(g.ch_names) < len(orig_chs)
    out_raw = g.transform_raw(raw, interpolate_bads=True, verbose=False)
    assert out_raw.ch_names == orig_chs
    assert out_raw.info["bads"] == []
    assert np.isfinite(out_raw.get_data()).all()
    assert np.all(np.std(out_raw.get_data(), axis=1) > 0)
    assert np.allclose(out_raw.get_data().mean(axis=0), 0.0, atol=1e-6)

    # 2. MultibandGedai Epochs
    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)
    mb = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb.fit_epochs(epochs, bad_channel_threshold=0.95, n_jobs=1, verbose=False)
    out_ep = mb.transform_epochs(epochs, n_jobs=1, interpolate_bads=True, verbose=False)
    assert out_ep.ch_names == orig_chs
    assert out_ep.info["bads"] == []
    assert np.isfinite(out_ep.get_data()).all()
    assert np.allclose(out_ep.get_data().mean(axis=1), 0.0, atol=1e-6)

    # 3. AdaptiveMultibandGedai Raw
    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, bad_channel_threshold=0.95, n_jobs=1, verbose=False)
    out_amb = amb.transform_raw(raw, n_jobs=1, interpolate_bads=True, verbose=False)
    assert out_amb.ch_names == orig_chs
    assert out_amb.info["bads"] == []
    assert np.allclose(out_amb.get_data().mean(axis=0), 0.0, atol=1e-6)

    # 4. Missing 3D channel coordinates raises informative ValueError
    raw_noloc = _make_synthetic_bad_channel_eeg(
        n_channels=6,
        sfreq=100.0,
        n_times=400,
        include_flat=False,
        include_noisy=False,
        with_montage=False,
    )
    raw_noloc.info["bads"] = [raw_noloc.ch_names[0]]
    cov = mne.Covariance(
        np.eye(len(raw_noloc.ch_names)), raw_noloc.ch_names, [], [], nfree=1
    )
    g_noloc = Gedai(engine="numpy")
    g_noloc.fit_raw(raw_noloc, reference_cov=cov, verbose=False)
    with pytest.raises(ValueError, match="valid 3D channel locations"):
        g_noloc.transform_raw(raw_noloc, interpolate_bads=True, verbose=False)
