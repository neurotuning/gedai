"""Integration tests for MATLAB-ported Tier 1 & Tier 2 GEDAI features."""

import mne
import numpy as np
import pytest

from gedai import AdaptiveMultibandGedai, Gedai, MultibandGedai
from gedai.covariance.covariance import _load_template_leadfield, _match_template_labels
from gedai.data import get_contaminated_eeg_set_path


def _sample_raw(n_channels: int = 12, duration_s: float = 8.0) -> mne.io.BaseRaw:
    raw = mne.io.read_raw(
        get_contaminated_eeg_set_path(), preload=False, verbose=False
    )
    raw.drop_channels([ch for ch in raw.ch_names if "BIP" in ch])
    raw.pick(raw.ch_names[:n_channels])
    raw.crop(0.0, duration_s)
    raw.load_data()
    return raw


def test_non_1020_auto_dipole_projection_and_bids_labels():
    """Non-10-20 montage auto-projects via 3D coords and BIDS names match exact."""
    raw = _sample_raw(n_channels=10, duration_s=4.0)

    # 1. Explicit BIDS/EDF channel renaming produces identical output
    bids_map = {ch: f"EEG {ch}-REF" for ch in raw.ch_names}
    raw_bids = raw.copy().rename_channels(bids_map)

    g_std = Gedai(engine="numpy")
    g_std.fit_raw(raw, duration=1.0, verbose=False)
    out_std = g_std.transform_raw(raw, verbose=False).get_data()

    g_bids = Gedai(engine="numpy")
    g_bids.fit_raw(raw_bids, duration=1.0, verbose=False)
    out_bids = g_bids.transform_raw(raw_bids, verbose=False).get_data()
    np.testing.assert_allclose(out_bids, out_std, rtol=1e-10, atol=1e-15)

    # 2. Custom non-10-20 names with valid 3D coords fall back to dipole projection
    custom_map = {ch: f"E{i + 1}" for i, ch in enumerate(raw.ch_names)}
    raw_custom = raw.copy().rename_channels(custom_map)
    g_custom = Gedai(engine="numpy")
    g_custom.fit_raw(raw_custom, duration=1.0, reference_cov="leadfield", verbose=False)
    out_custom = g_custom.transform_raw(raw_custom, verbose=False).get_data()
    assert out_custom.shape == raw_custom.get_data().shape
    assert np.all(np.isfinite(out_custom))

    # 3. Explicit reference_cov="projection" works on AdaptiveMultibandGedai
    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw_custom, reference_cov="projection", n_jobs=1, verbose=False)
    out_amb = amb.transform_raw(raw_custom, n_jobs=1, verbose=False).get_data()
    assert np.all(np.isfinite(out_amb))


@pytest.mark.parametrize("ref_mode", ["REST", "Cz", "M1+M2", "none"])
def test_reference_modes_end_to_end_and_flat_ref_preservation(ref_mode):
    """REST, channel, external, and none references work across estimators."""
    raw = _sample_raw(n_channels=10, duration_s=5.0)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, verbose=False
    )

    g_ep = Gedai(engine="numpy")
    g_ep.fit_epochs(epochs, reference=ref_mode, verbose=False)
    out_ep = g_ep.transform_epochs(epochs, verbose=False).get_data()
    assert np.all(np.isfinite(out_ep))

    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, reference=ref_mode, n_jobs=1, verbose=False)
    out_raw = amb.transform_raw(raw, n_jobs=1, verbose=False).get_data()
    assert np.all(np.isfinite(out_raw))


def test_flat_recording_reference_retained_and_noisy_falls_back():
    """Flat Cz recording ref is kept; noisy ref falls back to average."""
    raw = _sample_raw(n_channels=27, duration_s=6.0)
    assert "Cz" in raw.ch_names
    cz_idx = raw.ch_names.index("Cz")
    raw_cz = raw.copy()
    # Subtract Cz to simulate Cz as online 0-V recording reference
    raw_cz._data = raw_cz.get_data() - raw_cz.get_data()[cz_idx : cz_idx + 1, :]
    assert np.allclose(raw_cz.get_data()[cz_idx], 0.0)

    g = Gedai(engine="numpy")
    g.fit_raw(
        raw_cz,
        duration=1.0,
        reference="Cz",
        bad_channel_threshold="auto",
        verbose=False,
    )
    assert "Cz" in g.ch_names
    assert g.flat_reference_channels_ == ["Cz"]
    out = g.transform_raw(raw_cz, verbose=False)
    assert "Cz" in out.ch_names
    assert np.all(np.isfinite(out.get_data()))

    # Noisy linked-mastoid channel triggers automatic fallback to average reference
    raw_mastoid = _sample_raw(n_channels=27, duration_s=6.0)
    raw_mastoid.rename_channels({"T7": "M1", "T8": "M2"})
    m1_idx = raw_mastoid.ch_names.index("M1")
    rng = np.random.default_rng(7)
    ch_std = float(np.median(np.std(raw_mastoid.get_data(), axis=1)))
    raw_mastoid._data[m1_idx] += rng.standard_normal(raw_mastoid.n_times) * (
        15.0 * ch_std
    )
    g_fb = Gedai(engine="numpy")
    g_fb.fit_raw(
        raw_mastoid,
        duration=1.0,
        reference="M1+M2",
        bad_channel_threshold="auto",
        verbose=False,
    )
    assert "M1" in g_fb.noisy_channels_
    assert g_fb._reference == "average"

    # Test REST with bad channel interpolation
    raw_bad = _sample_raw(n_channels=12, duration_s=6.0)
    raw_bad.info["bads"] = [raw_bad.ch_names[0]]
    g_rest = Gedai(engine="numpy")
    g_rest.fit_raw(raw_bad, duration=1.0, reference="REST", verbose=False)
    out_interp = g_rest.transform_raw(
        raw_bad, interpolate_bads=True, verbose=False
    )
    assert out_interp.ch_names == raw_bad.ch_names
    assert np.all(np.isfinite(out_interp.get_data()))


def test_sliding_window_adaptation_parity_and_scale_invariance():
    """Sliding-window adaptation works across estimators, scales, and engines."""
    raw = _sample_raw(n_channels=8, duration_s=10.0)

    # 1. Gedai NumPy vs Torch parity + 1e6 scale invariance
    g_np = Gedai(engine="numpy")
    g_np.fit_raw(raw, duration=1.0, smoothing_window=4.0, verbose=False)
    assert "threshold_curve" in g_np._fit
    assert len(g_np._fit["threshold_curve"]) == g_np._fit["n_ep_fitted"]
    assert g_np._fit["n_ep_fitted"] >= 10
    out_np = g_np.transform_raw(raw, verbose=False).get_data()

    g_pt = Gedai(engine="torch")
    g_pt.fit_raw(raw, duration=1.0, smoothing_window=4.0, verbose=False)
    out_pt = g_pt.transform_raw(raw, verbose=False).get_data()
    np.testing.assert_allclose(out_pt, out_np, rtol=1e-9, atol=1e-12)

    raw_uv = raw.copy()
    raw_uv._data = raw_uv.get_data() * 1e6
    g_uv = Gedai(engine="numpy")
    g_uv.fit_raw(raw_uv, duration=1.0, smoothing_window=4.0, verbose=False)
    out_uv = g_uv.transform_raw(raw_uv, verbose=False).get_data() / 1e6
    np.testing.assert_allclose(
        out_uv, out_np, rtol=1e-9, atol=1e-11 * float(np.std(out_np))
    )

    # Shorter recording on transform falls back cleanly to scalar median threshold
    raw_short = raw.copy().crop(0.0, 4.0)
    out_short = g_np.transform_raw(raw_short, verbose=False).get_data()
    assert out_short.shape[1] == raw_short.n_times

    # 2. AdaptiveMultibandGedai sliding window + cache parity
    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, smoothing_window=4.0, n_jobs=1, verbose=False)
    out_cached = amb.transform_raw(raw, n_jobs=1, verbose=False).get_data()
    amb.clear_cache()
    out_uncached = amb.transform_raw(raw, n_jobs=1, verbose=False).get_data()
    np.testing.assert_allclose(out_cached, out_uncached, rtol=1e-10, atol=1e-12)

    # 3. MultibandGedai sliding window
    mb = MultibandGedai(wavelet_level=2, broadband_pass=False, engine="numpy")
    mb.fit_raw(raw, duration=1.0, smoothing_window=4.0, n_jobs=1, verbose=False)
    out_mb = mb.transform_raw(raw, n_jobs=1, verbose=False).get_data()
    assert np.all(np.isfinite(out_mb))


def test_auto_z3_scale_invariance_and_bad_epoch_annotation():
    """auto_z3 is scale-invariant and bad_epoch_threshold marks BAD_gedai."""
    raw = _sample_raw(n_channels=8, duration_s=8.0)
    cache = _load_template_leadfield()
    idx = _match_template_labels(raw.ch_names, cache["elec_names"])
    assert all(j >= 0 for j in idx)

    # Scale invariance for auto_z3 + dipolar_guardrail=True
    g_v = Gedai(engine="numpy")
    g_v.fit_raw(
        raw,
        duration=1.0,
        noise_multiplier="auto_z3",
        dipolar_guardrail=True,
        verbose=False,
    )
    out_v = g_v.transform_raw(raw, verbose=False).get_data()

    raw_s = raw.copy()
    raw_s._data = raw_s.get_data() * 1e4
    g_s = Gedai(engine="numpy")
    g_s.fit_raw(
        raw_s,
        duration=1.0,
        noise_multiplier="auto_z3",
        dipolar_guardrail=True,
        verbose=False,
    )
    out_s = g_s.transform_raw(raw_s, verbose=False).get_data() / 1e4
    np.testing.assert_allclose(
        out_s, out_v, rtol=1e-9, atol=1e-11 * float(np.std(out_v))
    )

    # Inject huge non-leadfield localized artifact on seconds 3-5
    raw_burst = raw.copy()
    sfreq = float(raw_burst.info["sfreq"])
    rng = np.random.default_rng(123)
    s_i, e_i = int(3.0 * sfreq), int(5.0 * sfreq)
    ch_std = float(np.std(raw_burst.get_data()))
    raw_burst._data[0, s_i:e_i] += 25.0 * ch_std * rng.standard_normal(e_i - s_i)

    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw_burst, n_jobs=1, verbose=False)
    clean_none = amb.transform_raw(raw_burst, n_jobs=1, verbose=False)
    assert len(clean_none.annotations) == len(raw_burst.annotations)

    clean_ann = amb.transform_raw(
        raw_burst, bad_epoch_threshold=0.5, n_jobs=1, verbose=False
    )
    assert amb.bad_epochs_ is not None
    assert np.any(amb.bad_epochs_[3:5])
    gedai_descs = [d for d in clean_ann.annotations.description if d == "BAD_gedai"]
    assert len(gedai_descs) >= 1

    epochs = mne.make_fixed_length_epochs(
        raw_burst, duration=1.0, preload=True, verbose=False
    )
    g_ep = Gedai(engine="numpy")
    g_ep.fit_epochs(epochs, verbose=False)
    _ = g_ep.transform_epochs(epochs, bad_epoch_threshold=0.5, verbose=False)
    assert g_ep.bad_epochs_ is not None
    assert g_ep.bad_epochs_.shape == (len(epochs),)
    assert np.any(g_ep.bad_epochs_[3:5])
