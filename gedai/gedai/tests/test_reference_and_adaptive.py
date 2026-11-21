"""Integration tests for reference modes, dipole projection, and adaptive processing."""

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


def test_auto_scale_invariance_and_bad_epoch_annotation():
    """Adaptive Z3 'auto' is scale-invariant and bad_epoch_threshold marks BAD_gedai."""
    raw = _sample_raw(n_channels=8, duration_s=8.0)
    cache = _load_template_leadfield()
    idx = _match_template_labels(raw.ch_names, cache["elec_names"])
    assert all(j >= 0 for j in idx)

    # Scale invariance for adaptive Z3 "auto" with built-in EEG dipolar guardrail
    g_v = Gedai(engine="numpy")
    g_v.fit_raw(
        raw,
        duration=1.0,
        strength="auto",
        verbose=False,
    )
    out_v = g_v.transform_raw(raw, verbose=False).get_data()

    raw_s = raw.copy()
    raw_s._data = raw_s.get_data() * 1e4
    g_s = Gedai(engine="numpy")
    g_s.fit_raw(
        raw_s,
        duration=1.0,
        strength="auto",
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


def test_plot_sensai_with_excluded_and_interpolated_bad_channel():
    """plot_sensai aligns channels and reference when bads are interpolated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _sample_raw(n_channels=10, duration_s=6.0)
    rng = np.random.default_rng(99)
    ch_std = float(np.median(np.std(raw.get_data(), axis=1)))
    raw._data[0] += 20.0 * ch_std * rng.standard_normal(raw.n_times)

    amb = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    amb.fit_raw(raw, bad_channel_threshold="auto", n_jobs=1, verbose=False)
    assert raw.ch_names[0] in amb.bad_channels_

    clean = amb.transform_raw(raw, interpolate_bads=True, n_jobs=1, verbose=False)
    assert len(clean.ch_names) == len(raw.ch_names)

    fig, metrics = amb.plot_sensai(raw_before=raw, raw_after=clean, show=False)
    plt.close(fig)
    assert metrics["ssi_before_mean"] > 0.1
    assert metrics["ssi_after_mean"] > 0.1
    assert np.isfinite(metrics["ssi_noise_mean"])


def test_perceptual_strength_equivalence_and_monotonicity():
    """Strength (0-10) matches auto/auto+/auto- and increases cleaning monotonically."""
    raw = _sample_raw(n_channels=8, duration_s=6.0)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, verbose=False
    )

    # 1. Gedai epochs & raw: strength=7.0 and int 7 match "auto"
    g_auto = Gedai(engine="numpy")
    g_auto.fit_epochs(epochs, strength="auto", verbose=False)
    out_auto = g_auto.transform_epochs(epochs, verbose=False).get_data()

    g_str = Gedai(engine="numpy")
    g_str.fit_epochs(epochs, strength=7.0, verbose=False)
    out_str = g_str.transform_epochs(epochs, verbose=False).get_data()
    np.testing.assert_allclose(out_str, out_auto, rtol=1e-12, atol=1e-15)
    assert g_str.fit_metrics_["strength"] == pytest.approx(7.0)
    assert "noise_multiplier" not in g_str.fit_metrics_

    g_str_int = Gedai(engine="numpy")
    g_str_int.fit_raw(raw, duration=1.0, strength=7, verbose=False)
    g_raw_auto = Gedai(engine="numpy")
    g_raw_auto.fit_raw(raw, duration=1.0, strength="auto", verbose=False)
    np.testing.assert_allclose(
        g_str_int.transform_raw(raw, verbose=False).get_data(),
        g_raw_auto.transform_raw(raw, verbose=False).get_data(),
        rtol=1e-12,
        atol=1e-15,
    )

    # 2. MultibandGedai & AdaptiveMultibandGedai: strength=8 matches "auto+"
    # and strength=6 matches "auto-"
    mb_plus = MultibandGedai(wavelet_level=2, broadband_pass=False, engine="numpy")
    mb_plus.fit_raw(raw, duration=1.0, strength="auto+", verbose=False)
    mb_s8 = MultibandGedai(wavelet_level=2, broadband_pass=False, engine="numpy")
    mb_s8.fit_raw(raw, duration=1.0, strength=8.0, verbose=False)
    np.testing.assert_allclose(
        mb_s8.transform_raw(raw, verbose=False).get_data(),
        mb_plus.transform_raw(raw, verbose=False).get_data(),
        rtol=1e-12,
        atol=1e-15,
    )

    amb_s6 = AdaptiveMultibandGedai(
        wavelet_level=2, broadband_pass=True, engine="numpy"
    )
    amb_s6.fit_raw(
        raw, strength=6.0, bad_channel_threshold="auto", n_jobs=1, verbose=False
    )
    amb_minus_ref = AdaptiveMultibandGedai(
        wavelet_level=2, broadband_pass=True, engine="numpy"
    )
    amb_minus_ref.fit_raw(
        raw,
        strength="auto-",
        bad_channel_threshold="auto",
        n_jobs=1,
        verbose=False,
    )
    np.testing.assert_allclose(
        amb_s6.transform_raw(raw, n_jobs=1, verbose=False).get_data(),
        amb_minus_ref.transform_raw(raw, n_jobs=1, verbose=False).get_data(),
        rtol=1e-12,
        atol=1e-15,
    )

    # 3. Monotonicity: higher strength -> stronger cleaning (>= mean ENOVA)
    g_lo = Gedai(engine="numpy")
    g_lo.fit_raw(raw, duration=1.0, strength=3.0, verbose=False)
    _ = g_lo.transform_raw(raw, verbose=False)
    g_hi = Gedai(engine="numpy")
    g_hi.fit_raw(raw, duration=1.0, strength=9.5, verbose=False)
    _ = g_hi.transform_raw(raw, verbose=False)
    assert g_hi.fit_metrics_["strength"] == pytest.approx(9.5)
    assert g_hi.metrics_["mean_enova"] >= g_lo.metrics_["mean_enova"] - 1e-12

    # 4. Out-of-range strength raises ValueError
    amb_err = AdaptiveMultibandGedai(wavelet_level=2, engine="numpy")
    with pytest.raises(ValueError, match="strength must be finite"):
        amb_err.fit_raw(raw, strength=12.0, verbose=False)


def test_sensai_viz_psd_topoplot_and_fallbacks():
    """3-column SENSAI dashboard renders ENOVA-weighted PSD, topoplot, and fallbacks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gedai.viz import plot_sensai_visualization

    # 1. Standard 10-20 recording with valid 3D coords -> topoplot path
    raw = _sample_raw(n_channels=10, duration_s=6.0)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    g = Gedai(engine="numpy")
    g.fit_raw(raw, duration=1.0, verbose=False)
    clean = g.transform_raw(raw, verbose=False)

    fig, metrics = g.plot_sensai(raw_before=raw, raw_after=clean, show=False)
    try:
        for key in (
            "enova_per_channel",
            "psd_before_db",
            "psd_after_db",
            "psd_noise_db",
            "psd_freqs",
            "enova_weights",
        ):
            assert key in metrics
            assert np.all(np.isfinite(metrics[key]))

        assert metrics["enova_per_channel"].shape == (len(g.ch_names),)
        assert metrics["enova_weights"].sum() == pytest.approx(1.0, abs=1e-12)
        assert float(np.diff(metrics["psd_freqs"])[0]) == pytest.approx(0.5, abs=1e-6)
        assert metrics["psd_before_db"].shape == metrics["psd_freqs"].shape
        assert metrics["psd_after_db"].shape == metrics["psd_freqs"].shape
        assert metrics["psd_noise_db"].shape == metrics["psd_freqs"].shape
        # Both SSI colorbar and ENOVA topoplot colorbar remain visible
        assert fig.axes[-1].get_visible()
    finally:
        plt.close(fig)

    # 2. 10-20 + legacy names (T3/T4) with missing coords resolve via 10-05 template
    ch_legacy = ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2"]
    sfreq = 250.0
    rng = np.random.default_rng(7)
    info_1020 = mne.create_info(ch_legacy, sfreq=sfreq, ch_types="eeg")
    data_b = rng.standard_normal((len(ch_legacy), int(4 * sfreq))) * 20e-6
    data_a = data_b * 0.6
    raw_b = mne.io.RawArray(data_b, info_1020, verbose=False)
    raw_a = mne.io.RawArray(data_a, info_1020, verbose=False)
    ref_cov = np.eye(len(ch_legacy))
    fig_1020, _ = plot_sensai_visualization(
        raw_b, raw_a, reference_cov=ref_cov, show=False
    )
    try:
        assert fig_1020.axes[-1].get_visible()
    finally:
        plt.close(fig_1020)

    # 3. Arbitrary names with missing coords -> bar-chart fallback + short recording
    ch_arb = [f"CH{i}" for i in range(len(ch_legacy))]
    info_arb = mne.create_info(ch_arb, sfreq=sfreq, ch_types="eeg")
    raw_arb_b = mne.io.RawArray(data_b[:, : int(1.0 * sfreq)], info_arb, verbose=False)
    raw_arb_a = mne.io.RawArray(data_a[:, : int(1.0 * sfreq)], info_arb, verbose=False)
    custom_enova = np.array([np.inf, np.nan, 0.2, 0.5, 0.8, 1.0, 0.1, 0.3])
    fig_bar, met_bar = plot_sensai_visualization(
        raw_arb_b,
        raw_arb_a,
        reference_cov=ref_cov,
        epoch_duration_sec=0.5,
        enova_per_channel=custom_enova,
        show=False,
    )
    try:
        assert not fig_bar.axes[-1].get_visible()
        assert met_bar["enova_weights"].sum() == pytest.approx(1.0, abs=1e-12)
        assert float(np.diff(met_bar["psd_freqs"])[0]) == pytest.approx(0.5, abs=1e-6)
    finally:
        plt.close(fig_bar)

    # 4. Proportion array with >1.5 outlier scales to %; explicit % stays intact
    prop_with_outlier = np.array([0.2, 0.5, 0.9, 1.8, 0.3, 0.4, 0.6, 0.7])
    fig_out, _ = plot_sensai_visualization(
        raw_arb_b,
        raw_arb_a,
        reference_cov=ref_cov,
        epoch_duration_sec=0.5,
        enova_per_channel=prop_with_outlier,
        show=False,
    )
    try:
        heights = [patch.get_height() for patch in fig_out.axes[-2].patches]
        np.testing.assert_allclose(
            heights, [20.0, 50.0, 90.0, 100.0, 30.0, 40.0, 60.0, 70.0]
        )
    finally:
        plt.close(fig_out)

    pct_explicit = np.array([10.0, 40.0, 80.0, 20.0, 30.0, 50.0, 60.0, 90.0])
    fig_pct, _ = plot_sensai_visualization(
        raw_arb_b,
        raw_arb_a,
        reference_cov=ref_cov,
        epoch_duration_sec=0.5,
        enova_per_channel=pct_explicit,
        show=False,
    )
    try:
        heights_pct = [patch.get_height() for patch in fig_pct.axes[-2].patches]
        np.testing.assert_allclose(heights_pct, pct_explicit)
    finally:
        plt.close(fig_pct)

    # 5. Length mismatch on explicit enova_per_channel raises ValueError
    with pytest.raises(ValueError, match="enova_per_channel has length"):
        plot_sensai_visualization(
            raw_arb_b,
            raw_arb_a,
            reference_cov=ref_cov,
            epoch_duration_sec=0.5,
            enova_per_channel=np.array([0.1, 0.2]),
            show=False,
        )
