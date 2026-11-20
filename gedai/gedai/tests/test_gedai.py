"""Test Gedai."""

import mne
import numpy as np
import pytest
from mne import make_fixed_length_epochs

import gedai.gedai.gedai as gedai_module
from gedai import AdaptiveMultibandGedai, Gedai, MultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.gedai.gedai import _get_channel_multiplier
from gedai.sensai.sensai import _artifact_mask

raw_fname = get_contaminated_eeg_set_path()
raw_eeg = mne.io.read_raw(raw_fname, preload=True)
raw_eeg.drop_channels([ch_name for ch_name in raw_eeg.ch_names if "BIP" in ch_name])
epochs_eeg = make_fixed_length_epochs(raw_eeg, duration=1.0, overlap=0)


def test_gedai_fit_transform_epochs():
    """Test Gedai transform on epochs data."""
    model = Gedai()
    model.fit_epochs(epochs_eeg)
    transformed_epochs = model.transform_epochs(epochs_eeg)
    assert transformed_epochs.info["ch_names"] == epochs_eeg.info["ch_names"]
    assert transformed_epochs.info["sfreq"] == epochs_eeg.info["sfreq"]
    assert epochs_eeg.metadata == transformed_epochs.metadata


def test_gedai_fit_transform_raw():
    """Test Gedai transform on raw data."""
    model = Gedai()
    model.fit_raw(raw_eeg)
    assert model.fit_summary() == model.summary
    transformed_raw = model.transform_raw(raw_eeg)
    assert transformed_raw.info["ch_names"] == raw_eeg.info["ch_names"]
    assert transformed_raw.info["sfreq"] == raw_eeg.info["sfreq"]
    assert raw_eeg.annotations == transformed_raw.annotations


def test_gedai_epochs_picks():
    """Test Gedai fit on epochs data."""
    model = Gedai()
    model.fit_epochs(epochs_eeg, picks="all")
    assert model.ch_names == epochs_eeg.ch_names

    model = Gedai()
    model.fit_epochs(epochs_eeg, picks="data")
    assert model.ch_names == epochs_eeg.ch_names

    model = Gedai()
    model.fit_epochs(epochs_eeg, picks=epochs_eeg.ch_names[:10])
    assert model.ch_names == epochs_eeg.ch_names[:10]

    epochs_transformed = model.transform_epochs(epochs_eeg)
    assert epochs_transformed.ch_names == epochs_eeg.ch_names[:10]

    epochs_test = epochs_eeg.copy()
    epochs_test.load_data()
    epochs_test.pick_channels(epochs_eeg.ch_names[:5])
    with pytest.raises(
        ValueError,
        match="The following channels are missing in the input inst but were present",
    ):
        model.transform_epochs(epochs_test)


def test_gedai_raw_picks():
    """Test Gedai fit on raw data."""
    model = Gedai()
    model.fit_raw(raw_eeg, picks="all")
    assert model.ch_names == raw_eeg.ch_names

    model = Gedai()
    model.fit_raw(raw_eeg, picks="data")
    assert model.ch_names == raw_eeg.ch_names

    model = Gedai()
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


def test_gedai_invalid_n_pc_raises():
    """Explicit n_pc values must be integer-like and within the valid range."""
    with pytest.raises(ValueError, match="n_pc must be an integer in the range"):
        Gedai().fit_epochs(epochs_eeg, n_pc=0)

    with pytest.raises(ValueError, match="n_pc must be an integer in the range"):
        Gedai().fit_epochs(epochs_eeg, n_pc=epochs_eeg.info["nchan"] + 1)


def test_invalid_channel_multiplier_env_raises(monkeypatch):
    """Invalid channel multiplier env vars should fail loudly."""
    monkeypatch.setenv("GEDAI_CHANNEL_MULTIPLIER", "not-a-float")
    with pytest.raises(ValueError, match="GEDAI_CHANNEL_MULTIPLIER"):
        _get_channel_multiplier()


def test_multiband_broadband_pass_forwards_n_pc():
    """The broadband pre-cleaning pass should honor the requested n_pc."""
    model = MultibandGedai(wavelet_level=2, broadband_pass=True)
    model.fit_epochs(epochs_eeg, picks="all", n_pc=2, n_jobs=1)

    assert model._broadband_model is not None
    assert model._broadband_model._n_pc == 2
    for wavelet_fit in model._wavelets_fits:
        if wavelet_fit["model"] is not None:
            assert wavelet_fit["model"]._n_pc == 2


def test_gedai_average_reference_not_reapplied():
    """Ensure data already average-referenced is not modified again."""
    from gedai.gedai._utils import (
        _check_average_reference,
        _prepare_epochs_fit,
        _prepare_raw_fit,
    )

    raw = raw_eeg.copy()
    raw.set_eeg_reference("average", projection=False)
    assert _check_average_reference(raw) is True

    raw_fit = _prepare_raw_fit(raw, picks="eeg")
    assert np.allclose(raw.get_data(), raw_fit.get_data(), atol=1e-12)

    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True)
    assert _check_average_reference(epochs) is True

    epochs_fit = _prepare_epochs_fit(epochs, picks="eeg")
    assert np.allclose(epochs.get_data(), epochs_fit.get_data(), atol=1e-12)


def test_gedai_raw_engine_override_reaches_prefilter(monkeypatch):
    """Explicit raw engine overrides should be forwarded to the prefilter."""
    rng = np.random.RandomState(0)
    ch_names = [f"EEG{i:03d}" for i in range(4)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(rng.randn(4, 400), info, verbose=False)
    cov = mne.Covariance(np.eye(4), ch_names, [], [], 0)
    calls = []
    original = gedai_module._apply_wavelet_highpass_prefilter

    def spy(data, sfreq, lowcut_hz=0.5, engine="auto"):
        calls.append(engine)
        return original(data, sfreq, lowcut_hz=lowcut_hz, engine=engine)

    monkeypatch.setattr(gedai_module, "_apply_wavelet_highpass_prefilter", spy)

    model = Gedai()
    model.fit_raw(
        raw.copy(),
        duration=1.0,
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
        engine="numpy",
    )
    model.transform_raw(raw.copy(), verbose=False, engine="numpy")

    assert calls
    assert all(engine == "numpy" for engine in calls)


def test_gedai_transform_epochs_forwards_threshold(monkeypatch):
    """NumPy epoch cleaning should receive the fitted calibrated threshold."""
    rng = np.random.RandomState(0)
    data = rng.randn(3, 4, 100)
    ch_names = [f"EEG{i:03d}" for i in range(4)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    cov = mne.Covariance(np.eye(4), ch_names, [], [], 0)
    calls = []
    original = gedai_module._process_single_epoch

    def spy(epoch_data, reference_cov, threshold=None):
        calls.append(threshold)
        return original(
            epoch_data,
            reference_cov,
            threshold=threshold,
        )

    model = Gedai(engine="numpy")
    model.fit_epochs(
        epochs.copy(),
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )

    monkeypatch.setattr(gedai_module, "_process_single_epoch", spy)

    model.transform_epochs(epochs.copy(), n_jobs=1, verbose=False, engine="numpy")

    assert calls == [model.threshold] * len(data)


@pytest.mark.parametrize(
    ("model_factory", "fit_kwargs"),
    [
        (lambda: Gedai(engine="numpy"), {"duration": 1.0}),
        (
            lambda: MultibandGedai(
                wavelet_level=2, broadband_pass=False, engine="numpy"
            ),
            {"duration": 1.0, "n_jobs": 1},
        ),
        (
            lambda: AdaptiveMultibandGedai(
                wavelet_level=2, broadband_pass=False, engine="numpy"
            ),
            {"n_jobs": 1},
        ),
    ],
)
def test_estimator_metrics_attributes_parity(model_factory, fit_kwargs):
    """All estimators initialize fit_metrics_ and metrics_ consistently."""
    raw_sub = raw_eeg.copy().pick(raw_eeg.ch_names[:6])
    model = model_factory()

    assert model.fit_metrics_ is None
    assert model.metrics_ is None
    assert model._info is None
    assert "unfitted" in repr(model)

    model.fit_raw(raw_sub, verbose=False, **fit_kwargs)
    assert isinstance(model.fit_metrics_, dict)
    assert "sensai_score" in model.fit_metrics_
    assert model.metrics_ is None
    assert "fitted" in repr(model)

    model.transform_raw(raw_sub, verbose=False)
    assert isinstance(model.metrics_, dict)
    for key in (
        "sensai_score",
        "mean_enova",
        "enova_per_epoch",
        "enova_per_channel",
    ):
        assert key in model.metrics_
    assert "SENSAI=" in repr(model)
    assert "Mean ENOVA=" in repr(model)


@pytest.mark.parametrize(
    ("model_factory", "fit_kwargs"),
    [
        (lambda: Gedai(engine="numpy"), {"duration": 1.0}),
        (
            lambda: MultibandGedai(
                wavelet_level=2, broadband_pass=False, engine="numpy"
            ),
            {"duration": 1.0, "n_jobs": 1},
        ),
        (
            lambda: AdaptiveMultibandGedai(
                wavelet_level=2, broadband_pass=False, engine="numpy"
            ),
            {"n_jobs": 1},
        ),
    ],
)
def test_plot_sensai_without_prior_transform_and_auto_transform(
    model_factory, fit_kwargs
):
    """plot_sensai works both before transform_raw and with raw_after=None."""
    import matplotlib.figure

    raw_sub = raw_eeg.copy().pick(raw_eeg.ch_names[:6])
    rng = np.random.default_rng(0)
    raw_after_custom = raw_sub.copy()
    raw_after_custom._data = raw_after_custom.get_data() + (
        rng.standard_normal(raw_after_custom._data.shape) * 1e-6
    )

    model = model_factory()
    model.fit_raw(raw_sub, verbose=False, **fit_kwargs)
    assert model.metrics_ is None

    fig, plot_metrics = model.plot_sensai(
        raw_sub, raw_after=raw_after_custom, show=False
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(plot_metrics, dict)
    assert model.metrics_ is None

    fig2, plot_metrics2 = model.plot_sensai(raw_sub, raw_after=None, show=False)
    assert isinstance(fig2, matplotlib.figure.Figure)
    assert isinstance(plot_metrics2, dict)
    assert model.metrics_ is not None
    assert "sensai_score" in model.metrics_


@pytest.mark.parametrize("sensai_method", ["gridsearch", "optimize"])
@pytest.mark.parametrize("scale", [1e3, 1e6, 1e-3])
def test_gedai_epochs_scale_invariance(sensai_method, scale):
    """Volts vs rescaled units (e.g. uV) produce identical denoised epochs."""
    ep_raw = epochs_eeg.copy().load_data()
    ep_avg = ep_raw.copy().set_eeg_reference("average", projection=False, verbose=False)

    for ep_base in (ep_raw, ep_avg):
        ep_scaled = ep_base.copy()
        ep_scaled._data = ep_scaled.get_data() * scale

        g_v = Gedai(engine="numpy")
        g_v.fit_epochs(
            ep_base,
            reference_cov="leadfield",
            sensai_method=sensai_method,
            noise_multiplier=3.0,
            verbose=False,
        )
        out_v = g_v.transform_epochs(ep_base, verbose=False).get_data()

        g_s = Gedai(engine="numpy")
        g_s.fit_epochs(
            ep_scaled,
            reference_cov="leadfield",
            sensai_method=sensai_method,
            noise_multiplier=3.0,
            verbose=False,
        )
        out_s = g_s.transform_epochs(ep_scaled, verbose=False).get_data() / scale

        np.testing.assert_allclose(
            out_v,
            out_s,
            rtol=1e-10,
            atol=1e-12 * float(np.std(out_v)),
        )
        np.testing.assert_allclose(g_v.threshold, g_s.threshold, rtol=1e-10)
        np.testing.assert_allclose(
            g_v.fit_metrics_["sensai_score"],
            g_s.fit_metrics_["sensai_score"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            g_v.metrics_["sensai_score"], g_s.metrics_["sensai_score"], atol=1e-6
        )
        np.testing.assert_allclose(
            g_v.metrics_["mean_enova"], g_s.metrics_["mean_enova"], atol=1e-6
        )

    # Custom reference covariance scaling invariance
    from gedai.covariance.covariance import _ensure_cov, _pick_cov

    cov_base = _pick_cov(_ensure_cov("leadfield"), ep_avg.info)
    cov_scaled = cov_base.copy()
    cov_scaled.update(data=cov_base.data * 1e8)

    g_c1 = Gedai(engine="numpy")
    g_c1.fit_epochs(ep_avg, reference_cov=cov_base, verbose=False)
    out_c1 = g_c1.transform_epochs(ep_avg, verbose=False).get_data()

    g_c2 = Gedai(engine="numpy")
    g_c2.fit_epochs(ep_avg, reference_cov=cov_scaled, verbose=False)
    out_c2 = g_c2.transform_epochs(ep_avg, verbose=False).get_data()
    np.testing.assert_allclose(
        out_c1, out_c2, rtol=1e-10, atol=1e-12 * float(np.std(out_c1))
    )


def test_artifact_mask_is_stable_near_threshold():
    """Eigenvalues near the cutoff use consistent classification."""
    threshold = 1.0
    eigenvalues = np.array([threshold * (1 - 5e-13), threshold * (1 + 5e-13)])

    np.testing.assert_array_equal(_artifact_mask(eigenvalues, threshold), [True, True])


def test_multiband_and_adaptive_scale_invariance():
    """Gedai raw, MultibandGedai, and AdaptiveMultibandGedai are scale-invariant."""
    scale = 1e6
    raw_v = raw_eeg.copy().pick(raw_eeg.ch_names[:6])
    raw_uv = raw_v.copy()
    raw_uv._data = raw_uv.get_data() * scale

    ep_v = make_fixed_length_epochs(raw_v, duration=1.0, preload=True, verbose=False)
    ep_uv = ep_v.copy()
    ep_uv._data = ep_uv.get_data() * scale

    # 1. Gedai raw
    g_v = Gedai(engine="numpy")
    g_v.fit_raw(raw_v, duration=1.0, verbose=False)
    out_gv = g_v.transform_raw(raw_v, verbose=False).get_data()

    g_uv = Gedai(engine="numpy")
    g_uv.fit_raw(raw_uv, duration=1.0, verbose=False)
    out_guv = g_uv.transform_raw(raw_uv, verbose=False).get_data() / scale
    np.testing.assert_allclose(
        out_gv, out_guv, rtol=1e-9, atol=1e-11 * float(np.std(out_gv))
    )

    # 2. MultibandGedai epochs
    mb_ep_v = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_ep_v.fit_epochs(ep_v, n_jobs=1, verbose=False)
    out_mb_ev = mb_ep_v.transform_epochs(ep_v, n_jobs=1, verbose=False).get_data()

    mb_ep_uv = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_ep_uv.fit_epochs(ep_uv, n_jobs=1, verbose=False)
    out_mb_euv = (
        mb_ep_uv.transform_epochs(ep_uv, n_jobs=1, verbose=False).get_data() / scale
    )
    np.testing.assert_allclose(
        out_mb_ev, out_mb_euv, rtol=1e-9, atol=1e-11 * float(np.std(out_mb_ev))
    )

    # 3. MultibandGedai raw
    mb_r_v = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_r_v.fit_raw(raw_v, duration=1.0, n_jobs=1, verbose=False)
    out_mb_rv = mb_r_v.transform_raw(raw_v, n_jobs=1, verbose=False).get_data()

    mb_r_uv = MultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    mb_r_uv.fit_raw(raw_uv, duration=1.0, n_jobs=1, verbose=False)
    out_mb_ruv = (
        mb_r_uv.transform_raw(raw_uv, n_jobs=1, verbose=False).get_data() / scale
    )
    np.testing.assert_allclose(
        out_mb_rv, out_mb_ruv, rtol=1e-9, atol=1e-11 * float(np.std(out_mb_rv))
    )

    # 4. AdaptiveMultibandGedai raw
    ad_v = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    ad_v.fit_raw(raw_v, n_jobs=1, verbose=False)
    out_adv = ad_v.transform_raw(raw_v, n_jobs=1, verbose=False).get_data()

    ad_uv = AdaptiveMultibandGedai(wavelet_level=2, broadband_pass=True, engine="numpy")
    ad_uv.fit_raw(raw_uv, n_jobs=1, verbose=False)
    out_aduv = (
        ad_uv.transform_raw(raw_uv, n_jobs=1, verbose=False).get_data() / scale
    )
    np.testing.assert_allclose(
        out_adv, out_aduv, rtol=1e-9, atol=1e-11 * float(np.std(out_adv))
    )


def test_check_average_reference_scale_invariance():
    """_check_average_reference should be invariant to amplitude units."""
    from gedai.gedai._utils import _check_average_reference

    rng = np.random.default_rng(0)
    info = mne.create_info(["Fp1", "Fp2", "F3", "F4"], sfreq=100.0, ch_types="eeg")
    # Mean offset 5e-7 V (~50% of 1e-6 V signal), previously missed by atol=1e-6
    data_v = rng.standard_normal((4, 200)) * 1e-6 + 5e-7
    raw_v = mne.io.RawArray(data_v, info, verbose=False)
    raw_uv = mne.io.RawArray(data_v * 1e6, info, verbose=False)

    assert _check_average_reference(raw_v) is False
    assert _check_average_reference(raw_uv) is False

    raw_v.set_eeg_reference("average", projection=False, verbose=False)
    raw_uv.set_eeg_reference("average", projection=False, verbose=False)
    assert _check_average_reference(raw_v) is True
    assert _check_average_reference(raw_uv) is True
