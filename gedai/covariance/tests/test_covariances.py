"""Test Covariances."""

import mne
import numpy as np
import pytest
from mne.datasets import testing

from gedai.covariance import (
    compute_covariance_from_channel_positions,
    compute_covariance_from_dipole_projection,
    compute_covariance_from_forward,
    project_leadfield_dipole,
)
from gedai.covariance.covariance import (
    _ensure_cov,
    _load_template_leadfield,
    _match_template_labels,
    _pick_cov,
    _resolve_reference_and_cov,
)

data_path = testing.data_path(download=False)
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif"


@pytest.fixture(scope="module")
def sample_info():
    """Create a small EEG montage fixture for covariance tests."""
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
    ]
    info = mne.create_info(ch_names, sfreq=250.0, ch_types="eeg")
    montage_name = (
        "colin27_1020"
        if "colin27_1020" in mne.channels.get_builtin_montages()
        else "standard_1020"
    )
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage)
    return info


def test_ensure_cov(sample_info):
    """Test _ensure_cov."""
    # test with a covariance object
    cov = mne.make_ad_hoc_cov(sample_info)
    assert _ensure_cov(cov) == cov

    # test with the string "leadfield"
    cov_leadfield = _ensure_cov("leadfield")
    assert isinstance(cov_leadfield, mne.Covariance)

    # test with an invalid string
    with pytest.raises(ValueError, match="Reference covariance must be 'leadfield'"):
        _ensure_cov("invalid_string")


def test_pick_cov(sample_info):
    """Test _pick_cov."""
    cov = mne.make_ad_hoc_cov(sample_info)
    ch_names = sample_info["ch_names"][:5]  # pick a subset of channels
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = [ch_name.lower() for ch_name in sample_info["ch_names"]]
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = ["nonexistent_channel"]
    with pytest.raises(
        ValueError, match="No matching channel names found between inst and cov"
    ):
        _pick_cov(cov, ch_names)

    ch_names = ["Fp1", "nonexistent_channel"]
    with pytest.raises(
        ValueError,
        match="Only a subset of channels in the instance are present in the covariance",
    ):
        _pick_cov(cov, ch_names)


def test_compute_covariance_from_channel_positions(sample_info):
    """Test compute_covariance_from_channel_positions."""
    cov = compute_covariance_from_channel_positions(sample_info)
    assert isinstance(cov, mne.Covariance)


@testing.requires_testing_data
def test_compute_covariance_from_forward():
    """Test compute_covariance_from_forward."""
    if not fname_fwd.exists():
        pytest.skip("Requires MNE testing dataset")
    forward = mne.read_forward_solution(fname_fwd)
    cov = compute_covariance_from_forward(forward)
    assert isinstance(cov, mne.Covariance)


def test_token_safe_label_matching_and_pick_cov():
    """Token-safe label matcher handles BIDS/EDF names without collisions."""
    tmpl = _load_template_leadfield()["elec_names"]
    clinical = [
        "EEG Fp1-REF",
        "EEG Fp2-REF",
        "EEG Cz-LE",
        "C3-M2",
        "C10",
        "AFp1",
        "EOG_Fp1",
    ]
    idxs = _match_template_labels(clinical, tmpl)
    matched = [tmpl[j] if j >= 0 else None for j in idxs]
    assert matched[:4] == ["Fp1", "Fp2", "Cz", "C3"]
    assert matched[4] != "C1"
    assert matched[5] == "AFp1"
    assert matched[6] is None

    cov_lf = _ensure_cov("leadfield")
    picked = _pick_cov(cov_lf, ["EEG Fp1-REF", "EEG Cz-LE", "C3-M2"])
    assert picked.ch_names == ["EEG Fp1-REF", "EEG Cz-LE", "C3-M2"]
    assert picked.data.shape == (3, 3)


def test_dipole_projection_and_gsn128():
    """3D dipole projection matches 10-20 BEM > 0.99 and supports GSN-128."""
    ch19 = [
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
    ]
    info19 = mne.create_info(ch19, sfreq=200.0, ch_types="eeg")
    montage_1020 = (
        "colin27_1020"
        if "colin27_1020" in mne.channels.get_builtin_montages()
        else "standard_1020"
    )
    info19.set_montage(montage_1020)

    g_proj = project_leadfield_dipole(info19, use_avgref=True)
    assert g_proj.shape == (19, 3630)

    cov_proj = compute_covariance_from_dipole_projection(info19, reference="average")
    cov_fif = _pick_cov(_ensure_cov("leadfield"), info19)
    c_raw_proj = cov_proj.data / np.trace(cov_proj.data)
    c_raw_fif = cov_fif.data / np.trace(cov_fif.data)
    raw_corr = float(np.corrcoef(c_raw_proj.ravel(), c_raw_fif.ravel())[0, 1])
    assert raw_corr > 0.97

    rn = np.eye(19) - np.ones((19, 19)) / 19.0
    c_proj = (rn @ cov_proj.data @ rn) / np.trace(rn @ cov_proj.data @ rn)
    c_ref = (rn @ cov_fif.data @ rn) / np.trace(rn @ cov_fif.data @ rn)
    corr = float(np.corrcoef(c_proj.ravel(), c_ref.ravel())[0, 1])
    assert corr > 0.99

    assert _ensure_cov("projection") == "projection"
    cov_auto_proj, _ = _resolve_reference_and_cov("projection", info19)
    np.testing.assert_allclose(cov_auto_proj.data, cov_proj.data, rtol=1e-12)

    m128 = mne.channels.make_standard_montage("GSN-HydroCel-128")
    info128 = mne.create_info(m128.ch_names, sfreq=250.0, ch_types="eeg")
    info128.set_montage(m128)
    cov128 = compute_covariance_from_dipole_projection(info128)
    assert cov128.data.shape == (128, 128)
    assert np.all(np.isfinite(cov128.data))

    info_noloc = mne.create_info(ch19[:4], sfreq=200.0, ch_types="eeg")
    with pytest.raises(ValueError, match="Valid 3D channel positions"):
        project_leadfield_dipole(info_noloc)


def test_unified_reference_algebra_and_rest():
    """Reference algebra handles REST, linked pairs, in-montage, and external refs."""
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "Cz",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "M1",
        "M2",
        "TP9",
        "TP10",
    ]
    info = mne.create_info(ch_names, sfreq=200.0, ch_types="eeg")
    montage_1005 = (
        "colin27_1005"
        if "colin27_1005" in mne.channels.get_builtin_montages()
        else "standard_1005"
    )
    info.set_montage(montage_1005)

    cov_rest, spec_rest = _resolve_reference_and_cov(
        "leadfield", info, reference="REST"
    )
    assert spec_rest["is_rest"] is True
    m_rest = spec_rest["rest_operator"]
    assert m_rest.shape == (len(ch_names), len(ch_names))
    assert np.max(np.abs(m_rest @ np.ones(len(ch_names)))) < 1e-12

    cache = _load_template_leadfield()
    idx = _match_template_labels(ch_names, cache["elec_names"])
    g_raw = cache["gain_raw"][idx]
    g_ar = g_raw - g_raw.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(11)
    _, _, vt = np.linalg.svd(g_ar, full_matrices=False)
    s_obs = vt[: len(ch_names) - 1].T @ rng.standard_normal((len(ch_names) - 1, 50))
    x_true = g_raw @ s_obs
    x_rec = x_true - x_true[5:6, :] + 1e3 * rng.standard_normal((1, 50))
    x_restored = m_rest @ x_rec
    assert np.linalg.norm(x_restored - x_true) / np.linalg.norm(x_true) < 1e-12

    for ref_mode in ("M1+M2", "TP9+TP10", "Cz", "A1+A2", "A1", "none"):
        cov_m, spec_m = _resolve_reference_and_cov(
            "leadfield", info, reference=ref_mode
        )
        assert cov_m.data.shape == (len(ch_names), len(ch_names))
        if ref_mode in ("A1+A2", "A1"):
            assert spec_m["is_external_ref"] is True
        elif ref_mode in ("M1+M2", "TP9+TP10", "Cz"):
            assert np.all(spec_m["required_indices"]) or len(
                spec_m["required_indices"]
            ) in (1, 2)


def test_compute_covariance_from_forward_eeg_reference(sample_info):
    """compute_covariance_from_forward matches reference space of EEG forwards."""
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.095)
    src = mne.setup_volume_source_space(
        pos=35.0, sphere=(0.0, 0.0, 0.04, 0.075), verbose=False
    )
    fwd = mne.make_forward_solution(
        sample_info, trans=None, src=src, bem=sphere, meg=False, eeg=True, verbose=False
    )

    cov_avg = compute_covariance_from_forward(fwd)
    assert np.max(np.abs(cov_avg.data.sum(axis=1))) < 1e-12 * np.trace(cov_avg.data)

    cov_none = compute_covariance_from_forward(fwd, reference="none")
    raw_gram = fwd["sol"]["data"] @ fwd["sol"]["data"].T
    np.testing.assert_allclose(cov_none.data, 0.5 * (raw_gram + raw_gram.T), rtol=1e-12)
    assert not np.allclose(cov_avg.data, cov_none.data)
