"""Tests for ENOVA and cleaning metrics module."""

import mne
import numpy as np
import pytest

from gedai.metrics import (
    annotate_bad_epochs,
    compute_composite_sensai,
    compute_enova_per_channel,
    compute_enova_per_epoch,
    enova_summary,
)


def test_enova_metrics():
    """Exercise the ENOVA and composite SENSAI metric utilities."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 8, 1000
    epoch_samples = 200

    clean = rng.standard_normal((n_ch, n_times))
    noise = 0.5 * rng.standard_normal((n_ch, n_times))

    # Per-epoch ENOVA
    enova_ep = compute_enova_per_epoch(clean, noise, epoch_samples)
    assert len(enova_ep) == n_times // epoch_samples
    assert np.all(enova_ep >= 0)

    # Per-channel ENOVA
    enova_ch = compute_enova_per_channel(clean, noise, epoch_samples)
    assert len(enova_ch) == n_ch
    assert np.all(enova_ch >= 0)

    # Summary
    stats = enova_summary(enova_ep)
    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert stats["min"] <= stats["mean"] <= stats["max"]

    # Composite SENSAI
    ref_cov = np.eye(n_ch)
    score = compute_composite_sensai(clean, noise, sfreq=200.0, reference_cov=ref_cov)
    assert isinstance(score, float)


def test_enova_dc_offset_invariance():
    """Verify ENOVA is invariant to static inter-channel DC offsets."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 10, 2000
    epoch_samples = 400

    clean = rng.standard_normal((n_ch, n_times))
    noise = 0.3 * rng.standard_normal((n_ch, n_times))

    enova_baseline = compute_enova_per_epoch(clean, noise, epoch_samples)

    # Add huge DC offset across channels (typical in raw MEG/SQUID sensors)
    dc_offsets = rng.uniform(-1e6, 1e6, size=(n_ch, 1))
    noise_with_dc = noise + dc_offsets

    enova_dc = compute_enova_per_epoch(clean, noise_with_dc, epoch_samples)

    # ENOVA should be virtually identical because temporal variance
    # ignores static channel means.
    np.testing.assert_allclose(enova_dc, enova_baseline, rtol=1e-5)


def test_enova_flat_dc_roundoff_guard():
    """Constant-DC or zero epochs yield exact 0.0 without float64 blowup."""
    rng = np.random.default_rng(0)
    n_ch, ep_samples = 6, 100
    orig = rng.standard_normal((n_ch, 3 * ep_samples)) * 1e-5
    # Epoch 1 is constant DC (exact signal with 0 noise)
    dc = 1e-3
    orig[:, ep_samples : 2 * ep_samples] = dc
    clean = orig.copy()
    # Subtract and add DC to simulate floating-point residual ~1e-19
    clean[:, ep_samples : 2 * ep_samples] = (dc + 1e-20) - 1e-20
    noise = orig - clean

    ep_enova = compute_enova_per_epoch(clean, noise, ep_samples)
    ch_enova = compute_enova_per_channel(clean, noise, ep_samples)
    assert ep_enova[1] == 0.0
    assert np.all(np.isfinite(ch_enova))
    assert np.all(ch_enova == 0.0)


def test_annotate_bad_epochs():
    """annotate_bad_epochs merges contiguous intervals and validates inputs."""
    info = mne.create_info(["Fp1", "Fp2"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 600)), info, first_samp=100, verbose=False)
    enova = np.array([0.1, 0.9, 0.95, 0.2, 0.91, 0.3], dtype=np.float32)

    ann = annotate_bad_epochs(raw, enova, epoch_duration=1.0, threshold=0.85)
    assert len(ann) == 2
    assert list(ann.description) == ["BAD_gedai", "BAD_gedai"]
    np.testing.assert_allclose(ann.onset, [2.0, 5.0])
    np.testing.assert_allclose(ann.duration, [2.0, 1.0])

    empty_ann = annotate_bad_epochs(
        raw, enova * 0.1, epoch_duration=1.0, threshold=0.85
    )
    assert len(empty_ann) == 0

    with pytest.raises(ValueError, match="threshold must be > 0"):
        annotate_bad_epochs(raw, enova, threshold=0.0)
