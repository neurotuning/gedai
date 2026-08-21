"""Tests for ENOVA and cleaning metrics module."""

import numpy as np
import pytest

from gedai.metrics import (
    compute_composite_sensai,
    compute_enova_per_channel,
    compute_enova_per_epoch,
    enova_summary,
)


def test_enova_metrics():
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
