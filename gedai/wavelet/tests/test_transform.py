"""Unit tests for wavelet transforms, chunked MODWT, and backend equivalence."""

import numpy as np
import pytest

from gedai.utils._torch_backend import has_torch
from gedai.wavelet.transform import (
    _apply_wavelet_highpass_prefilter,
    _modwt_haar_single_band,
    compute_wavelet_level,
    get_modwt_band_limits,
)


def test_modwt_haar_single_band_shape_and_dtype():
    """Verify output shape and dtype preservation (no forced float64 cast)."""
    n_times, n_ch = 2000, 8
    level = 4
    band_idx = 1

    # float32 input
    data_f32 = np.random.randn(n_times, n_ch).astype(np.float32)
    out_f32 = _modwt_haar_single_band(data_f32, level, band_idx, engine="numpy")
    assert out_f32.shape == (n_ch, n_times)
    assert out_f32.dtype == np.float32

    # float64 input
    data_f64 = np.random.randn(n_times, n_ch).astype(np.float64)
    out_f64 = _modwt_haar_single_band(data_f64, level, band_idx, engine="numpy")
    assert out_f64.shape == (n_ch, n_times)
    assert out_f64.dtype == np.float64


@pytest.mark.skipif(not has_torch(), reason="PyTorch not available")
def test_modwt_haar_single_band_numpy_torch_parity():
    """Verify exact numerical parity between NumPy and PyTorch implementations."""
    n_times, n_ch = 4000, 16
    data = np.random.randn(n_times, n_ch).astype(np.float32)

    for level in [3, 5]:
        for band in range(level + 1):
            out_numpy = _modwt_haar_single_band(data, level, band, engine="numpy")
            out_torch = _modwt_haar_single_band(data, level, band, engine="torch")

            max_err = np.max(np.abs(out_numpy - out_torch))
            assert max_err < 1e-6, f"Parity failure level={level}, band={band}, err={max_err}"


def test_modwt_haar_single_band_chunking_parity_numpy():
    """Verify stateful overlap-save chunking matches unchunked output in NumPy."""
    n_times, n_ch = 8000, 10
    data = np.random.randn(n_times, n_ch).astype(np.float32)

    level = 4
    for band in range(level + 1):
        out_global = _modwt_haar_single_band(
            data, level, band, chunk_size=20000, engine="numpy"
        )
        out_chunked = _modwt_haar_single_band(
            data, level, band, chunk_size=1500, engine="numpy"
        )

        max_err = np.max(np.abs(out_global - out_chunked))
        assert max_err < 1e-6, f"NumPy chunking mismatch band={band}, err={max_err}"


@pytest.mark.skipif(not has_torch(), reason="PyTorch not available")
def test_modwt_haar_single_band_chunking_parity_torch():
    """Verify stateful overlap-save chunking matches unchunked output in PyTorch."""
    n_times, n_ch = 8000, 10
    data = np.random.randn(n_times, n_ch).astype(np.float32)

    level = 4
    for band in range(level + 1):
        out_global = _modwt_haar_single_band(
            data, level, band, chunk_size=20000, engine="torch"
        )
        out_chunked = _modwt_haar_single_band(
            data, level, band, chunk_size=1500, engine="torch"
        )

        max_err = np.max(np.abs(out_global - out_chunked))
        assert max_err < 1e-6, f"Torch chunking mismatch band={band}, err={max_err}"


def test_apply_wavelet_highpass_prefilter():
    """Test drift pre-filter functionality and dtype preservation."""
    sfreq = 200.0
    n_ch, n_times = 8, 4000
    t = np.linspace(0, n_times / sfreq, n_times)
    # 0.1 Hz drift + random noise
    drift = np.sin(2 * np.pi * 0.1 * t)
    data = (np.random.randn(n_ch, n_times) * 0.1 + drift).astype(np.float32)

    filtered_np = _apply_wavelet_highpass_prefilter(
        data, sfreq, lowcut_hz=0.5, engine="numpy"
    )
    assert filtered_np.shape == data.shape
    assert filtered_np.dtype == np.float32

    if has_torch():
        filtered_th = _apply_wavelet_highpass_prefilter(
            data, sfreq, lowcut_hz=0.5, engine="torch"
        )
        assert filtered_th.shape == data.shape
        max_diff = np.max(np.abs(filtered_np - filtered_th))
        assert max_diff < 1e-6
