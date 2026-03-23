import numpy as np
import pywt
from joblib import Parallel, delayed


def _swt_single_channel(signal, wavelet_type, level):
    """Run SWT on a single 1-D signal, return MATLAB-order coefficients."""
    try:
        coeffs = pywt.swt(signal, wavelet_type, level=level, norm=True, axis=0)
    except TypeError:
        coeffs = pywt.swt(signal, wavelet_type, level=level, axis=0)
    VJ = coeffs[0][0]
    details = [c[1] for c in coeffs]
    details.reverse()
    return np.array(details + [VJ])  # (n_bands, n_samples)


def modwt(data, wavelet_type, level, n_jobs=1):
    """
    Perform MODWT using PyWavelets (pywt.swt), matching MATLAB coefficient order.

    Input: data (Samples x Channels)
    Output: (Bands x Samples x Channels). Order: [W1, ..., WJ, VJ].

    Notes
    -----
    ``n_jobs`` defaults to 1 (sequential).  Pass a larger value only when
    this function is **not** already called from a parallel outer loop;
    using ``n_jobs=-1`` inside a parallel context causes oversubscription.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape

    if n_channels == 1 or n_jobs == 1:
        results = [_swt_single_channel(data[:, i], wavelet_type, level)
                   for i in range(n_channels)]
    else:
        # loky (process-based) backend: bypasses the GIL so each worker
        # fully occupies its own core.  thread-based backends are GIL-bound
        # and effectively run on a single core for Python-heavy callees.
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_swt_single_channel)(data[:, i], wavelet_type, level)
            for i in range(n_channels)
        )

    wpt = np.stack(results, axis=-1)  # (n_bands, n_samples, n_channels)
    return wpt


def _iswt_channel(channel_coeffs, wavelet_type):
    """Run a single iSWT reconstruction for one channel.

    Parameters
    ----------
    channel_coeffs : list of (cA, cD) tuples
        Coefficient pairs ordered from coarsest to finest level, as expected
        by ``pywt.iswt``.
    wavelet_type : str
        Wavelet name.

    Returns
    -------
    signal : np.ndarray, shape (n_samples,)
    """
    try:
        return pywt.iswt(channel_coeffs, wavelet_type, norm=True)
    except TypeError:
        return pywt.iswt(channel_coeffs, wavelet_type)


def _mra_single_channel(wpt_ch, wavelet_type):
    """Compute all MRA bands for a single channel using an O(L) cascade.

    Instead of running one iSWT per band (O(L²) total), we compute successive
    approximation signals A_J, A_{J-1}, …, A_0 by progressively zeroing detail
    coefficients from the finest level upward.  The detail bands are then
    recovered by differencing::

        D_j = A_j - A_{j-1}   (j = 1 … J)
        A_J  = full reconstruction (no change)

    This requires only L+1 iSWT calls versus (L+1)² for the naive approach.

    Parameters
    ----------
    wpt_ch : np.ndarray, shape (n_bands, n_samples)
        MODWT coefficients for one channel.  Band order: [D1, …, DJ, AJ].
    wavelet_type : str
        Wavelet name.

    Returns
    -------
    mra_ch : np.ndarray, shape (n_bands, n_samples)
        MRA reconstruction.  Same band order: [D1, …, DJ, AJ].
    """
    n_bands, n_samples = wpt_ch.shape
    level = n_bands - 1

    # wpt_ch layout: indices 0..level-1 are details D1..DJ,
    # index `level` is the approximation AJ.
    # pywt.iswt expects a list of (cA_k, cD_k) from coarsest (k=L) to finest (k=1).
    # In our array: detail k is at index k-1,  approximation (only at coarsest) at `level`.

    def _build_coeffs(zero_details_above):
        """Build pywt.iswt input list zeroing details finer than `zero_details_above`.

        `zero_details_above` is the finest detail index (0-based in wpt_ch) to
        *keep*.  Details at indices > zero_details_above are set to zero.
        """
        pairs = []
        zeros = np.zeros(n_samples)
        for k in range(level, 0, -1):          # k = L … 1  (coarsest to finest)
            detail_idx = k - 1                 # D_k lives at wpt_ch[k-1]
            cA = wpt_ch[level] if k == level else zeros
            cD = wpt_ch[detail_idx] if detail_idx <= zero_details_above else zeros
            pairs.append((cA, cD))
        return pairs

    mra_ch = np.empty_like(wpt_ch)

    # Cascade: A_J → A_{J-1} → … → A_0
    #   A_J   = iSWT(all coefficients)          = full reconstruction
    #   A_{j} = iSWT(keep details 1…j only)     = partial reconstruction
    #   A_0   = iSWT(no details, only VJ)       = smooth approximation S_J
    #
    #   D_j = A_j − A_{j-1}     (detail band j)
    #   S_J = A_0                (approximation band)

    # Start from the full reconstruction (keep all details)
    prev_approx = _iswt_channel(_build_coeffs(level - 1), wavelet_type)  # A_J

    for j in range(level, 0, -1):       # j = L … 1
        detail_idx = j - 1              # D_j is stored at mra_ch[j-1]
        # A_{j-1}: keep details D1…D_{j-1}, zero D_j…D_J
        # For j=1: _build_coeffs(-1) zeros ALL details → A_0 = S_J
        curr_approx = _iswt_channel(_build_coeffs(j - 2), wavelet_type)
        mra_ch[detail_idx] = prev_approx - curr_approx
        prev_approx = curr_approx

    # After the loop, prev_approx = A_0 = smooth approximation S_J
    mra_ch[level] = prev_approx

    return mra_ch


def modwtmra(wpt, wavelet_type, n_jobs=1):
    """
    Perform MODWTMRA (Multiresolution Analysis) using PyWavelets (pywt.iswt).

    Input/Output: (Bands x Samples x Channels). Order: [D1, ..., DJ, AJ].

    Notes
    -----
    Uses an O(L) cascade of iSWT calls per channel (L+1 total) instead of
    the naive O(L²) approach (one full iSWT per band).  This is substantially
    faster at higher decomposition levels.

    ``n_jobs`` defaults to 1 (sequential).  Pass a larger value only when
    this function is **not** already inside a parallel outer loop.
    """
    if wpt.ndim == 2:
        wpt = wpt[:, :, np.newaxis]

    n_bands, n_samples, n_channels = wpt.shape
    mra = np.empty_like(wpt)

    if n_channels == 1 or n_jobs == 1:
        for i in range(n_channels):
            mra[:, :, i] = _mra_single_channel(wpt[:, :, i], wavelet_type)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_mra_single_channel)(wpt[:, :, i], wavelet_type)
            for i in range(n_channels)
        )
        for i, ch_mra in enumerate(results):
            mra[:, :, i] = ch_mra

    return mra
