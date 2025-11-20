import numpy as np
import pywt


def modwt(data, wavelet_type, level):
    """
    PerformsMODWT using PyWavelets (pywt.swt), matching MATLAB coefficient order.

    Input: data (Samples x Channels)
    Output: (Bands x Samples x Channels). Order: [W1, ..., WJ, VJ].
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape
    n_bands = level + 1

    wpt = np.zeros((n_bands, n_samples, n_channels))

    for i in range(n_channels):
        # Use norm=True for MODWT equivalence.
        # pywt.swt output structure: [(cAJ, cDJ), ..., (cA1, cD1)]
        try:
            coeffs = pywt.swt(data[:, i], wavelet_type, level=level, norm=True, axis=0)
        except TypeError:
            coeffs = pywt.swt(data[:, i], wavelet_type, level=level, axis=0)

        # Reformat to MATLAB convention [W1, ..., WJ, VJ]

        # 1. Extract VJ (cAJ, Approximation at final level)
        VJ = coeffs[0][0]

        # 2. Extract Details W1 to WJ (cD1 to cDJ)
        details = [c[1] for c in coeffs]  # [cDJ, ..., cD1]
        details.reverse()  # [cD1, ..., cDJ]

        # 3. Combine
        matlab_order_coeffs = details + [VJ]
        wpt[:, :, i] = np.array(matlab_order_coeffs)

    return wpt


def modwtmra(wpt, wavelet_type):
    """
    Perform MODWTMRA (Multiresolution Analysis) using PyWavelets (pywt.iswt).

    Input/Output: (Bands x Samples x Channels). Order: [D1, ..., DJ, AJ].
    """
    if wpt.ndim == 2:
        wpt = wpt[:, :, np.newaxis]

    n_bands, n_samples, n_channels = wpt.shape
    level = n_bands - 1
    mra = np.zeros_like(wpt)

    for i in range(n_channels):
        channel_coeffs = wpt[:, :, i]

        # MRA: Isolate one band (j) and reconstruct
        for j in range(n_bands):
            isolated_swt_structure = []

            # Iterate from highest level (J) down to 1 (to match pywt order)
            for k in range(level, 0, -1):
                # Default to zero arrays
                cA_k = np.zeros_like(channel_coeffs[0])
                cD_k = np.zeros_like(channel_coeffs[0])

                # If we are reconstructing detail band Wk, set cD_k
                if j == k - 1:
                    cD_k = channel_coeffs[j]

                # If we are at the highest level and reconstructing VJ, set cA_k
                if k == level and j == level:
                    cA_k = channel_coeffs[j]

                isolated_swt_structure.append((cA_k, cD_k))

            # 2. Inverse SWT (ISWT/IMODWT).
            try:
                mra[j, :, i] = pywt.iswt(
                    isolated_swt_structure, wavelet_type, norm=True
                )
            except TypeError:
                mra[j, :, i] = pywt.iswt(isolated_swt_structure, wavelet_type)

    return mra
