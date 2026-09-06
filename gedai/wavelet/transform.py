import numpy as np
from mne.parallel import parallel_func

from ..utils._checks import _check_n_jobs
from ..utils._docs import fill_doc
from ._modwt import modwt, modwtmra


def _process_epoch_wavelet(epoch_data, wavelet, level):
    """Process a single epoch with wavelet transform.

    Parameters
    ----------
    epoch_data : np.ndarray
        Single epoch data with shape (n_channels, n_times).
    wavelet : str
        The type of wavelet to use.
    level : int
        The level of decomposition.

    Returns
    -------
    transformed_epoch : np.ndarray
        Transformed epoch with shape (n_channels, level+1, n_times).
    """
    n_channels, n_times = epoch_data.shape
    transformed_epoch = np.zeros((n_channels, level + 1, n_times))

    for c, ch_data in enumerate(epoch_data):
        coeffs = modwt(ch_data, wavelet, level)
        modwtmra_data = modwtmra(coeffs, wavelet)
        modwtmra_data = np.squeeze(modwtmra_data, axis=-1)
        transformed_epoch[c, :, :] = modwtmra_data

    return transformed_epoch


@fill_doc
def epochs_to_wavelet(data, sfreq, wavelet, level, n_jobs=None, verbose=None):
    """Apply MODWT to each epoch in the epochs object.

    Parameters
    ----------
    data : np.ndarray
        The epochs data with shape (n_epochs, n_channels, n_times).
    sfreq : float
        The sampling frequency of the data.
    wavelet : str
        The type of wavelet to use (e.g., 'haar', 'db4', etc.).
    level : int
        The level of decomposition. If 0, no decomposition is performed.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    transformed_data : np.ndarray
        The transformed data with shape (n_epochs, n_channels, level+1, n_times).
    freq_bands : list of tuple
        Frequency bands for each component, ordered to match transformed_data.
    levels : int
        The actual decomposition level used.
    """
    n_jobs = _check_n_jobs(n_jobs)
    n_epochs, n_channels, n_times = data.shape

    if level == 0:
        # No wavelet decomposition - return original data as single band
        transformed_data = data[:, :, np.newaxis, :]
        freq_bands = [(0, sfreq / 2)]
        levels = 0
    else:
        freq_bands = []

        # Approximation (index 0): lowest frequencies
        freq_bands.append((0, sfreq / (2 ** (level + 1))))

        # Details (indices 1 to level): from coarse to fine
        for i in range(level, 0, -1):
            fmin = sfreq / (2 ** (i + 1))
            fmax = sfreq / (2**i)
            freq_bands.append((fmin, fmax))

        # Parallelize the wavelet transform across epochs
        if n_jobs == 1:
            # Sequential processing
            transformed_data = np.zeros((n_epochs, n_channels, level + 1, n_times))
            for e, epoch in enumerate(data):
                transformed_data[e] = _process_epoch_wavelet(epoch, wavelet, level)
        else:
            # Parallel processing using MNE's parallel_func
            parallel, p_fun, n_jobs = parallel_func(
                _process_epoch_wavelet, n_jobs, total=n_epochs, verbose=verbose
            )
            transformed_epochs = parallel(
                p_fun(epoch, wavelet, level) for epoch in data
            )
            transformed_data = np.array(transformed_epochs)

        levels = level

    return transformed_data, freq_bands, levels


def compute_wavelet_level(
    sfreq: float,
    lowcut_hz: float = 0.5,
    n_times: int | None = None,
    wavelet_low_cutoff: float | None = None,
    cycles_per_wavelet: int | None = None,
) -> int:
    """Compute number of wavelet decomposition levels matching MATLAB GEDAI.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    lowcut_hz : float
        Low cutoff frequency in Hz (default 0.5 Hz).
    n_times : int | None
        Number of time samples in data (optional, used to bound max levels).
    wavelet_low_cutoff : float | None
        Alias for lowcut_hz.
    cycles_per_wavelet : int | None
        Number of cycles per wavelet band (for adaptive windowing).

    Returns
    -------
    level : int
        Recommended wavelet decomposition level.
    """
    if wavelet_low_cutoff is not None:
        lowcut_hz = wavelet_low_cutoff
    lowcut_hz = max(float(lowcut_hz), 0.01)
    ideal = int(np.ceil(np.log2(sfreq / lowcut_hz)))
    if n_times is not None:
        if cycles_per_wavelet is not None and cycles_per_wavelet > 0:
            max_possible = max(
                1, int(np.floor(np.log2(max(1.0, n_times / cycles_per_wavelet)))) - 1
            )
        else:
            max_possible = max(1, int(np.floor(np.log2(n_times))))
        return max(2, min(ideal, max_possible))
    return max(6, ideal)


def get_modwt_band_limits(sfreq: float, n_bands: int) -> list[tuple[float, float]]:
    """Frequency limits for each MODWT band (index 0 = finest/highest detail band).

    Band f: [sfreq / 2^(f+2), sfreq / 2^(f+1)]
    Approximation (last band): [0.0, sfreq / 2^n_bands]
    """
    limits = []
    for f in range(n_bands - 1):
        lo = sfreq / (2 ** (f + 2))
        hi = sfreq / (2 ** (f + 1))
        limits.append((lo, hi))
    limits.append((0.0, sfreq / (2**n_bands)))
    return limits


def _modwt_haar_single_band_core_torch(data_T, level: int, target_band: int):
    """Core single-band forward and inverse Haar MODWT in PyTorch."""
    import torch

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    n_bands = level + 1
    current_approx = data_T
    max_level_needed = min(target_band, level)
    target_coefs = None

    for j in range(1, max_level_needed + 1):
        step = 2 ** (j - 1)
        shifted_approx = torch.roll(current_approx, shifts=step, dims=0)
        if j == target_band:
            target_coefs = (shifted_approx - current_approx) * inv_sqrt2
        else:
            current_approx = (current_approx + shifted_approx) * inv_sqrt2

    if target_band == n_bands:
        target_coefs = current_approx

    current_recon = target_coefs.clone()
    if target_band == n_bands:
        for j in range(level, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = torch.roll(current_recon, shifts=-step, dims=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)
    else:
        j = target_band
        step = 2 ** (j - 1)
        D_shifted = torch.roll(current_recon, shifts=-step, dims=0)
        current_recon = 0.5 * inv_sqrt2 * (D_shifted - current_recon)
        for j in range(target_band - 1, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = torch.roll(current_recon, shifts=-step, dims=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)

    return current_recon


def _modwt_haar_single_band_core_numpy(data_T: np.ndarray, level: int, target_band: int) -> np.ndarray:
    """Core single-band forward and inverse Haar MODWT in NumPy."""
    dtype = data_T.dtype
    inv_sqrt2 = dtype.type(1.0 / np.sqrt(2.0))
    half_inv_sqrt2 = dtype.type(0.5 / np.sqrt(2.0))
    n_bands = level + 1
    current_approx = data_T
    max_level_needed = min(target_band, level)
    target_coefs = None

    for j in range(1, max_level_needed + 1):
        step = 2 ** (j - 1)
        shifted_approx = np.roll(current_approx, step, axis=0)
        if j == target_band:
            target_coefs = (shifted_approx - current_approx) * inv_sqrt2
        else:
            current_approx = (current_approx + shifted_approx) * inv_sqrt2

    if target_band == n_bands:
        target_coefs = current_approx

    current_recon = target_coefs.copy()
    if target_band == n_bands:
        for j in range(level, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = half_inv_sqrt2 * (current_recon + A_shifted)
    else:
        j = target_band
        step = 2 ** (j - 1)
        D_shifted = np.roll(current_recon, -step, axis=0)
        current_recon = half_inv_sqrt2 * (D_shifted - current_recon)
        for j in range(target_band - 1, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = half_inv_sqrt2 * (current_recon + A_shifted)

    return current_recon


def _modwt_haar_single_band(
    data_T: np.ndarray,
    level: int,
    band_idx: int,
    chunk_size: int = 50000,
    engine: str = "auto",
) -> np.ndarray:
    """Haar MODWT single-band reconstruction matching MATLAB modwt_single_band.m.

    Uses circular shifts matching MATLAB circshift with stateful overlap-save
    chunking (matching stateful_modwt_single_band.m). Bounded memory ceiling
    and accelerated via PyTorch when available.

    Parameters
    ----------
    data_T : (n_times, n_ch) float32 or float64
        Samples x channels.
    level : int
        Decomposition level.
    band_idx : int
        0-indexed band (0 = finest detail D1, ..., level = approximation A_J).
    chunk_size : int
        Chunk size for overlap-save processing (default 50,000 samples).
    engine : str
        Computation engine ('auto', 'torch', or 'numpy').

    Returns
    -------
    (n_ch, n_times) ndarray
        Reconstructed band signal in time domain.
    """
    from ..utils._torch_backend import has_torch

    target_band = band_idx + 1
    use_torch = (engine == "torch" or (engine == "auto" and has_torch())) and has_torch()

    num_samples, num_channels = data_T.shape

    if use_torch:
        import torch

        is_numpy = isinstance(data_T, np.ndarray)
        if is_numpy:
            tensor_data = torch.from_numpy(data_T)
        else:
            tensor_data = data_T

        if num_samples <= chunk_size:
            recon = _modwt_haar_single_band_core_torch(tensor_data, level, target_band).T
        else:
            P = 2 ** level
            band_signal = torch.empty(
                (num_samples, num_channels),
                dtype=tensor_data.dtype,
                device=tensor_data.device,
            )
            num_chunks = int(np.ceil(num_samples / chunk_size))

            for chunk in range(num_chunks):
                c_start = chunk * chunk_size
                c_end = min(num_samples, (chunk + 1) * chunk_size)
                c_len = c_end - c_start

                # 1. Prepend buffer (wrap around if needed)
                if c_start >= P:
                    prepend = tensor_data[c_start - P : c_start]
                else:
                    needed = P - c_start
                    wrap_end = tensor_data[-needed:]
                    if c_start > 0:
                        prepend = torch.cat([wrap_end, tensor_data[:c_start]], dim=0)
                    else:
                        prepend = wrap_end

                # 2. Append buffer (wrap around if needed)
                if c_end + P <= num_samples:
                    append = tensor_data[c_end : c_end + P]
                else:
                    needed = P - (num_samples - c_end)
                    wrap_start = tensor_data[:needed]
                    if c_end < num_samples:
                        append = torch.cat([tensor_data[c_end:], wrap_start], dim=0)
                    else:
                        append = wrap_start

                padded_block = torch.cat([prepend, tensor_data[c_start:c_end], append], dim=0)
                recon_padded = _modwt_haar_single_band_core_torch(padded_block, level, target_band)
                band_signal[c_start:c_end] = recon_padded[P : P + c_len]

            recon = band_signal.T

        return recon.cpu().numpy() if is_numpy else recon

    # NumPy path
    data_T_np = np.asarray(data_T)
    if num_samples <= chunk_size:
        return _modwt_haar_single_band_core_numpy(data_T_np, level, target_band).T

    P = 2 ** level
    band_signal = np.empty((num_samples, num_channels), dtype=data_T_np.dtype)
    num_chunks = int(np.ceil(num_samples / chunk_size))

    for chunk in range(num_chunks):
        c_start = chunk * chunk_size
        c_end = min(num_samples, (chunk + 1) * chunk_size)
        c_len = c_end - c_start

        if c_start >= P:
            prepend = data_T_np[c_start - P : c_start]
        else:
            needed = P - c_start
            wrap_end = data_T_np[-needed:]
            if c_start > 0:
                prepend = np.concatenate([wrap_end, data_T_np[:c_start]], axis=0)
            else:
                prepend = wrap_end

        if c_end + P <= num_samples:
            append = data_T_np[c_end : c_end + P]
        else:
            needed = P - (num_samples - c_end)
            wrap_start = data_T_np[:needed]
            if c_end < num_samples:
                append = np.concatenate([data_T_np[c_end:], wrap_start], axis=0)
            else:
                append = wrap_start

        padded_block = np.concatenate([prepend, data_T_np[c_start:c_end], append], axis=0)
        recon_padded = _modwt_haar_single_band_core_numpy(padded_block, level, target_band)
        band_signal[c_start:c_end] = recon_padded[P : P + c_len]

    return band_signal.T


def _apply_wavelet_highpass_prefilter(
    data: np.ndarray, sfreq: float, lowcut_hz: float = 0.5, engine: str = "auto"
) -> np.ndarray:
    """Remove sub-lowcut_hz slow drift using continuous Haar MODWT decomposition.

    Decomposes the continuous multichannel signal into Haar wavelet bands and
    subtracts all sub-lowcut_hz bands before broadband GED covariance estimation,
    preventing slow drift and DC offsets from biasing broadband covariance.
    Matches MATLAB GEDAI.m lines 520-603.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Continuous multichannel data.
    sfreq : float
        Sampling frequency in Hz.
    lowcut_hz : float
        Highpass cutoff frequency (default 0.5 Hz).
    engine : str
        Computation engine ('auto', 'torch', or 'numpy').

    Returns
    -------
    filtered_data : np.ndarray, shape (n_channels, n_times)
        Data with sub-lowcut_hz drift subtracted.
    """
    if lowcut_hz is None or lowcut_hz <= 0:
        return data

    n_times = data.shape[1]
    hp_wavelet_levels = int(
        np.ceil(np.log2(sfreq / max(0.01, min(0.1, lowcut_hz))) - 1)
    )
    hp_wavelet_levels = max(hp_wavelet_levels, 3)
    hp_wavelet_levels = min(hp_wavelet_levels, int(np.floor(np.log2(n_times))))
    n_bands_hp = hp_wavelet_levels + 1

    bands_to_hp_zero = [
        j for j in range(n_bands_hp) if (sfreq / (2 ** (j + 1))) <= lowcut_hz
    ]
    if not bands_to_hp_zero:
        return data

    low_freq_noise = np.zeros_like(data)
    data_T = data.T
    for b in bands_to_hp_zero:
        low_freq_noise += _modwt_haar_single_band(
            data_T, hp_wavelet_levels, b, engine=engine
        )

    return (data - low_freq_noise).astype(data.dtype)
