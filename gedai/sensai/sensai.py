import numpy as np
from mne.parallel import parallel_func
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from ..gedai.decompose import _clean_epochs


def subspace_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate the principal angles (in radians) between two subspaces.

    Parameters
    ----------
    A : np.ndarray
        Orthonormal basis for the first subspace (columns = basis vectors).
    B : np.ndarray
        Orthonormal basis for the second subspace.

    Returns
    -------
    angles_rad : np.ndarray
        Vector of principal angles in radians, sorted in ascending order.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    S = np.linalg.svd(A.T @ B, compute_uv=False)
    S_clipped = np.clip(S, -1.0, 1.0)
    angles_rad = np.arccos(S_clipped)
    return np.sort(angles_rad)


def _sensai_to_eigen(sensai_value, eigenvalues, percentile=95):
    """Convert a SENSAI score (0-105 scale) to an eigenvalue threshold.

    Parameters
    ----------
    sensai_value : float
        SENSAI threshold on the 0–105 scale.
    eigenvalues : np.ndarray
        All per-epoch GEVD eigenvalues (used to calibrate the scale).
    percentile : float
        Percentile of the log-eigenvalue distribution used as the reference
        point.  MATLAB uses **98** for EEG and **99** for MEG (the original
        Python code was 95).  Default is 95 (backward compatible).

    Returns
    -------
    eigenvalue : float
        The corresponding raw eigenvalue threshold.
    """
    all_diagonals = np.abs(eigenvalues.T.flatten())
    valid_diags = all_diagonals[all_diagonals > 0]
    if len(valid_diags) == 0:
        return 0.0

    # Auto-adjust scale back to EEGLAB microvolts for the non-invariant Math
    scale_factor = 1.0
    if np.median(valid_diags) < 1e-5:
        scale_factor = 1e12
        
    log_eig = np.log(valid_diags * scale_factor) + 100.0

    T1 = (105.0 - sensai_value) / 100.0
    threshold1 = T1 * np.percentile(log_eig, percentile)

    eigenvalue = np.exp(threshold1 - 100.0) / scale_factor
    return float(eigenvalue)


def _eigen_to_sensai(eigenvalue, eigenvalues, percentile=95):
    """Inverse of :func:`_sensai_to_eigen` — convert an eigenvalue threshold
    back to the SENSAI 0–105 scale (used for plotting).

    Parameters
    ----------
    eigenvalue : float
        Raw eigenvalue threshold.
    eigenvalues : np.ndarray
        All per-epoch GEVD eigenvalues.
    percentile : float
        Must match the value used in the forward conversion.  Default 95.
    """
    all_diagonals = np.abs(eigenvalues.T.flatten())
    valid_diags = all_diagonals[all_diagonals > 0]
    if len(valid_diags) == 0:
        return 0.0

    if eigenvalue <= 0:
        return 105.0

    scale_factor = 1.0
    if np.median(valid_diags) < 1e-5:
        scale_factor = 1e12
        
    log_eig = np.log(valid_diags * scale_factor) + 100.0
    threshold1 = np.log(eigenvalue * scale_factor) + 100.0
    percentile_val = np.percentile(log_eig, percentile)

    if percentile_val == 0:
        return 105.0

    T1 = threshold1 / percentile_val
    sensai_value = 105.0 - T1 * 100.0
    return float(sensai_value)


# ---------------------------------------------------------------------------
# Fast analytical SENSAI score — port of MATLAB clean_SENSAI.m
# ---------------------------------------------------------------------------

def _sensai_score_fast(
    threshold,
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
):
    """Compute the SENSAI score analytically from pre-computed GEVD results.

    This is a direct port of the MATLAB ``clean_SENSAI.m`` / ``SENSAI.m``
    pipeline.  Instead of reconstructing time series and re-computing
    covariances from data (as the legacy ``_sensai_score`` does), this
    function exploits the GEVD B-orthonormality property::

        V⁻¹ = Vᵀ @ B_reg

    to reconstruct signal and noise covariances analytically:

        V_bad_rows  = V_bad.T  @ B_reg          # (K_bad,  N)
        cov_noise   = V_bad_rows.T @ (V_bad_rows  * d_bad [:, None])

        V_good_rows = V_good.T @ B_reg          # (K_good, N)
        cov_signal  = V_good_rows.T @ (V_good_rows * d_good[:, None])

    No time series are reconstructed.  The GEVD is computed **once** per
    band (outside this function) and the results are reused across every
    threshold evaluation, yielding large speed gains especially for low-
    frequency bands with many samples per epoch.

    Parameters
    ----------
    threshold : float
        Eigenvalue threshold.  Components with ``|eigenvalue| >= threshold``
        are classified as artifacts.
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
        Per-epoch GEVD eigenvalues (ascending order, from ``scipy.linalg.eigh``).
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
        Per-epoch GEVD eigenvectors (columns = generalised eigenvectors).
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
        Regularised reference covariance matrix (the **B** matrix of the GEVD,
        already regularised before calling ``_fit_single_band``).
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
        Top ``n_pc`` eigenvectors of the reference covariance (template
        subspace), pre-computed once per band outside the threshold loop.
    n_pc : int
        Number of principal components used for subspace similarity.
    noise_multiplier : float
        Noise multiplier for the SENSAI score.

    Returns
    -------
    score : float
    signal_subspace_similarity : float
    noise_subspace_similarity : float
    """
    n_epochs = len(epochs_eigenvalues)
    signal_similarities = np.empty(n_epochs)
    noise_similarities = np.empty(n_epochs)

    for e in range(n_epochs):
        eigenvalues = epochs_eigenvalues[e]    # (n_ch,)
        eigenvectors = epochs_eigenvectors[e]  # (n_ch, n_ch)

        bad_mask = np.abs(eigenvalues) >= threshold
        good_mask = ~bad_mask

        # --- Noise covariance: artifact components ---
        if np.any(bad_mask):
            V_bad = eigenvectors[:, bad_mask]               # (n_ch, K_bad)
            V_bad_rows = V_bad.T @ reference_cov_reg        # (K_bad, n_ch)
            d_bad = np.abs(eigenvalues[bad_mask])           # (K_bad,)
            # cov_noise = Σ_k d_k * v_k^T v_k
            cov_noise = V_bad_rows.T @ (V_bad_rows * d_bad[:, np.newaxis])
            cov_noise = (cov_noise + cov_noise.T) * 0.5    # enforce symmetry
            _, evecs_noise = eigh(cov_noise)
            evecs_noise = evecs_noise[:, ::-1][:, :n_pc]
            angles = subspace_angles(evecs_noise, evecs_reference)
            noise_similarities[e] = np.prod(np.cos(angles))
        else:
            noise_similarities[e] = 0.0

        # --- Signal covariance: clean components ---
        if np.any(good_mask):
            V_good = eigenvectors[:, good_mask]             # (n_ch, K_good)
            V_good_rows = V_good.T @ reference_cov_reg     # (K_good, n_ch)
            d_good = np.abs(eigenvalues[good_mask])        # (K_good,)
            cov_signal = V_good_rows.T @ (V_good_rows * d_good[:, np.newaxis])
            cov_signal = (cov_signal + cov_signal.T) * 0.5
            _, evecs_signal = eigh(cov_signal)
            evecs_signal = evecs_signal[:, ::-1][:, :n_pc]
            angles = subspace_angles(evecs_signal, evecs_reference)
            signal_similarities[e] = np.prod(np.cos(angles))
        else:
            signal_similarities[e] = 0.0

    signal_subspace_similarity = 100.0 * np.mean(signal_similarities)
    noise_subspace_similarity  = 100.0 * np.mean(noise_similarities)
    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def _sensai_gridsearch_fast(
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
    eigen_thresholds,
    n_jobs=1,
):
    """Grid-search SENSAI threshold using the fast analytical scorer.

    Parameters
    ----------
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
    n_pc : int
    noise_multiplier : float
    eigen_thresholds : list of float
    n_jobs : int

    Returns
    -------
    best_threshold : float
    sensai_data : list of [threshold, score, signal_ss, noise_ss]
    """
    if n_jobs == 1:
        runs = [
            _sensai_score_fast(
                threshold,
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov_reg,
                evecs_reference,
                n_pc,
                noise_multiplier,
            )
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
            _sensai_score_fast, n_jobs, total=len(eigen_thresholds)
        )
        runs = parallel(
            p_fun(
                threshold,
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov_reg,
                evecs_reference,
                n_pc,
                noise_multiplier,
            )
            for threshold in eigen_thresholds
        )

    best_threshold = eigen_thresholds[np.argmax([run[0] for run in runs])]
    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sign(x):
    return -1 if x < 0 else 0 if x == 0 else 1

def _minimize_scalar_bounded(func, x1, x2, xtol=1e-5, maxiter=500):
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    import math
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x)
    num = 1
    fmin_data = (1, xf, fx)
    fu = float("inf")

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xtol / 3.0
    tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = _sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:
                golden = 1

        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = _sign(rat) + (rat == 0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xtol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            break

    fval = fx
    return xf, fval


def _sensai_optimize_fast(
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
    bounds,
    percentile=95,
):
    """Optimize SENSAI threshold using the fast analytical scorer.

    Parameters
    ----------
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
    n_pc : int
    noise_multiplier : float
    bounds : tuple of (float, float)
        SENSAI-scale bounds for the scalar minimisation.
    percentile : float
        Percentile of the log-eigenvalue distribution used for threshold
        conversion.  Use **98** for EEG, **99** for MEG (MATLAB defaults).
        Default is 95 (backward compatible).

    Returns
    -------
    eigen_threshold : float
    runs : list of [eigen_threshold, score, signal_ss, noise_ss]
    """
    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(
            sensai_threshold, epochs_eigenvalues, percentile=percentile
        )
        score, sig_ss, noise_ss = _sensai_score_fast(
            eigen_threshold,
            epochs_eigenvalues,
            epochs_eigenvectors,
            reference_cov_reg,
            evecs_reference,
            n_pc,
            noise_multiplier,
        )
        runs.append([eigen_threshold, score, sig_ss, noise_ss])
        return -score

    best_thresh, _ = _minimize_scalar_bounded(objective_function, bounds[0], bounds[1], xtol=0.01)

    eigen_threshold = _sensai_to_eigen(best_thresh, epochs_eigenvalues, percentile=percentile)
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs


# ---------------------------------------------------------------------------
# Legacy API — kept for backward compatibility (used by fit_epochs path
# via direct epoch objects; not called in the spectral hot path).
# ---------------------------------------------------------------------------

def _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier):
    """Compute the SENSAI score for a given threshold (legacy, epoch-based).

    .. deprecated::
        Prefer ``_sensai_score_fast`` which avoids time-series reconstruction.
    """
    epochs_data = epochs.get_data(verbose=False)
    epochs_clean, epochs_artefacts = _clean_epochs(
        epochs_data, reference_cov, threshold
    )

    reference_eigenvalues, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvalues = reference_eigenvalues[::-1]
    reference_eigenvectors = reference_eigenvectors[:, ::-1]
    reference_eigenvalues = reference_eigenvalues[:n_pc]
    reference_eigenvectors = reference_eigenvectors[:, :n_pc]

    signal_subspace_similarity = np.zeros((len(epochs_data), n_pc))
    noise_subspace_similarity = np.zeros((len(epochs_data), n_pc))

    for e, (epoch_clean_data, epoch_artefact_data) in enumerate(
        zip(epochs_clean, epochs_artefacts, strict=False)
    ):
        epoch_clean_covariance = np.cov(epoch_clean_data)
        _, epoch_clean_eigenvectors = eigh(epoch_clean_covariance)
        epoch_clean_eigenvectors = epoch_clean_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_clean_eigenvectors, reference_eigenvectors)
        signal_subspace_similarity[e] = np.prod(np.cos(angles))

        epoch_artefact_covariance = np.cov(epoch_artefact_data)
        _, epoch_artefact_eigenvectors = eigh(epoch_artefact_covariance)
        epoch_artefact_eigenvectors = epoch_artefact_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_artefact_eigenvectors, reference_eigenvectors)
        noise_subspace_similarity[e] = np.prod(np.cos(angles))

    signal_subspace_similarity = 100 * np.mean(signal_subspace_similarity)
    noise_subspace_similarity = 100 * np.mean(noise_subspace_similarity)

    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def _sensai_gridsearch(
    epochs, reference_cov, n_pc, noise_multiplier, eigen_thresholds, n_jobs=1
):
    """Legacy grid-search (epoch-based). Prefer ``_sensai_gridsearch_fast``."""
    if n_jobs == 1:
        runs = [
            _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
            _sensai_score, n_jobs, total=len(eigen_thresholds)
        )
        runs = parallel(
            p_fun(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        )

    best_threshold = eigen_thresholds[np.argmax([run[0] for run in runs])]

    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sensai_optimize(
    epochs, reference_cov, n_pc, noise_multiplier, epochs_eigenvalues, bounds
):
    """Legacy optimize (epoch-based). Prefer ``_sensai_optimize_fast``."""
    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
        score, signal_subspace_similarity, noise_subspace_similarity = _sensai_score(
            epochs,
            eigen_threshold,
            reference_cov,
            n_pc=n_pc,
            noise_multiplier=noise_multiplier,
        )
        runs.append(
            [
                eigen_threshold,
                score,
                signal_subspace_similarity,
                noise_subspace_similarity,
            ]
        )
        return -score

    result = minimize_scalar(objective_function, bounds=bounds, method="bounded")

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    sensai_threshold = result.x
    eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs


def score_sensai_basic(
    raw_clean_data: np.ndarray,
    raw_artifacts_data: np.ndarray,
    sfreq: float,
    duration: float,
    reference_cov: np.ndarray,
    noise_multiplier: float = 1.0,
    signal_type: str = "eeg",
):
    """
    Direct Python port of MATLAB's SENSAI_basic.m.
    Evaluates SENSAI over the global re-assembled broadband signals by parsing them
    into non-overlapping discrete epochs and evaluating subspace similarities directly.

    Parameters
    ----------
    raw_clean_data : np.ndarray, shape (n_channels, n_times)
        The fully cleaned data array (e.g. from ``raw_clean.get_data()``)
    raw_artifacts_data : np.ndarray, shape (n_channels, n_times)
        The artefactual noise data array (e.g. ``raw.get_data() - raw_clean.get_data()``)
    sfreq : float
        The sampling frequency.
    duration : float
        The epoch size in seconds (typically the same as broadband epoch size).
    reference_cov : np.ndarray, shape (n_channels, n_channels)
        The unregularized BEM leadfield or reference covariance matrix.
    noise_multiplier : float
        The multiplier for the noise term (default 1.0 to match MATLAB's final calculation).
    signal_type : str
        'eeg' or 'meg'.

    Returns
    -------
    sensai_score : float
        The composite analytical SENSAI score.
    signal_ss : float
        The global signal subspace similarity (0-100%%).
    noise_ss : float
        The global noise subspace similarity (0-100%%).
    mean_enova : float
        The mean Explained Noise Variance per epoch.
    enova_per_epoch : np.ndarray
        Array of ENOVA ratios evaluated per epoch.
    """
    n_ch, pnts = raw_clean_data.shape
    epoch_samples = int(round(sfreq * duration))
    num_epochs = pnts // epoch_samples

    if num_epochs == 0:
        raise ValueError("Data is too short to extract even a single epoch.")

    # Truncate to whole epochs
    valid_len = num_epochs * epoch_samples
    clean = raw_clean_data[:, :valid_len]
    artifacts = raw_artifacts_data[:, :valid_len]

    # Reshape into (n_epochs, n_channels, n_samples)
    clean_epoched = clean.reshape(n_ch, num_epochs, epoch_samples).transpose((1, 0, 2))
    artifacts_epoched = artifacts.reshape(n_ch, num_epochs, epoch_samples).transpose((1, 0, 2))

    # Regularize reference covariance identically to GEVD
    reg_lambda = 0.05
    reg_val = np.trace(reference_cov) / n_ch
    ref_cov_reg = (1 - reg_lambda) * reference_cov + reg_lambda * reg_val * np.eye(n_ch)

    if signal_type.lower() == "meg":
        refcov_top_pcs = 5
    else:
        refcov_top_pcs = 3

    top_pcs = 3

    # Top PCs of Reference Covariance
    evals_ref, evecs_ref = eigh(ref_cov_reg)
    evecs_ref = evecs_ref[:, ::-1][:, :refcov_top_pcs]

    signal_similarities = np.zeros(num_epochs)
    noise_similarities = np.zeros(num_epochs)
    enova_per_epoch = np.zeros(num_epochs)

    for e in range(num_epochs):
        c_ep = clean_epoched[e]
        a_ep = artifacts_epoched[e]
        
        # Signal Subspace
        c_cov = np.cov(c_ep)
        evals_c, evecs_c = eigh(c_cov)
        evecs_c = evecs_c[:, ::-1][:, :top_pcs]
        sig_angles = subspace_angles(evecs_c, evecs_ref)
        signal_similarities[e] = np.prod(np.cos(sig_angles))

        # Noise Subspace
        a_cov = np.cov(a_ep)
        evals_a, evecs_a = eigh(a_cov)
        evecs_a = evecs_a[:, ::-1][:, :top_pcs]
        noise_angles = subspace_angles(evecs_a, evecs_ref)
        noise_similarities[e] = np.prod(np.cos(noise_angles))

        # ENOVA
        orig_ep = c_ep + a_ep
        var_orig = np.var(orig_ep)
        var_noise = np.var(a_ep)
        enova_per_epoch[e] = var_noise / var_orig if var_orig > 0 else 0.0

    sig_ss = 100.0 * np.mean(signal_similarities)
    noise_ss = 100.0 * np.mean(noise_similarities)
    mean_enova = np.mean(enova_per_epoch)
    
    sensai_score = sig_ss - noise_multiplier * noise_ss

    return sensai_score, sig_ss, noise_ss, mean_enova, enova_per_epoch
