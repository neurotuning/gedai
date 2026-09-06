import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from ..utils._checks import ensure_int
from ..utils._torch_backend import precompute_gevd_torch, resolve_engine


def subspace_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate the principal angles (in radians) between two subspaces.

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
    # Ensure inputs are float64
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    # Compute the SVD of A.T @ B
    S = np.linalg.svd(A.T @ B, compute_uv=False)

    # Clamp singular values to [-1, 1] to avoid invalid acos input
    S_clipped = np.clip(S, -1.0, 1.0)

    # Compute principal angles in radians
    angles_rad = np.arccos(S_clipped)

    # Return sorted angles
    return np.sort(angles_rad)


def subspace_similarity(A: np.ndarray, B: np.ndarray, n_pc: int | None = 3) -> float:
    """Compute rank-normalized volumetric subspace similarity.

    When n_pc == 3, uses the rank-normalized 3D volumetric scale:
    - rank 0: 0.0
    - rank 1: s_1^3
    - rank 2: s_1 * s_2 * ((s_1 + s_2) / 2)
    - rank >= 3: s_1 * s_2 * s_3

    Parameters
    ----------
    A, B : np.ndarray
        (n, k) matrices whose columns span the subspaces.
    n_pc : int | None
        Target principal component dimension (default 3).

    Returns
    -------
    similarity : float in [0, 1]
    """
    if n_pc is not None:
        n_pc = ensure_int(n_pc, "n_pc")
        if n_pc < 1:
            raise ValueError(f"n_pc must be >= 1, got {n_pc!r}.")

    S = np.linalg.svd(A.T @ B, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    k = len(S)
    if k == 0:
        return 0.0

    if n_pc == 3:
        if k == 1:
            return float(S[0] ** 3)
        elif k == 2:
            return float(S[0] * S[1] * (0.5 * (S[0] + S[1])))
        else:
            return float(S[0] * S[1] * S[2])
    elif n_pc is not None and n_pc > 0:
        S_sub = S[: min(n_pc, len(S))]
        return float(np.prod(S_sub))
    return float(np.prod(S))


def _prescan_meg_artifact_spectrum(data: np.ndarray, reference_cov: np.ndarray) -> int:
    """Prescan GEVD artifact spectrum to adaptively choose n_pc (2 or 3) for MEG.

    Parameters
    ----------
    data : np.ndarray
        Raw or epoched MEG data.
    reference_cov : np.ndarray
        Reference leadfield covariance matrix.

    Returns
    -------
    n_pc : int
        Recommended number of principal components (2 or 3).
    """
    if data.ndim == 3:
        n_ch = data.shape[1]
        data_2d = data.transpose(1, 0, 2).reshape(n_ch, -1)
    else:
        data_2d = data
    cov_prescan = np.cov(data_2d)
    cov_prescan = (cov_prescan + cov_prescan.T) * 0.5
    evals_prescan, _ = eigh(cov_prescan, reference_cov, check_finite=False)
    evals_desc = np.sort(np.abs(evals_prescan))[::-1]
    med_val = np.median(evals_desc)
    norm_evals = evals_desc / med_val if med_val > 0 else evals_desc
    n_ch = reference_cov.shape[0]
    if len(norm_evals) > 2 and (
        norm_evals[2] > 200
        or (norm_evals[1] / max(norm_evals[0], 1e-12) > 0.20 and norm_evals[0] < 12000)
    ):
        return min(3, n_ch)
    else:
        return min(2, n_ch)


def _compute_default_n_pc(
    reference_cov: np.ndarray,
    signal_type: str = "eeg",
    data: np.ndarray | None = None,
) -> int:
    """Compute default number of PCs for SENSAI (2/3 for MEG, 3 for EEG).

    Parameters
    ----------
    reference_cov : np.ndarray
        Reference leadfield covariance matrix.
    signal_type : str
        The detected signal type ('eeg' or 'meg').
    data : np.ndarray | None
        Input signal data. If MEG and data is provided, an artifact spectrum prescan
        is performed.

    Returns
    -------
    n_pc : int
        Default number of principal components to retain.
    """
    n_ch = reference_cov.shape[0]
    if signal_type == "meg":
        if data is not None:
            return _prescan_meg_artifact_spectrum(data, reference_cov)
        return min(2, n_ch)
    return min(3, n_ch)


def _sensai_to_eigen(sensai_value, eigenvalues, percentile=98):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    T1 = (105 - sensai_value) / 100
    threshold1 = T1 * np.percentile(log_eig_val_all, percentile)
    eigenvalue = np.exp(threshold1 - 100)
    return eigenvalue


def _eigen_to_sensai(eigenvalue, eigenvalues, percentile=98):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    threshold1 = np.log(eigenvalue) + 100
    T1 = threshold1 / np.percentile(log_eig_val_all, percentile)
    sensai_value = 105 - T1 * 100
    return sensai_value


def _precompute_gevd(
    epochs_data: np.ndarray, reference_cov: np.ndarray, engine: str = "numpy"
):
    """Precompute generalized eigenvalue decomposition across all epochs.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    reference_cov : np.ndarray, shape (n_channels, n_channels)
    engine : str, default 'numpy'
        Computation engine ('numpy', 'torch', or 'auto').

    Returns
    -------
    all_eval : np.ndarray, shape (n_epochs, n_channels)
    all_evec : np.ndarray, shape (n_epochs, n_channels, n_channels)
    """
    resolved = resolve_engine(engine)
    if resolved == "torch":
        return precompute_gevd_torch(epochs_data, reference_cov)

    n_ep, n_ch, n_times = epochs_data.shape
    if n_times < 2:
        raise ValueError(
            "epochs_data must contain at least 2 time points per epoch to "
            "compute covariance."
        )
    all_eval = np.zeros((n_ep, n_ch), dtype=np.float64)
    all_evec = np.zeros((n_ep, n_ch, n_ch), dtype=np.float64)

    # Vectorized covariance: batch all epochs at once via einsum
    centered = epochs_data - epochs_data.mean(axis=-1, keepdims=True)
    all_covs = np.einsum("eij,ekj->eik", centered, centered)
    all_covs *= 1.0 / (n_times - 1)
    for i in range(n_ep):
        evals, evecs = eigh(all_covs[i], reference_cov, check_finite=False)
        all_eval[i] = evals
        all_evec[i] = evecs

    return all_eval, all_evec


def _sensai_score_loop(
    abs_evals: np.ndarray,
    all_VR: np.ndarray,
    template: np.ndarray,
    threshold: float,
    n_pc: int = 3,
    signal_type: str = "eeg",
    all_evec: np.ndarray | None = None,
    reference_cov: np.ndarray | None = None,
):
    """NumPy optimized loop for fast power-weighted SENSAI optimization scoring."""
    n_ep, n_ch = abs_evals.shape
    sig_sims = np.zeros(n_ep, dtype=np.float64)
    noi_sims = np.zeros(n_ep, dtype=np.float64)
    eye_n_pc = np.eye(n_ch)[:, :n_pc]
    empty_noi_sim = abs(float(np.linalg.det(eye_n_pc.T @ template)))

    for e in range(n_ep):
        evals_e = abs_evals[e]
        VR_e = all_VR[e]
        bad_mask = evals_e >= threshold
        num_bad = int(np.sum(bad_mask))

        # --- Artifact noise subspace ---
        if signal_type == "meg":
            if num_bad > 0:
                P_bad = VR_e[:, bad_mask]
                Q_bad, _ = np.linalg.qr(P_bad)
                s = np.linalg.svd(Q_bad.T @ template, compute_uv=False)
                noi_sims[e] = float(np.sum(s**6))
            else:
                noi_sims[e] = 0.0
        else:
            if num_bad >= n_pc:
                VR_bad = VR_e[:, bad_mask]
                d_bad = evals_e[bad_mask, None]
                Y1_n = VR_bad @ (d_bad * (VR_bad.T @ template))
                Q1_n, _ = np.linalg.qr(Y1_n)
                Y2_n = VR_bad @ (d_bad * (VR_bad.T @ Q1_n))
                basis_n, _ = np.linalg.qr(Y2_n)
                noi_sims[e] = abs(float(np.linalg.det(basis_n[:, :n_pc].T @ template)))
            elif num_bad > 0:
                VR_bad = VR_e[:, bad_mask]
                d_bad = evals_e[bad_mask]
                cov_noise = (VR_bad * d_bad) @ VR_bad.T
                cov_noise = (cov_noise + cov_noise.T) * 0.5
                if np.max(np.abs(cov_noise)) < 1e-12:
                    Y1_n = eye_n_pc
                else:
                    Y1_n = cov_noise @ template
                Q1_n, _ = np.linalg.qr(Y1_n)
                basis_n, _ = np.linalg.qr(cov_noise @ Q1_n)
                noi_sims[e] = abs(float(np.linalg.det(basis_n[:, :n_pc].T @ template)))
            else:
                noi_sims[e] = empty_noi_sim

        # --- Clean signal subspace ---
        good_mask = ~bad_mask
        num_good = int(np.sum(good_mask))
        if num_good >= n_pc:
            VR_good = VR_e[:, good_mask]
            d_good = evals_e[good_mask, None]
            Y1_s = VR_good @ (d_good * (VR_good.T @ template))
            Q1_s, _ = np.linalg.qr(Y1_s)
            Y2_s = VR_good @ (d_good * (VR_good.T @ Q1_s))
            basis_s, _ = np.linalg.qr(Y2_s)
            sig_sims[e] = abs(float(np.linalg.det(basis_s[:, :n_pc].T @ template)))
        elif num_good > 0:
            VR_good = VR_e[:, good_mask]
            d_good = evals_e[good_mask]
            cov_signal = (VR_good * d_good) @ VR_good.T
            cov_signal = (cov_signal + cov_signal.T) * 0.5
            if np.max(np.abs(cov_signal)) < 1e-12:
                Y1_s = eye_n_pc
            else:
                Y1_s = cov_signal @ template
            Q1_s, _ = np.linalg.qr(Y1_s)
            basis_s, _ = np.linalg.qr(cov_signal @ Q1_s)
            sig_sims[e] = abs(float(np.linalg.det(basis_s[:, :n_pc].T @ template)))
        else:
            sig_sims[e] = empty_noi_sim

    return sig_sims, noi_sims


def _sensai_score_torch(
    abs_evals_t,
    all_VR_t,
    template_t,
    threshold: float,
    n_pc: int = 3,
    signal_type: str = "eeg",
):
    """Batched PyTorch implementation for fast power-weighted SENSAI scoring."""
    import torch

    n_ep, n_ch = abs_evals_t.shape
    eye_n_pc = torch.eye(n_ch, dtype=template_t.dtype, device=template_t.device)[:, :n_pc]
    empty_noi_sim = float(torch.abs(torch.linalg.det(eye_n_pc.T @ template_t)))

    # --- Clean signal subspace ---
    w_good = torch.where(abs_evals_t < threshold, abs_evals_t, 0.0).unsqueeze(-1)
    num_good = (abs_evals_t < threshold).sum(dim=1)
    Y1_s = torch.bmm(all_VR_t, w_good * torch.matmul(all_VR_t.transpose(1, 2), template_t))
    Q1_s = torch.linalg.qr(Y1_s).Q
    Y2_s = torch.bmm(all_VR_t, w_good * torch.bmm(all_VR_t.transpose(1, 2), Q1_s))
    basis_s = torch.linalg.qr(Y2_s).Q
    overlap_s = torch.matmul(basis_s[:, :, :n_pc].transpose(1, 2), template_t)
    sig_sims = torch.abs(torch.linalg.det(overlap_s))
    sig_sims = torch.where(num_good > 0, sig_sims, empty_noi_sim)

    fallback_good = (num_good > 0) & (num_good < n_pc)
    if fallback_good.any():
        for e in torch.where(fallback_good)[0]:
            mask_e = abs_evals_t[e] < threshold
            VR_good = all_VR_t[e, :, mask_e]
            d_good = abs_evals_t[e, mask_e]
            cov_s = (VR_good * d_good) @ VR_good.T
            cov_s = (cov_s + cov_s.T) * 0.5
            if torch.max(torch.abs(cov_s)) < 1e-12:
                Y1_s_e = eye_n_pc
            else:
                Y1_s_e = cov_s @ template_t
            Q1_s_e = torch.linalg.qr(Y1_s_e).Q
            basis_s_e = torch.linalg.qr(cov_s @ Q1_s_e).Q
            sig_sims[e] = torch.abs(torch.linalg.det(basis_s_e[:, :n_pc].T @ template_t))

    # --- Artifact noise subspace ---
    if signal_type == "meg":
        noi_sims = torch.zeros(n_ep, dtype=template_t.dtype, device=template_t.device)
        bad_mask = abs_evals_t >= threshold
        for e in range(n_ep):
            mask_e = bad_mask[e]
            if mask_e.any():
                P_bad = all_VR_t[e, :, mask_e]
                Q_bad = torch.linalg.qr(P_bad).Q
                s = torch.linalg.svdvals(Q_bad.T @ template_t)
                noi_sims[e] = torch.sum(s**6)
    else:
        w_bad = torch.where(abs_evals_t >= threshold, abs_evals_t, 0.0).unsqueeze(-1)
        num_bad = (abs_evals_t >= threshold).sum(dim=1)
        Y1_n = torch.bmm(all_VR_t, w_bad * torch.matmul(all_VR_t.transpose(1, 2), template_t))
        Q1_n = torch.linalg.qr(Y1_n).Q
        Y2_n = torch.bmm(all_VR_t, w_bad * torch.bmm(all_VR_t.transpose(1, 2), Q1_n))
        basis_n = torch.linalg.qr(Y2_n).Q
        overlap_n = torch.matmul(basis_n[:, :, :n_pc].transpose(1, 2), template_t)
        noi_sims = torch.abs(torch.linalg.det(overlap_n))
        noi_sims = torch.where(num_bad > 0, noi_sims, empty_noi_sim)

        fallback_bad = (num_bad > 0) & (num_bad < n_pc)
        if fallback_bad.any():
            for e in torch.where(fallback_bad)[0]:
                mask_e = abs_evals_t[e] >= threshold
                VR_bad = all_VR_t[e, :, mask_e]
                d_bad = abs_evals_t[e, mask_e]
                cov_n = (VR_bad * d_bad) @ VR_bad.T
                cov_n = (cov_n + cov_n.T) * 0.5
                if torch.max(torch.abs(cov_n)) < 1e-12:
                    Y1_n_e = eye_n_pc
                else:
                    Y1_n_e = cov_n @ template_t
                Q1_n_e = torch.linalg.qr(Y1_n_e).Q
                basis_n_e = torch.linalg.qr(cov_n @ Q1_n_e).Q
                noi_sims[e] = torch.abs(torch.linalg.det(basis_n_e[:, :n_pc].T @ template_t))

    return sig_sims, noi_sims


def _sensai_score_from_gevd(
    all_eigenvalues: np.ndarray,
    all_eigenvectors: np.ndarray,
    reference_cov: np.ndarray,
    reference_eigenvectors: np.ndarray,
    threshold: float,
    n_pc: int = 3,
    noise_multiplier: float = 3.0,
    signal_type: str = "eeg",
) -> tuple[float, float, float]:
    """Fast SENSAI score using cached GEVD and power-weighted 2-step QR subspace."""
    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])

    # Precompute reference_cov @ eigenvectors for all epochs at once
    all_VR = np.einsum("ij,ejk->eik", reference_cov, all_eigenvectors)
    abs_evals = np.abs(all_eigenvalues)

    sig_sims, noi_sims = _sensai_score_loop(
        np.ascontiguousarray(abs_evals),
        np.ascontiguousarray(all_VR),
        template,
        threshold,
        n_pc=n_pc,
        signal_type=signal_type,
        all_evec=all_eigenvectors,
        reference_cov=reference_cov,
    )

    sig_sim = float(np.mean(sig_sims) * 100.0)
    noi_sim = float(np.mean(noi_sims) * 100.0)
    score = float(sig_sim - (noise_multiplier * noi_sim))

    return score, sig_sim, noi_sim


def _find_changepoint(y: np.ndarray, smooth_window: int = 6) -> int | None:
    """Detect the first changepoint in a 1-D signal's gradient mean.

    Equivalent to MATLAB's `findchangepts(diff(smoothdata(y, "movmean", 6)),
    Statistic="mean", MaxNumChanges=2)`.

    Used as a safeguard against degenerate SENSAI curves.
    """
    if len(y) < smooth_window + 2:
        return None
    w = max(2, min(smooth_window, len(y) // 2))
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    y_padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(w) / w
    y_smooth = np.convolve(y_padded, kernel, mode="valid")
    grad = np.diff(y_smooth)
    if len(grad) < 3:
        return None
    n = len(grad)
    best_score = -np.inf
    best_idx = None
    for k in range(2, n - 1):
        left_mean = grad[:k].mean()
        right_mean = grad[k:].mean()
        score = abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n)
        if score > best_score:
            best_score = score
            best_idx = k
    if best_idx is None or best_score < 1e-6:
        return None
    return best_idx + 1


def _sensai_score(
    epochs,
    threshold,
    reference_cov,
    n_pc=3,
    noise_multiplier=3.0,
    engine="numpy",
):
    """Compute the SENSAI score for given threshold.

    Parameters
    ----------
    epochs : mne.Epochs or np.ndarray
        Input epochs data (or Epochs instance).
    threshold : float
        Eigenvalue threshold.
    reference_cov : np.ndarray
        Reference covariance matrix.
    n_pc : int
        Number of principal components.
    noise_multiplier : float
        Noise multiplier.
    engine : str
        Computation engine ('numpy', 'torch', or 'auto').

    Returns
    -------
    score : float
    signal_subspace_similarity : float
    noise_subspace_similarity : float
    """
    if n_pc is None:
        raise ValueError("n_pc must be a positive integer, got None.")
    n_pc = ensure_int(n_pc, "n_pc")
    if not 1 <= n_pc <= reference_cov.shape[0]:
        raise ValueError(
            f"n_pc must be an integer in the range [1, {reference_cov.shape[0]}], "
            f"got {n_pc!r}."
        )

    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov, engine=engine)
    return _sensai_score_from_gevd(
        all_eval,
        all_evec,
        reference_cov,
        reference_eigenvectors,
        threshold,
        n_pc,
        noise_multiplier,
    )


def _sensai_gridsearch(
    epochs,
    reference_cov,
    n_pc,
    noise_multiplier,
    eigen_thresholds,
    n_jobs=1,
    verbose=None,
    all_eval=None,
    all_evec=None,
    sensai_thresholds=None,
    signal_type="eeg",
    engine="numpy",
):
    n_pc = ensure_int(n_pc, "n_pc")
    if not 1 <= n_pc <= reference_cov.shape[0]:
        raise ValueError(
            f"n_pc must be an integer in the range [1, {reference_cov.shape[0]}], "
            f"got {n_pc!r}."
        )

    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    # Subsample if too many epochs
    MAX_EPOCHS = 500
    if len(epochs_data) > MAX_EPOCHS:
        idx = np.linspace(0, len(epochs_data) - 1, MAX_EPOCHS, dtype=int)
        epochs_data = epochs_data[idx]
        if all_eval is not None:
            all_eval = all_eval[idx]
        if all_evec is not None:
            all_evec = all_evec[idx]

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    if all_eval is None or all_evec is None:
        all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov, engine=engine)

    resolved = resolve_engine(engine)

    # Precompute template and all_VR once for all scoring evaluations
    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])
    all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", reference_cov, all_evec))
    abs_evals = np.ascontiguousarray(np.abs(all_eval))

    if resolved == "torch":
        import torch
        abs_evals_t = torch.from_numpy(abs_evals)
        all_VR_t = torch.from_numpy(all_VR)
        template_t = torch.from_numpy(template)

    runs = []
    for threshold in eigen_thresholds:
        if resolved == "torch":
            sig_sims_t, noi_sims_t = _sensai_score_torch(
                abs_evals_t,
                all_VR_t,
                template_t,
                threshold,
                n_pc=n_pc,
                signal_type=signal_type,
            )
            sig_sim = float(sig_sims_t.mean().item() * 100.0)
            noi_sim = float(noi_sims_t.mean().item() * 100.0)
        else:
            sig_sims, noi_sims = _sensai_score_loop(
                abs_evals,
                all_VR,
                template,
                threshold,
                n_pc=n_pc,
                signal_type=signal_type,
            )
            sig_sim = float(np.mean(sig_sims) * 100.0)
            noi_sim = float(np.mean(noi_sims) * 100.0)

        score = float(sig_sim - (noise_multiplier * noi_sim))
        runs.append((score, sig_sim, noi_sim))

    scores = np.array([run[0] for run in runs])
    noise_sims = np.array([run[2] for run in runs])
    best_idx = int(np.argmax(scores))

    # Check for degenerate monotonic / boundary curve safeguards
    if len(runs) >= 4:
        peak_is_at_boundary = best_idx >= len(runs) - 2
        baseline_score = float(np.median(scores[: max(1, len(runs) // 4)]))
        peak_score = float(scores[best_idx])
        dramatic_rise = (
            abs(peak_score) > 5 * max(abs(baseline_score), 1.0)
            and peak_score > baseline_score + 20
        )
        if peak_is_at_boundary and dramatic_rise:
            baseline_tol = max(1.0, 0.1 * abs(baseline_score))
            near_baseline = np.where(np.abs(scores - baseline_score) <= baseline_tol)[0]
            if len(near_baseline) > 0:
                best_idx = int(near_baseline[-1])
            else:
                best_idx = 0
        else:
            noise_changepoint_idx = _find_changepoint(noise_sims)
            if (
                noise_changepoint_idx is not None
                and best_idx > noise_changepoint_idx
                and noise_changepoint_idx > 0
                and peak_is_at_boundary
            ):
                best_idx = noise_changepoint_idx

    best_threshold = eigen_thresholds[best_idx]

    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sensai_optimize(
    epochs,
    reference_cov,
    n_pc,
    noise_multiplier,
    epochs_eigenvalues,
    bounds,
    all_eval=None,
    all_evec=None,
    percentile=98,
    signal_type="eeg",
    engine="numpy",
    sensai_tol: float = 0.1,
):
    n_pc = ensure_int(n_pc, "n_pc")
    if not 1 <= n_pc <= reference_cov.shape[0]:
        raise ValueError(
            f"n_pc must be an integer in the range [1, {reference_cov.shape[0]}], "
            f"got {n_pc!r}."
        )

    resolved = resolve_engine(engine)

    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    # Subsample if too many epochs
    MAX_EPOCHS = 500
    if len(epochs_data) > MAX_EPOCHS:
        idx = np.linspace(0, len(epochs_data) - 1, MAX_EPOCHS, dtype=int)
        epochs_data = epochs_data[idx]
        if all_eval is not None:
            all_eval = all_eval[idx]
        if all_evec is not None:
            all_evec = all_evec[idx]

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    if all_eval is None or all_evec is None:
        all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov, engine=engine)

    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])
    all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", reference_cov, all_evec))
    abs_evals = np.ascontiguousarray(np.abs(all_eval))

    # Precompute log-percentile scalar once outside the objective loop
    all_diagonals = np.abs(np.asarray(epochs_eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    p_val = float(np.percentile(log_eig_val_all, percentile))

    if resolved == "torch":
        import torch
        abs_evals_t = torch.from_numpy(abs_evals)
        all_VR_t = torch.from_numpy(all_VR)
        template_t = torch.from_numpy(template)

    runs = []

    def objective_function(sensai_threshold):
        T1 = (105 - sensai_threshold) / 100
        eigen_threshold = float(np.exp(T1 * p_val - 100))

        if resolved == "torch":
            sig_sims_t, noi_sims_t = _sensai_score_torch(
                abs_evals_t,
                all_VR_t,
                template_t,
                eigen_threshold,
                n_pc=n_pc,
                signal_type=signal_type,
            )
            sig_sim = float(sig_sims_t.mean().item() * 100.0)
            noi_sim = float(noi_sims_t.mean().item() * 100.0)
        else:
            sig_sims, noi_sims = _sensai_score_loop(
                abs_evals,
                all_VR,
                template,
                eigen_threshold,
                n_pc=n_pc,
                signal_type=signal_type,
            )
            sig_sim = float(np.mean(sig_sims) * 100.0)
            noi_sim = float(np.mean(noi_sims) * 100.0)

        score = float(sig_sim - (noise_multiplier * noi_sim))

        runs.append(
            [
                eigen_threshold,
                score,
                sig_sim,
                noi_sim,
            ]
        )
        return -score

    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method="bounded",
        options={
            "xatol": float(sensai_tol),
            "maxiter": 500,
        },
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    sensai_threshold = result.x
    T1_opt = (105 - sensai_threshold) / 100
    eigen_threshold = float(np.exp(T1_opt * p_val - 100))
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs
