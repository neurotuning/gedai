import mne
import numpy as np
import sklearn.metrics

from ..data import get_leadfield_cov_path
from ..utils._checks import _check_type


def _ensure_cov(reference_cov):
    _check_type(reference_cov, (str, mne.Covariance, mne.Forward), "reference_cov")
    if isinstance(reference_cov, mne.Forward):
        return compute_covariance_from_forward(reference_cov)
    if isinstance(reference_cov, str):
        if reference_cov in ("leadfield", "leadfield_geometric", "geometric"):
            cov = mne.read_cov(str(get_leadfield_cov_path()))
            if reference_cov in ("leadfield_geometric", "geometric"):
                cov["_align_to_sensors"] = True
            return cov
        else:
            raise ValueError(
                "Reference covariance must be 'leadfield', 'leadfield_geometric', "
                "'geometric', an mne.Covariance, or an mne.Forward instance; "
                f"got '{reference_cov}' instead."
            )
    return reference_cov


def _pick_cov(cov, ch_names, info=None):
    if isinstance(ch_names, mne.Info):
        info = ch_names
        ch_names = info["ch_names"]
    elif hasattr(ch_names, "info"):
        info = ch_names.info
        ch_names = info["ch_names"]

    cov_ch_names = cov.ch_names

    picks_cov = []
    picks_ch_names = []
    for ch_name in ch_names:
        for cov_name in cov_ch_names:
            if ch_name.lower() == cov_name.lower():
                picks_cov.append(cov_name)
                picks_ch_names.append(ch_name)
                break
    if len(picks_cov) == 0:
        msg = (
            "No matching channel names found between inst and cov.\n"
            f"Available channels in covariance are {cov_ch_names}.\n"
            f"but instance has channels {ch_names}."
        )
        is_meg = False
        if info is not None:
            ch_types = info.get_channel_types(unique=True)
            is_meg = any(t in ("mag", "grad", "ref_meg") for t in ch_types)
        if is_meg:
            msg += (
                "\nNote: If you are processing MEG data ('mag' or 'grad'), "
                "the default 'leadfield' bundled with GEDAI is an EEG leadfield. "
                "For MEG data, please provide an MEG forward model "
                "(mne.Forward) or reference covariance (mne.Covariance) via "
                "the 'reference_cov' argument."
            )
        raise ValueError(msg)
    if len(picks_cov) < len(ch_names):
        raise ValueError(
            "Only a subset of channels in the instance are present"
            " in the covariance.\n"
            f"Use inst.pick_channels({picks_ch_names}) to select only the channels"
            f" that are in the covariance or provide a covariance that contains"
            f" all channels in the instance."
        )
    align_flag = cov.get("_align_to_sensors", False)
    cov = cov.copy().pick_channels(picks_cov)
    # Update the channel names in the covariance to match those in the instance
    cov.update(names=ch_names)
    if align_flag and info is not None:
        cov = align_covariance_to_channel_positions(cov, info)
    return cov


def compute_covariance_from_forward(forward):
    """Compute covariance matrix from the leadfield of a forward solution.

    Parameters
    ----------
    forward : mne.Forward
        The forward solution from which to compute the covariance matrix.

    Returns
    -------
    cov : mne.Covariance
        The computed covariance matrix.
    """
    _check_type(forward, (mne.Forward,), "forward")
    if forward["coord_frame"] != mne._fiff.constants.FIFF.FIFFV_COORD_HEAD:
        raise ValueError("Forward solution must be in head coordinates.")
    data = forward["sol"]["data"] @ forward["sol"]["data"].T
    ch_names = forward["info"]["ch_names"]
    bads = forward["info"]["bads"]
    nfree = len(ch_names)  # TODO: fix
    cov = mne.Covariance(
        data, names=ch_names, bads=bads, projs=[], nfree=nfree, verbose=None
    )
    return cov


def compute_covariance_from_channel_positions(info, method="geometric"):
    """Compute covariance matrix from channel positions.

    Parameters
    ----------
    info : instance of mne.Info
        The info structure containing channel information.
    method : str, default 'geometric'
        Method to compute covariance:
        - 'geometric': Aligns template leadfield with 3D sensor coordinates via QR surgery.
        - 'rbf': Computes an exponential distance Gaussian kernel.

    Returns
    -------
    cov : instance of mne.Covariance
        The computed covariance matrix.
    """
    if method == "geometric":
        try:
            cov = mne.read_cov(str(get_leadfield_cov_path()))
            cov = _pick_cov(cov, info)
            return align_covariance_to_channel_positions(cov, info)
        except Exception:
            # Fall back to rbf if leadfield channel matching fails
            pass

    ch_positions = [info["chs"][i]["loc"][:3] for i in range(info["nchan"])]
    ch_distance_matrix = sklearn.metrics.pairwise_distances(
        ch_positions, metric="euclidean"
    )
    nonzero = ch_distance_matrix[ch_distance_matrix > 0]
    ell = np.median(nonzero) if nonzero.size else 1.0
    sigma2 = 1.0
    eps = 1e-6

    data = sigma2 * np.exp(-(ch_distance_matrix**2) / (2 * ell**2))
    data += eps * np.eye(data.shape[0])

    ch_names = info["ch_names"]
    bads = info["bads"]
    nfree = len(ch_names)
    cov = mne.Covariance(data, ch_names, bads, nfree=nfree, projs=[], verbose=None)
    return cov


def align_covariance_to_channel_positions(cov, info, n_geom_pcs=3):
    """Align leading eigenvectors of a reference covariance to 3D sensor coordinates.

    Applies Rank-3 Subspace Surgery using the QR-orthonormalized Cartesian coordinates
    (X, Y, Z) of the sensors. In volume conduction physics, the lowest-order multipoles
    of dipolar cortical sources project along Cartesian axes:
    - Mode 1: Left-Right dipole gradient (proportional to X)
    - Mode 2: Anterior-Posterior dipole gradient (proportional to Y)
    - Mode 3: Inferior-Superior dipole gradient (proportional to Z)

    This aligns the top 3 principal components of the reference covariance directly
    with the physical sensor geometry without requiring clean empirical EEG data, while
    preserving the full-rank eigenvalue spectrum and the remaining (n_channels - 3)
    orthogonal volume-conduction dimensions for GEVD.

    Parameters
    ----------
    cov : mne.Covariance | np.ndarray
        The reference covariance matrix to align (e.g. from template leadfield).
    info : mne.Info | mne.io.BaseRaw | mne.BaseEpochs | np.ndarray
        Object containing sensor channel locations or (n_channels, 3) coordinates.
    n_geom_pcs : int, default 3
        Number of geometric Cartesian PCs to align (default 3 for X, Y, Z).

    Returns
    -------
    cov_aligned : mne.Covariance | np.ndarray
        The geometrically aligned covariance matrix.
    """
    if hasattr(info, "info"):
        info_obj = info.info
    else:
        info_obj = info

    ch_names = None
    bads = []
    if isinstance(info_obj, mne.Info):
        ch_names = info_obj["ch_names"]
        bads = info_obj["bads"]
        coords = np.array([ch["loc"][:3] for ch in info_obj["chs"]], dtype=np.float64)
    elif isinstance(info_obj, np.ndarray):
        coords = np.asarray(info_obj, dtype=np.float64)
    else:
        raise TypeError(f"info must be an mne.Info, Raw, Epochs, or ndarray, got {type(info_obj)}")

    is_mne_cov = isinstance(cov, mne.Covariance)
    if is_mne_cov:
        cov_data = cov.data.copy()
        if ch_names is None:
            ch_names = cov.ch_names
            bads = cov["bads"]
    else:
        cov_data = np.asarray(cov, dtype=np.float64).copy()

    n_channels = cov_data.shape[0]
    if coords.shape[0] != n_channels:
        raise ValueError(
            f"Number of channels in cov ({n_channels}) does not match coords ({coords.shape[0]})."
        )

    # If coordinates are missing (all zeros or NaNs), cannot align geometrically
    if np.all(coords == 0) or np.isnan(coords).any():
        return cov

    # Center coordinates
    coords_centered = coords - np.mean(coords, axis=0)
    if np.max(np.std(coords_centered, axis=0)) < 1e-12:
        return cov

    # Number of geometric PCs to align
    n_pcs = min(int(n_geom_pcs), n_channels - 1, coords.shape[1])
    if n_pcs < 1:
        return cov

    # Compute orthonormal basis for Cartesian coordinates via QR
    Q_coords, _ = np.linalg.qr(coords_centered[:, :n_pcs])

    # Eigendecomposition of reference covariance
    evals, evecs = np.linalg.eigh(cov_data)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Project remaining dimensions orthogonal to Q_coords
    P_orth = np.eye(n_channels) - Q_coords @ Q_coords.T
    V_rest = P_orth @ evecs[:, n_pcs:]
    Q_rest, _ = np.linalg.qr(V_rest)

    # Assemble aligned orthonormal basis
    V_aligned = np.hstack([Q_coords, Q_rest[:, :n_channels - n_pcs]])
    aligned_data = V_aligned @ np.diag(evals) @ V_aligned.T
    aligned_data = (aligned_data + aligned_data.T) * 0.5

    if is_mne_cov:
        nfree = getattr(cov, "nfree", n_channels)
        return mne.Covariance(
            aligned_data, names=ch_names, bads=bads, projs=[], nfree=nfree, verbose=False
        )
    return aligned_data
