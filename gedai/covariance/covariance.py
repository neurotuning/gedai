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
        if reference_cov == "leadfield":
            reference_cov = mne.read_cov(str(get_leadfield_cov_path()))
        else:
            raise ValueError(
                "Reference covariance must be 'leadfield', an mne.Covariance, or an mne.Forward instance, "
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
                "\nNote: If you are processing MEG data ('mag' or 'grad'), the default 'leadfield' "
                "bundled with GEDAI is an EEG leadfield. For MEG data, please provide an MEG forward "
                "model (mne.Forward) or reference covariance (mne.Covariance) via the 'reference_cov' argument."
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
    cov = cov.copy().pick_channels(picks_cov)
    # Update the channel names in the covariance to match those in the instance
    cov.update(names=ch_names)
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


def compute_covariance_from_channel_positions(info):
    """Compute covariance matrix from channel positions.

    Parameters
    ----------
    info : instance of mne.Info
        The info structure containing channel information.

    Returns
    -------
    cov : instance of mne.Covariance
        The computed covariance matrix.
    """
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
    nfree = len(ch_names)  # TODO: fix
    cov = mne.Covariance(data, ch_names, bads, nfree=nfree, projs=[], verbose=None)
    return cov
