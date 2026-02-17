import os
import mne
import sklearn.metrics

from ..utils._checks import check_type


def _compute_distance_cov(raw):
    ch_positions = [raw.info["chs"][i]["loc"][:3] for i in range(raw.info["nchan"])]
    ch_distance_matrix = sklearn.metrics.pairwise_distances(
        ch_positions, metric="euclidean"
    )
    cov = 1 - ch_distance_matrix
    return cov


def _ensure_cov(reference_cov):
    check_type(reference_cov, (str, mne.Covariance), "reference_cov")
    if isinstance(reference_cov, str):
        if reference_cov == "leadfield":
            reference_cov = mne.read_cov(os.path.join(os.path.dirname(__file__), "../data/fsavLEADFIELD_4_GEDAI-cov.fif"))
        else:
            raise ValueError(
                "Reference covariance must be 'leadfield'"
                f"got '{reference_cov}' instead."
            )
    return reference_cov


def _pick_cov(inst, cov):
    inst_ch_names = inst.ch_names
    cov_ch_names = cov.ch_names

    picks = []
    for cov_name in cov_ch_names:
        for inst_name in inst_ch_names:
            if inst_name.lower() == cov_name.lower():
                picks.append(cov_name)
                break
    if len(picks) == 0:
        raise ValueError("No matching channel names found between inst and cov.\n"
                         f"Available channels in covariance are {cov_ch_names}.\n"
                         f"but instance has channels {inst_ch_names}.")
    if len(picks) < len(inst_ch_names):
        raise ValueError("Only a subset of channels in the instance are present"
                         " in the covariance.\n"
                        f"Use inst.pick_channels({picks}) to select only the channels"
                        f" that are in the covariance or provide a covariance that contains"
                        f" all channels in the instance.")
    cov = cov.copy().pick_channels(picks)
    return cov
