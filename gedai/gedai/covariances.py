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


def _pick_cov(cov, ch_names):
    cov_ch_names = cov.ch_names

    picks_cov = []
    picks_ch_names = []
    for cov_name in cov_ch_names:
        for ch_name in ch_names:
            if ch_name.lower() == cov_name.lower():
                picks_cov.append(cov_name)
                picks_ch_names.append(ch_name)
                break
    if len(picks_cov) == 0:
        raise ValueError("No matching channel names found between inst and cov.\n"
                         f"Available channels in covariance are {cov_ch_names}.\n"
                         f"but instance has channels {ch_names}.")
    if len(picks_cov) < len(ch_names):
        raise ValueError("Only a subset of channels in the instance are present"
                         " in the covariance.\n"
                        f"Use inst.pick_channels({picks_ch_names}) to select only the channels"
                        f" that are in the covariance or provide a covariance that contains"
                        f" all channels in the instance.")
    cov = cov.copy().pick_channels(picks_cov)
    # Update the channel names in the covariance to match those in the instance
    cov.update(names=ch_names)
    return cov
