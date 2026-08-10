
import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw

from ..utils._checks import _check_picks_uniqueness
from ..utils.logs import logger


def _check_fit_info(model, inst):
    missing_ch = set(model.ch_names) - set(inst.info["ch_names"])
    if len(missing_ch) > 0:
        raise ValueError(
            "The following channels are missing in the input inst but were "
            "present during fitting: "
            f"{missing_ch}. \n"
            "Please make sure to include the same channels during transform "
            "as were used during fit. \n"
            "See "
            f"{model.__class__.__name__}.ch_names "
            "for the list of channels used during fit."
        )
    if model._info["sfreq"] != inst.info["sfreq"]:
        raise ValueError(
            f"Sampling frequency of input instance ({inst.info['sfreq']} Hz)"
            f"  does not match sampling frequency of the data used during fit"
            f" ({model._info['sfreq']} Hz). You can resample the input instance"
            f" to {model._info['sfreq']} Hz before calling transform_raw."
        )
    return


def _check_average_reference(inst):
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()
        data = np.vstack(data)
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")

    mean_across_channels = np.mean(data, axis=0)
    is_average_ref = False
    if np.allclose(mean_across_channels, 0):
        is_average_ref = True
    return is_average_ref


def _check_reference_channel(inst):
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()
        data = np.vstack(data)
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")
    for data_channel in data:
        is_reference_channel = np.allclose(data_channel, 0)
        if is_reference_channel:
            return
    logger.warning(
        "Input data does not contain a flat reference channel. "
        "GEDAI will apply average referencing."
        "Consider adding the reference channel(s) using "
        ":func:`mne.mne.add_reference_channels` before using GEDAI."
    )
    return


def _prepare_epochs_fit(epochs, picks):
    picks = _picks_to_idx(epochs.info, picks, none="all", exclude=[])
    _check_picks_uniqueness(epochs.info, picks)
    epochs_fit = epochs.copy()
    epochs_fit.load_data()
    epochs_fit = epochs_fit.pick(picks)

    ch_type = epochs_fit.info.get_channel_types()[0]
    if ch_type == "eeg":
        is_average_ref = _check_average_reference(epochs_fit)
        if not is_average_ref:
            _check_reference_channel(epochs_fit)
            logger.info("Setting average reference.")
            epochs_fit.set_eeg_reference("average", projection=False)

    return epochs_fit


def _prepare_epochs_transform(epochs, picks):
    picks = _picks_to_idx(epochs.info, picks, none="all", exclude=[])
    epochs_transform = epochs.copy()
    epochs_transform.load_data()
    epochs_transform = epochs_transform.pick(picks)

    extra_ch = set(epochs.info["ch_names"]) - set(epochs_transform.ch_names)
    if len(extra_ch) > 0:
        logger.warning(
            "The following channels are present in the input inst but were "
            "not present during fitting: "
            f"{extra_ch}. \n"
            "These channels will be ignored during transformation. \n"
        )

    ch_type = epochs_transform.info.get_channel_types()[0]
    if ch_type == "eeg":
        is_average_ref = _check_average_reference(epochs_transform)
        if not is_average_ref:
            _check_reference_channel(epochs_transform)
            logger.info("Setting average reference.")
            epochs_transform.set_eeg_reference("average", projection=False)

    return epochs_transform


def _prepare_raw_fit(raw, picks):
    picks = _picks_to_idx(raw.info, picks, none="all", exclude=[])
    _check_picks_uniqueness(raw.info, picks)
    raw_fit = raw.copy().load_data().pick(picks)

    ch_type = raw.info.get_channel_types()[0]
    if ch_type == "eeg":
        is_average_ref = _check_average_reference(raw_fit)
        if not is_average_ref:
            _check_reference_channel(raw_fit)
            logger.info("Setting average reference.")
            raw_fit.set_eeg_reference("average", projection=False)
    return raw_fit


def _prepare_raw_transform(raw, picks):
    picks = _picks_to_idx(raw.info, picks, none="all", exclude=[])
    raw_transform = raw.copy().load_data().pick(picks)

    extra_ch = set(raw.ch_names) - set(raw_transform.info["ch_names"])
    if len(extra_ch) > 0:
        logger.warning(
            "The following channels are present in the input inst but were "
            "not present during fitting: "
            f"{extra_ch}. \n"
            "These channels will be ignored during transformation. \n"
        )

    ch_type = raw_transform.info.get_channel_types()[0]
    if ch_type == "eeg":
        is_average_ref = _check_average_reference(raw_transform)
        if not is_average_ref:
            _check_reference_channel(raw_transform)
            logger.info("Setting average reference.")
            raw_transform.set_eeg_reference("average", projection=False)
    return raw_transform
