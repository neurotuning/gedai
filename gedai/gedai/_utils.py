import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw

from ..utils._checks import _check_picks_uniqueness
from ..utils.logs import logger


def _detect_signal_type(info):
    """Detect whether data is 'eeg' or 'meg' based on channel types."""
    ch_types = info.get_channel_types(unique=True)
    if any(t in ("mag", "grad", "ref_meg") for t in ch_types):
        return "meg"
    return "eeg"


def _ensure_wavelet_low_cutoff(
    wavelet_low_cutoff, filter_highpass=None, epoch_duration=None
):
    """Resolve wavelet low cutoff frequency from user parameter and filter highpass."""
    if wavelet_low_cutoff == "auto":
        if filter_highpass is not None and filter_highpass > 0:
            return max(0.5, float(filter_highpass))
        return 0.5
    elif wavelet_low_cutoff is None:
        return 0.0
    return float(wavelet_low_cutoff)


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
        mean_across_channels = np.mean(data, axis=0)
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()
        mean_across_channels = np.mean(data, axis=1)
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")

    return np.allclose(mean_across_channels, 0, atol=1e-6)


def _check_reference_channel(inst):
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
        flat_mask = [np.allclose(ch, 0, atol=1e-8) for ch in data]
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()
        flat_mask = [
            np.allclose(data[:, c, :], 0, atol=1e-8) for c in range(data.shape[1])
        ]
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")

    if any(flat_mask):
        return

    logger.warning(
        "Input data does not contain a flat reference channel. "
        "GEDAI will apply average referencing. "
        "Consider adding the reference channel(s) using "
        ":func:`mne.add_reference_channels` before using GEDAI."
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

    ch_type = raw_fit.info.get_channel_types()[0]
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


def _format_summary_table(model) -> str:
    """Format a MATLAB-style summarized table for a fitted/transformed GEDAI model."""
    lines = []
    lines.append("=" * 82)
    lines.append(f"  {model.__class__.__name__} Summary Table")
    lines.append("=" * 82)

    headers = ("Frequency Band", "Epoch (s)", "Threshold", "SENSAI (%)", "ENOVA (%)")
    row_fmt = "  {:<26} | {:>10} | {:>10} | {:>11} | {:>10}"
    div_line = (
        "  "
        + "-" * 26
        + "-+-"
        + "-" * 10
        + "-+-"
        + "-" * 10
        + "-+-"
        + "-" * 11
        + "-+-"
        + "-" * 10
    )

    lines.append(row_fmt.format(*headers))
    lines.append(div_line)

    if hasattr(model, "_broadband_model") and model._broadband_model is not None:
        bm = model._broadband_model
        if getattr(bm, "fitted", False) and hasattr(bm, "_fit") and bm._fit is not None:
            t = bm._fit.get("threshold", 0.0)
            bm_sensai = (
                bm.fit_metrics_.get("sensai_score")
                if getattr(bm, "fit_metrics_", None)
                else None
            )
            s_str = f"{bm_sensai:.2f} %" if bm_sensai is not None else "--"
            lines.append(
                row_fmt.format("Pass 1: Broadband", "1.00 s", f"{t:.4g}", s_str, "--")
            )

    if hasattr(model, "_wavelets_fits") and model._wavelets_fits is not None:
        for wf in model._wavelets_fits:
            w_idx = wf.get("band_index", 0)
            fmin = wf.get("fmin", 0.0)
            fmax = wf.get("fmax", 0.0)
            dur = wf.get("duration", 1.0)
            dur_str = f"{dur:.2f} s" if dur is not None else "--"
            band_label = f"Band {w_idx} ({fmin:.2f}-{fmax:.2f} Hz)"
            if wf.get("ignore", False):
                lines.append(row_fmt.format(band_label, dur_str, "IGNORED", "--", "--"))
            else:
                m = wf.get("model")
                t = m.threshold if m is not None else 0.0
                sensai_val = wf.get("sensai")
                sensai_str = f"{sensai_val:.2f} %" if sensai_val is not None else "--"
                enova_val = wf.get("enova")
                enova_str = (
                    f"{enova_val * 100:.2f} %"
                    if enova_val is not None and enova_val > 0
                    else "--"
                )
                lines.append(
                    row_fmt.format(
                        band_label, dur_str, f"{t:.4g}", sensai_str, enova_str
                    )
                )
    elif hasattr(model, "_fit") and model._fit is not None:
        t = model.threshold
        sensai_val = (
            model.fit_metrics_.get("sensai_score")
            if getattr(model, "fit_metrics_", None)
            else None
        )
        sensai_str = f"{sensai_val:.2f} %" if sensai_val is not None else "--"
        dur_str = (
            f"{model._duration:.2f} s"
            if hasattr(model, "_duration") and model._duration is not None
            else "--"
        )
        lines.append(
            row_fmt.format(
                "Broadband (all freqs)", dur_str, f"{t:.4g}", sensai_str, "--"
            )
        )

    lines.append("=" * 82)

    if hasattr(model, "fit_metrics_") and model.fit_metrics_ is not None:
        score = model.fit_metrics_.get("sensai_score")
        if score is not None:
            lines.append(f"  Fitted SENSAI Optimization Score: {score:.2f} %")
            lines.append("=" * 82)

    return "\n".join(lines)
