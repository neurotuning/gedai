import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw

from ..utils._checks import _check_picks_uniqueness, _check_type, _check_value
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
            return max(0.1, float(filter_highpass))
        return 0.1
    elif wavelet_low_cutoff is None:
        return 0.0
    return float(wavelet_low_cutoff)


def _check_on_bad_channel(on_bad_channel: str) -> str:
    """Validate the ``on_bad_channel`` handling mode."""
    _check_type(on_bad_channel, (str,), "on_bad_channel")
    _check_value(on_bad_channel, ("exclude", "ignore"), "on_bad_channel")
    return on_bad_channel


def _check_bad_channel_threshold(
    bad_channel_threshold: float | str | None,
) -> float | None:
    """Validate and resolve the automatic bad-channel ENOVA threshold."""
    if bad_channel_threshold is None:
        return None
    if isinstance(bad_channel_threshold, bool):
        raise TypeError(
            "bad_channel_threshold must be float, 'auto', or None, got bool instead."
        )
    if isinstance(bad_channel_threshold, str):
        val = bad_channel_threshold.strip().lower()
        if val == "auto":
            return 0.95
        raise ValueError(
            f"Unknown bad_channel_threshold string: {bad_channel_threshold!r}. "
            "Expected 'auto', a positive float, np.inf, or None."
        )
    _check_type(bad_channel_threshold, ("numeric",), "bad_channel_threshold")
    val = float(bad_channel_threshold)
    if np.isnan(val) or val <= 0:
        raise ValueError(
            "bad_channel_threshold must be > 0 (or None / np.inf), "
            f"got {bad_channel_threshold!r}."
        )
    if np.isinf(val):
        return None
    return val


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


def _good_channel_indices(inst) -> list[int]:
    """Return indices of channels not listed in ``inst.info['bads']``."""
    bads = set(inst.info["bads"])
    return [i for i, ch in enumerate(inst.ch_names) if ch not in bads]


def _check_average_reference(inst):
    good_idx = _good_channel_indices(inst)
    if not good_idx:
        return False
    if isinstance(inst, BaseRaw):
        data = inst.get_data(picks=good_idx)
        mean_across_channels = np.mean(data, axis=0)
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data(picks=good_idx)
        mean_across_channels = np.mean(data, axis=1)
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")

    return bool(np.allclose(mean_across_channels, 0, atol=1e-6))


def _check_reference_channel(inst):
    good_idx = _good_channel_indices(inst)
    if not good_idx:
        flat_mask = []
    elif isinstance(inst, BaseRaw):
        data = inst.get_data(picks=good_idx)
        flat_mask = [np.allclose(ch, 0, atol=1e-8) for ch in data]
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data(picks=good_idx)
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


def _detect_flat_channels(inst) -> list[str]:
    """Detect flat channels using unit-invariant first temporal differences."""
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
        if data.shape[-1] < 2:
            return []
        diffs_2d = np.diff(data, axis=-1)
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()
        if data.shape[-1] < 2 or data.shape[0] == 0:
            return []
        diffs = np.diff(data, axis=-1)
        diffs_2d = diffs.transpose(1, 0, 2).reshape(data.shape[1], -1)
    else:
        raise ValueError("Instance must be either a Raw or Epochs object.")

    ddof = 1 if diffs_2d.shape[1] > 1 else 0
    channel_diff_std = np.std(diffs_2d, axis=1, ddof=ddof)
    positive = channel_diff_std[np.isfinite(channel_diff_std) & (channel_diff_std > 0)]
    typical = float(np.median(positive)) if positive.size > 0 else np.nan
    tol = 1e-7 * typical if np.isfinite(typical) else 1e-7
    return [
        inst.ch_names[i]
        for i, val in enumerate(channel_diff_std)
        if not np.isfinite(val) or val < tol
    ]


def _resolve_fit_picks(inst, picks, on_bad_channel: str = "exclude"):
    """Resolve channel pick indices for fitting according to ``on_bad_channel``."""
    _check_on_bad_channel(on_bad_channel)
    pick_idx = _picks_to_idx(inst.info, picks, none="all", exclude=[])
    _check_picks_uniqueness(inst.info, pick_idx)

    bads_set = set(inst.info["bads"])
    picked_bads = [inst.ch_names[i] for i in pick_idx if inst.ch_names[i] in bads_set]

    if on_bad_channel == "exclude":
        good_idx = np.array(
            [i for i in pick_idx if inst.ch_names[i] not in bads_set],
            dtype=pick_idx.dtype,
        )
        if len(good_idx) == 0:
            raise ValueError(
                "No channels remain after excluding bad channels "
                f"{picked_bads}. Pass on_bad_channel='ignore' to include them."
            )
        if picked_bads:
            logger.warning(
                "Excluding bad channel(s) %s from GEDAI fit "
                "(on_bad_channel='exclude').",
                picked_bads,
            )
        return good_idx, picked_bads

    if picked_bads:
        logger.warning(
            "Ignoring bad channel annotation(s) for %s (on_bad_channel='ignore').",
            picked_bads,
        )
    return pick_idx, []


def _apply_eeg_average_reference_if_needed(inst):
    """Apply EEG average reference when non-bad EEG channels are not avg-referenced."""
    ch_type = inst.info.get_channel_types()[0]
    if ch_type == "eeg":
        is_average_ref = _check_average_reference(inst)
        if not is_average_ref:
            _check_reference_channel(inst)
            logger.info("Setting average reference.")
            inst.set_eeg_reference("average", projection=False)


def _prepare_inst_fit(inst, picks, on_bad_channel="exclude", return_bads=False):
    pick_idx, excluded_bads = _resolve_fit_picks(
        inst, picks, on_bad_channel=on_bad_channel
    )
    inst_fit = inst.copy()
    inst_fit.load_data()
    inst_fit = inst_fit.pick(pick_idx)
    inst_fit.info["bads"] = []

    _apply_eeg_average_reference_if_needed(inst_fit)

    if return_bads:
        return inst_fit, excluded_bads
    return inst_fit


def _prepare_inst_transform(
    inst,
    picks,
    on_bad_channel="exclude",
    excluded_bads=None,
):
    _check_on_bad_channel(on_bad_channel)
    input_bads = set(inst.info["bads"])
    fitted_bads = [ch for ch in picks if ch in input_bads]
    if fitted_bads:
        if on_bad_channel == "exclude":
            raise ValueError(
                "The following fitted channel(s) are marked as bad in the input "
                f"instance: {fitted_bads}. Re-fit the model including these "
                "annotations, clear info['bads'], or fit with "
                "on_bad_channel='ignore'."
            )
        logger.warning(
            "Ignoring bad channel annotation(s) for %s during transform "
            "(on_bad_channel='ignore').",
            fitted_bads,
        )

    pick_idx = _picks_to_idx(inst.info, picks, none="all", exclude=[])
    inst_transform = inst.copy()
    inst_transform.load_data()
    inst_transform = inst_transform.pick(pick_idx)
    inst_transform.info["bads"] = []

    extra_ch = set(inst.info["ch_names"]) - set(inst_transform.ch_names)
    known_excluded = extra_ch & set(excluded_bads or ())
    unexpected_extra = extra_ch - known_excluded
    if known_excluded:
        logger.info(
            "Excluded bad channel(s) %s are omitted from transformed output.",
            sorted(known_excluded),
        )
    if unexpected_extra:
        logger.warning(
            "The following channels are present in the input inst but were "
            "not present during fitting: "
            f"{unexpected_extra}. \n"
            "These channels will be ignored during transformation. \n"
        )

    _apply_eeg_average_reference_if_needed(inst_transform)
    return inst_transform


def _prepare_epochs_fit(epochs, picks, on_bad_channel="exclude", return_bads=False):
    return _prepare_inst_fit(
        epochs, picks, on_bad_channel=on_bad_channel, return_bads=return_bads
    )


def _prepare_epochs_transform(
    epochs,
    picks,
    on_bad_channel="exclude",
    excluded_bads=None,
):
    return _prepare_inst_transform(
        epochs,
        picks,
        on_bad_channel=on_bad_channel,
        excluded_bads=excluded_bads,
    )


def _prepare_raw_fit(raw, picks, on_bad_channel="exclude", return_bads=False):
    return _prepare_inst_fit(
        raw, picks, on_bad_channel=on_bad_channel, return_bads=return_bads
    )


def _prepare_raw_transform(
    raw,
    picks,
    on_bad_channel="exclude",
    excluded_bads=None,
):
    return _prepare_inst_transform(
        raw,
        picks,
        on_bad_channel=on_bad_channel,
        excluded_bads=excluded_bads,
    )


def _interpolate_bad_channels(inst_orig, inst_clean, bad_channels):
    """Interpolate excluded bad channels back into the cleaned instance."""
    clean_set = set(inst_clean.ch_names)
    orig_set = set(inst_orig.ch_names)
    bad_to_interp = [
        ch for ch in bad_channels if ch in orig_set and ch not in clean_set
    ]
    if not bad_to_interp:
        return inst_clean

    keep_set = clean_set | set(bad_to_interp)
    target_chs = [ch for ch in inst_orig.ch_names if ch in keep_set]
    inst_out = inst_orig.copy()
    inst_out.load_data()
    inst_out = inst_out.pick(target_chs)

    missing_loc_chs = []
    for ch in inst_out.info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=float)
        if not np.all(np.isfinite(loc)) or np.allclose(loc, 0.0):
            missing_loc_chs.append(ch["ch_name"])
    if missing_loc_chs:
        raise ValueError(
            "Cannot interpolate bad channels without valid 3D channel locations "
            f"for {missing_loc_chs}. Set a montage via inst.set_montage(...) "
            "before transforming with interpolate_bads=True."
        )

    clean_data = inst_clean.get_data(verbose=False)
    out_data = inst_out.get_data(verbose=False).copy()
    out_idx_map = {ch: idx for idx, ch in enumerate(inst_out.ch_names)}
    for clean_i, ch_name in enumerate(inst_clean.ch_names):
        out_i = out_idx_map[ch_name]
        if isinstance(inst_out, BaseRaw):
            out_data[out_i, :] = clean_data[clean_i, :]
        else:
            out_data[:, out_i, :] = clean_data[:, clean_i, :]
    inst_out._data = out_data

    inst_out.info["bads"] = list(bad_to_interp)
    inst_out.interpolate_bads(reset_bads=True, verbose=False)
    if _detect_signal_type(inst_out.info) == "eeg":
        inst_out.set_eeg_reference("average", projection=False, verbose=False)
    return inst_out


def _run_two_pass_bad_channel_detection(
    model,
    inst,
    fit_method_name,
    transform_method_name,
    picks,
    on_bad_channel,
    threshold,
    fit_kwargs,
    transform_kwargs,
):
    """Run MATLAB-parity two-pass bad-channel detection and rejection.

    Pre-pass identifies flat channels via unit-invariant first temporal
    differences before average referencing. Pass 1 fits and transforms the
    non-flat channels to evaluate per-channel ENOVA. When any channels exceed
    ``threshold`` (or are flat), Pass 2 re-fits cleanly on the remaining
    good channels.
    """
    pick_idx, excluded_annotated_bads = _resolve_fit_picks(
        inst, picks, on_bad_channel=on_bad_channel
    )
    inst_cand = inst.copy()
    inst_cand.load_data()
    inst_cand = inst_cand.pick(pick_idx)
    inst_cand.info["bads"] = []

    flat_channels = _detect_flat_channels(inst_cand)
    flat_set = set(flat_channels)
    non_flat = [ch for ch in inst_cand.ch_names if ch not in flat_set]
    if len(non_flat) < 2:
        raise ValueError(
            "At least 2 non-flat channels are required for bad-channel detection, "
            f"got {len(non_flat)} (flat channels detected: {flat_channels})."
        )

    pass1_model = model._clone_unfitted()
    getattr(pass1_model, fit_method_name)(
        inst_cand,
        picks=non_flat,
        on_bad_channel="ignore",
        bad_channel_threshold=None,
        **fit_kwargs,
    )
    pass1_model.bad_channels_ = list(flat_channels)
    getattr(pass1_model, transform_method_name)(
        inst_cand,
        verbose=False,
        **transform_kwargs,
    )

    enova_vals = pass1_model.metrics_["enova_per_channel"]
    enova_map = {
        ch: float(val) for ch, val in zip(non_flat, enova_vals, strict=True)
    }
    enova_dict = {
        ch: (float("inf") if ch in flat_set else enova_map[ch])
        for ch in inst_cand.ch_names
    }
    noisy_channels = [ch for ch in non_flat if enova_dict[ch] > threshold]
    detected_set = flat_set | set(noisy_channels)
    detected_bads = [ch for ch in inst_cand.ch_names if ch in detected_set]
    all_bad_set = set(excluded_annotated_bads) | detected_set
    all_bads = [ch for ch in inst.ch_names if ch in all_bad_set]

    if not detected_bads:
        model.__dict__.update(pass1_model.__dict__)
        model.metrics_ = None
    else:
        remaining = [ch for ch in inst_cand.ch_names if ch not in detected_set]
        if len(remaining) < 2:
            raise ValueError(
                "Fewer than 2 channels remain after automatic bad-channel rejection "
                f"(flat={flat_channels}, noisy={noisy_channels})."
            )
        logger.warning(
            "Automatic bad-channel detection excluded %d channel(s) "
            "(flat=%s, noisy=%s, threshold=%.4g); re-fitting Pass 2.",
            len(detected_bads),
            flat_channels,
            noisy_channels,
            threshold,
        )
        getattr(model, fit_method_name)(
            inst_cand,
            picks=remaining,
            on_bad_channel="ignore",
            bad_channel_threshold=None,
            **fit_kwargs,
        )

    if hasattr(model, "_fitted_raw_ref"):
        model._fitted_raw_ref = inst
        model._fitted_raw_id = id(inst)
        model._fitted_raw_shape = inst._data.shape if hasattr(inst, "_data") else None
        model._fitted_raw_ch_names = list(inst.ch_names)
        model._fitted_raw_bads = tuple(inst.info["bads"])

    model._on_bad_channel = on_bad_channel
    model._bad_channel_threshold = threshold
    model.flat_channels_ = list(flat_channels)
    model.noisy_channels_ = list(noisy_channels)
    model.detected_bad_channels_ = list(detected_bads)
    model.bad_channels_ = list(all_bads)
    model.enova_per_channel_ = enova_dict


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

    if getattr(model, "bad_channels_", None):
        bads_str = ", ".join(model.bad_channels_)
        lines.append(
            f"  Bad channels excluded ({len(model.bad_channels_)}): {bads_str}"
        )
        lines.append("=" * 82)

    if hasattr(model, "fit_metrics_") and model.fit_metrics_ is not None:
        score = model.fit_metrics_.get("sensai_score")
        if score is not None:
            lines.append(f"  Fitted SENSAI Optimization Score: {score:.2f} %")
            lines.append("=" * 82)

    return "\n".join(lines)
