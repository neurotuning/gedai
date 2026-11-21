import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eigh, svd
from scipy.signal import welch
from scipy.signal.windows import hann
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_samples

_LEGACY_1020_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _compute_enova_weighted_psd(
    data_before: np.ndarray,
    data_after: np.ndarray,
    data_noise: np.ndarray,
    sfreq: float,
    enova_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ENOVA-weighted Welch PSD curves (in dB) at 0.5 Hz resolution."""
    n_ch, n_times = data_after.shape
    nfft_psd = max(2, int(round(2.0 * sfreq)))
    nperseg = min(nfft_psd, n_times)
    noverlap = nperseg // 2
    win = hann(nperseg, sym=True) if nperseg >= 2 else np.ones(nperseg)

    welch_kwargs = {
        "fs": sfreq,
        "window": win,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft_psd,
        "detrend": "constant",
        "scaling": "density",
        "axis": -1,
    }
    f_psd, pxx_b = welch(data_before, **welch_kwargs)
    _, pxx_a = welch(data_after, **welch_kwargs)
    _, pxx_n = welch(data_noise, **welch_kwargs)

    w_ch = np.asarray(enova_pct, dtype=np.float64).ravel()
    if w_ch.size != n_ch or not np.all(np.isfinite(w_ch)) or float(np.sum(w_ch)) <= 0.0:
        w_ch = np.full(n_ch, 1.0 / max(1, n_ch), dtype=np.float64)
    else:
        w_ch = w_ch / float(np.sum(w_ch))

    eps = float(np.finfo(np.float64).eps)
    psd_b_db = 10.0 * np.log10(np.maximum(w_ch @ pxx_b, eps))
    psd_a_db = 10.0 * np.log10(np.maximum(w_ch @ pxx_a, eps))
    psd_n_db = 10.0 * np.log10(np.maximum(w_ch @ pxx_n, eps))
    return f_psd, psd_b_db, psd_a_db, psd_n_db, w_ch


def _normalize_eeg_label(label: str) -> str:
    """Normalize a channel label for 10-05 montage lookup."""
    key = label.strip().upper()
    if key.startswith("EEG "):
        key = key[4:].strip()
    for suffix in ("-REF", "-AVG", "-LE", "-M1", "-M2"):
        if key.endswith(suffix) and len(key) > len(suffix):
            key = key[: -len(suffix)].strip()
            break
    return _LEGACY_1020_ALIASES.get(key, key)


def _prepare_topomap_info(info: mne.Info) -> tuple[mne.Info, np.ndarray]:
    """Populate missing EEG 3D coordinates from standard 10-05 template."""
    info_copy = info.copy()

    def _has_valid_xyz(ch_dict: dict) -> bool:
        xyz = np.asarray(ch_dict["loc"][:3], dtype=np.float64)
        return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-6)

    valid_mask = np.array([_has_valid_xyz(ch) for ch in info_copy["chs"]], dtype=bool)
    if not np.all(valid_mask):
        ch_pos_upper: dict[str, np.ndarray] = {}
        for montage_name in ("colin27_1005", "fsaverage_1005", "standard_1005"):
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                ch_pos = montage.get_positions().get("ch_pos", {})
                ch_pos_upper = {
                    k.strip().upper(): np.asarray(v, dtype=np.float64)
                    for k, v in ch_pos.items()
                }
                break
            except Exception:
                continue

        if ch_pos_upper:
            for idx, is_valid in enumerate(valid_mask):
                if is_valid or mne.channel_type(info_copy, idx) != "eeg":
                    continue
                norm_key = _normalize_eeg_label(info_copy["ch_names"][idx])
                xyz = ch_pos_upper.get(norm_key)
                if (
                    xyz is not None
                    and xyz.shape == (3,)
                    and np.all(np.isfinite(xyz))
                    and np.linalg.norm(xyz) > 1e-6
                ):
                    info_copy["chs"][idx]["loc"][:3] = xyz

        valid_mask = np.array(
            [_has_valid_xyz(ch) for ch in info_copy["chs"]], dtype=bool
        )

    return info_copy, valid_mask


def plot_sensai_visualization(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    reference_cov: np.ndarray | mne.Covariance | str = "leadfield",
    epoch_duration_sec: float = 1.0,
    n_pc: int = 3,
    sensai_score: float | None = None,
    mean_enova: float | None = None,
    enova_per_channel: np.ndarray | None = None,
    title_suffix: str = "",
    show: bool = True,
):
    """Plot SENSAI subspace similarity, manifold classification, PSD, and ENOVA map.

    Replica of MATLAB's ``SENSAI_visualization.m`` with side-by-side
    Before/After subspace projections, soft pastel LDA decision shading,
    marginal KDE distributions, an ENOVA-weighted Welch PSD panel, and a
    per-channel ENOVA scalp topoplot (with bar-chart fallback when sensor
    coordinates are unavailable).

    Parameters
    ----------
    raw_before : mne.io.BaseRaw
        Original EEG recording before denoising.
    raw_after : mne.io.BaseRaw
        Cleaned EEG recording after denoising.
    reference_cov : np.ndarray | mne.Covariance | str
        Reference leadfield covariance matrix, ``mne.Covariance``, or
        ``"leadfield"`` / ``"projection"``.
    epoch_duration_sec : float
        Epoch length in seconds (default 1.0s).
    n_pc : int
        Number of principal components for SSI calculation (default 3 for EEG).
    sensai_score : float | None
        Overall SENSAI score (%) to display in title.
    mean_enova : float | None
        Mean ENOVA (%) to display in title.
    enova_per_channel : np.ndarray | None
        Optional 1D array of per-channel ENOVA values (proportion in 0-1 or
        percentage in 0-100) aligned with the common evaluated channels. When
        ``None``, computed directly from ``raw_before`` and ``raw_after``.
    title_suffix : str
        Additional parameters string for plot title.
    show : bool
        Whether to call plt.show() or return the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created visualization figure.
    metrics : dict
        Computed subspace similarity, LDA classification, ENOVA, and PSD
        metrics.
    """
    from ..covariance.covariance import _pick_cov, _resolve_reference_and_cov
    from ..metrics.enova import compute_enova_per_channel

    # 1. Align common channels and resolve reference covariance
    common_chs = [ch for ch in raw_after.ch_names if ch in raw_before.ch_names]
    if isinstance(reference_cov, mne.Covariance):
        ch_names = [ch for ch in reference_cov.ch_names if ch in common_chs]
        ref_cov_data = _pick_cov(reference_cov, ch_names).data
    elif isinstance(reference_cov, str):
        ch_names = common_chs
        info_sub = raw_after.copy().pick(ch_names).reorder_channels(ch_names).info
        ref_cov_data = _resolve_reference_and_cov(reference_cov, info_sub)[0].data
    else:
        ch_names = common_chs
        ref_cov_data = np.asarray(reference_cov, dtype=np.float64)
        if ref_cov_data.shape != (len(ch_names), len(ch_names)):
            raise ValueError(
                f"reference_cov shape {ref_cov_data.shape} does not match the "
                f"{len(ch_names)} common channels between raw_before and raw_after."
            )

    # Extract aligned channel data in microvolts (uV) for standard EEG dB scaling
    data_before = (
        raw_before.copy()
        .pick(ch_names)
        .reorder_channels(ch_names)
        .get_data(verbose=False)
        * 1e6
    )
    raw_after_picked = raw_after.copy().pick(ch_names).reorder_channels(ch_names)
    data_after = raw_after_picked.get_data(verbose=False) * 1e6
    info_eval = raw_after_picked.info
    data_noise = data_before - data_after

    sfreq = raw_after.info["sfreq"]
    epoch_samples = max(1, round(sfreq * epoch_duration_sec))
    n_ch, n_times = data_after.shape
    n_epochs = n_times // epoch_samples

    if n_epochs == 0:
        raise ValueError(
            "Recording duration "
            f"({n_times / sfreq:.2f}s) is too short for epoch duration "
            f"{epoch_duration_sec}s."
        )

    # Regularize reference covariance and extract top subspace basis
    trace_ref = np.trace(ref_cov_data) / ref_cov_data.shape[0]
    ref_cov_reg = 0.95 * ref_cov_data + 0.05 * trace_ref * np.eye(ref_cov_data.shape[0])
    _, Vref = eigh(ref_cov_reg)
    basis_ref = Vref[:, ::-1][:, :n_pc]

    # 2. Extract per-epoch Covariances, Powers (in dB), and SSIs
    def _extract_epoch_metrics(data_2d):
        ssi_list = np.zeros(n_epochs)
        power_list = np.zeros(n_epochs)
        for i in range(n_epochs):
            s = i * epoch_samples
            e = s + epoch_samples
            ep = data_2d[:, s:e]
            ep_centered = ep - np.mean(ep, axis=1, keepdims=True)
            c = (ep_centered @ ep_centered.T) / (epoch_samples - 1)

            # Power in dB (microvolts squared sum): 10 * log10(sum(diag(C)))
            pwr = np.sum(np.diag(c))
            power_list[i] = 10.0 * np.log10(max(pwr, 1e-12))

            # Subspace Similarity Index (SSI)
            try:
                _, Vc = eigh(c)
                basis_c = Vc[:, ::-1][:, :n_pc]
                S = svd(basis_c.T @ basis_ref, compute_uv=False)
                S = np.clip(S[:n_pc], -1.0, 1.0)
                ssi_list[i] = float(np.prod(S) ** (1.0 / n_pc))
            except Exception:
                ssi_list[i] = 0.0
        return power_list, ssi_list

    lpow_before, ssi_before = _extract_epoch_metrics(data_before)
    lpow_after, ssi_after = _extract_epoch_metrics(data_after)
    lpow_noise, ssi_noise = _extract_epoch_metrics(data_noise)

    ideal_power_target = float(np.median(lpow_after))

    # 3. LDA Classification on (SSI, Power)
    X_lda = np.vstack(
        [
            np.column_stack([lpow_after, ssi_after]),
            np.column_stack([lpow_noise, ssi_noise]),
        ]
    )
    y_lda = np.array([1] * n_epochs + [0] * n_epochs)

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(X_lda, y_lda)
        float(lda.score(X_lda, y_lda) * 100.0)
    except Exception:
        lda = None
        float("nan")

    try:
        # Match MATLAB SENSAI_visualization.m lines 98-101:
        # sil_scores = silhouette(X_lda(:, 1), Y_lda, 'sqEuclidean');
        # sil_signal = mean(sil_scores(Y_lda == 1));
        sil_scores = silhouette_samples(X_lda[:, [1]], y_lda, metric="sqeuclidean")
        sil_signal = float(np.mean(sil_scores[y_lda == 1]))
    except Exception:
        sil_signal = float("nan")

    # 4. Determine matched plot limits
    chi2_95 = -2.0 * np.log(1.0 - 0.95)

    def _get_extents(x):
        return [
            np.mean(x) - np.sqrt(np.var(x) * chi2_95),
            np.mean(x) + np.sqrt(np.var(x) * chi2_95),
        ]

    ext_b = _get_extents(lpow_before)
    ext_a = _get_extents(lpow_after)
    ext_n = _get_extents(lpow_noise)
    all_vals = np.concatenate(
        [lpow_before, lpow_after, lpow_noise, ext_b, ext_a, ext_n]
    )
    all_vals_finite = all_vals[np.isfinite(all_vals)]
    if len(all_vals_finite) == 0:
        x_lims = (-10.0, 10.0)
    else:
        x_min, x_max = float(np.min(all_vals_finite)), float(np.max(all_vals_finite))
        x_lims = (x_min - 2.0, x_max + 5.0)

    # 4b. Compute Per-Channel ENOVA (%) and ENOVA-Weighted Welch PSD
    if enova_per_channel is not None:
        enova_ch = np.asarray(enova_per_channel, dtype=np.float64).ravel()
        if enova_ch.shape != (n_ch,):
            raise ValueError(
                f"enova_per_channel has length {enova_ch.shape[0]}, expected {n_ch}."
            )
    else:
        enova_ch = compute_enova_per_channel(
            data_after, data_noise, epoch_samples
        ).astype(np.float64)

    enova_plot_val = enova_ch.copy()
    finite_mask = np.isfinite(enova_plot_val)
    if np.any(finite_mask) and float(np.max(enova_plot_val[finite_mask])) <= 1.5:
        enova_plot_val = enova_plot_val * 100.0
    enova_plot_val = np.where(
        np.isnan(enova_plot_val) | np.isneginf(enova_plot_val) | (enova_plot_val < 0.0),
        0.0,
        enova_plot_val,
    )
    enova_plot_val = np.where(
        np.isposinf(enova_plot_val) | (enova_plot_val > 100.0),
        100.0,
        enova_plot_val,
    )

    f_psd, psd_b_db, psd_a_db, psd_n_db, w_ch = _compute_enova_weighted_psd(
        data_before, data_after, data_noise, sfreq, enova_plot_val
    )

    # 5. Create Figure & 3-Column Layout Matching MATLAB
    fig = plt.figure(figsize=(18.0, 6.5), facecolor="white")

    # Column 1: Before Denoising + SSI Colorbar
    ax1 = fig.add_axes([0.048, 0.12, 0.195, 0.61])
    cbar_ax = fig.add_axes([0.249, 0.12, 0.009, 0.61])
    ax1_top = fig.add_axes([0.048, 0.745, 0.195, 0.080])
    ax1_top.axis("off")

    # Column 2: After Denoising + Marginals
    ax2 = fig.add_axes([0.325, 0.12, 0.195, 0.61])
    ax2_top = fig.add_axes([0.325, 0.745, 0.195, 0.080])
    ax2_right = fig.add_axes([0.526, 0.12, 0.024, 0.61])

    # Column 3 (Upper): ENOVA-weighted Welch PSD
    ax_psd = fig.add_axes([0.708, 0.530, 0.255, 0.295])

    # Column 3 (Lower): ENOVA Topoplot + Colorbar
    ax_topo = fig.add_axes([0.730, 0.080, 0.175, 0.325])
    cbar_topo_ax = fig.add_axes([0.912, 0.100, 0.010, 0.285])

    # Exact MATLAB Color Palette
    col_sig = np.array([0.08, 0.72, 0.22])  # Green
    col_noise = np.array([0.85, 0.13, 0.13])  # Red
    col_star = np.array([1.00, 0.88, 0.00])  # Gold
    col_star_dark = np.array([0.50, 0.44, 0.00])

    # Custom Parula Colormap (or turbo fallback)
    try:
        cmap_parula = plt.get_cmap("viridis")
    except Exception:
        cmap_parula = plt.cm.jet

    # Custom LDA Soft Pastel Background Colormap (Red -> White -> Green)
    lda_cmap = LinearSegmentedColormap.from_list(
        "matlab_lda_bg",
        [(0.99, 0.92, 0.92), (1.0, 1.0, 1.0), (0.93, 0.98, 0.93)],
        N=128,
    )

    # ── PANEL 1: BEFORE DENOISING ─────────────────────────────────────────
    si = np.argsort(ssi_before)
    sc1 = ax1.scatter(
        lpow_before[si],
        ssi_before[si],
        c=ssi_before[si],
        cmap=cmap_parula,
        vmin=0.0,
        vmax=1.0,
        s=38,
        edgecolors="none",
        alpha=0.75,
        zorder=3,
    )
    ax1.axhline(1.0, color=col_star, linestyle="--", lw=1.5, alpha=0.6, zorder=2)
    ax1.text(
        float(np.mean(x_lims)),
        1.10,
        "Leadfield Subspace",
        color=col_star_dark,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax1.set_xlim(x_lims)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("Epoch Power (dB)", fontsize=10.5)
    ax1.set_ylabel(f"SSI (geom. mean of top-{n_pc} PC cosines)", fontsize=10.5)
    ax1.tick_params(direction="inout", top=False, right=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1_top.set_title(
        f"Before Denoising\nMean SSI: {np.mean(ssi_before):.2f}",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    # Colorbar
    cb = fig.colorbar(sc1, cax=cbar_ax)
    cb.set_label("SSI (Subspace Similarity Index) relative to Leadfield", fontsize=9.5)
    cb.set_ticks(np.linspace(0, 1, 11))

    # ── PANEL 2: AFTER DENOISING ──────────────────────────────────────────
    # Soft Pastel LDA Shading
    if lda is not None:
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_lims[0], x_lims[1], 200), np.linspace(-0.05, 1.15, 200)
        )
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        try:
            probs = lda.predict_proba(grid_pts)[:, 1].reshape(grid_x.shape)
            ax2.imshow(
                probs,
                extent=[x_lims[0], x_lims[1], -0.05, 1.15],
                origin="lower",
                aspect="auto",
                cmap=lda_cmap,
                vmin=0.0,
                vmax=1.0,
                zorder=1,
            )
        except Exception:
            pass

    h_noise = ax2.scatter(
        lpow_noise,
        ssi_noise,
        color=col_noise,
        s=38,
        alpha=0.40,
        edgecolors="none",
        zorder=3,
        label=f"Noise (mean SSI={np.mean(ssi_noise):.2f})",
    )
    h_sig = ax2.scatter(
        lpow_after,
        ssi_after,
        color=col_sig,
        s=38,
        alpha=0.40,
        edgecolors="none",
        zorder=4,
        label=f"Signal (mean SSI={np.mean(ssi_after):.2f})",
    )
    ax2.axhline(1.0, color=col_star, linestyle="--", lw=1.5, alpha=0.6, zorder=2)
    h_star = ax2.scatter(
        [ideal_power_target],
        [1.0],
        color=col_star,
        marker="*",
        s=250,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Leadfield Subspace",
    )
    ax2.text(
        ideal_power_target,
        1.10,
        "Leadfield Subspace",
        color=col_star_dark,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )

    ax2.set_xlim(x_lims)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("Epoch Power (dB)", fontsize=10.5)
    # Redundant y-axis label omitted on Panel 2 (matching MATLAB)
    ax2.tick_params(direction="inout", top=False, right=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Boxed Legend matching MATLAB
    ax2.legend(
        handles=[h_star, h_sig, h_noise],
        labels=[
            "Leadfield Subspace",
            f"Signal (mean SSI={np.mean(ssi_after):.2f})",
            f"Noise (mean SSI={np.mean(ssi_noise):.2f})",
        ],
        loc="lower left",
        bbox_to_anchor=(1.03, 1.02),
        frameon=True,
        edgecolor="black",
        facecolor="white",
        fontsize=8.0,
    )

    # ── PANEL 2 MARGINALS (KDE Distributions) ─────────────────────────────
    # Top Marginal (Power distribution)
    try:
        kde_pwr_sig = gaussian_kde(lpow_after)
        kde_pwr_noi = gaussian_kde(lpow_noise)
        x_grid = np.linspace(x_lims[0], x_lims[1], 300)

        ax2_top.fill_between(
            x_grid,
            kde_pwr_sig(x_grid),
            color=col_sig,
            alpha=0.25,
            edgecolor=col_sig,
            lw=1.2,
        )
        ax2_top.fill_between(
            x_grid,
            kde_pwr_noi(x_grid),
            color=col_noise,
            alpha=0.25,
            edgecolor=col_noise,
            lw=1.2,
        )
    except Exception:
        pass
    ax2_top.set_xlim(x_lims)
    ax2_top.set_xticks([])
    ax2_top.set_yticks([])
    ax2_top.spines["top"].set_visible(False)
    ax2_top.spines["right"].set_visible(False)
    ax2_top.spines["left"].set_color("#888888")
    ax2_top.spines["bottom"].set_color("#888888")
    ax2_top.set_facecolor("none")

    sil_str = (
        f"\nSSI Silhouette Score: {sil_signal:.2f}" if not np.isnan(sil_signal) else ""
    )
    ax2_top.set_title(
        "After Denoising\n"
        f"Mean SSSI: {np.mean(ssi_after):.2f}  |  "
        f"Mean NSSI: {np.mean(ssi_noise):.2f}{sil_str}",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )

    # Right Marginal (SSI distribution)
    try:
        kde_ssi_sig = gaussian_kde(ssi_after)
        kde_ssi_noi = gaussian_kde(ssi_noise)
        y_grid = np.linspace(-0.05, 1.15, 300)

        ax2_right.fill_betweenx(
            y_grid,
            kde_ssi_sig(y_grid),
            color=col_sig,
            alpha=0.25,
            edgecolor=col_sig,
            lw=1.2,
        )
        ax2_right.fill_betweenx(
            y_grid,
            kde_ssi_noi(y_grid),
            color=col_noise,
            alpha=0.25,
            edgecolor=col_noise,
            lw=1.2,
        )
    except Exception:
        pass
    ax2_right.set_ylim(-0.05, 1.15)
    ax2_right.set_xticks([])
    ax2_right.set_yticks([])
    ax2_right.spines["top"].set_visible(False)
    ax2_right.spines["right"].set_visible(False)
    ax2_right.spines["left"].set_color("#888888")
    ax2_right.spines["bottom"].set_color("#888888")
    ax2_right.set_facecolor("none")

    # ── PANEL 3 (UPPER): ENOVA-WEIGHTED PSD OVER CHANNELS ────────────────
    max_f_disp = min(45.0, float(sfreq) / 2.0)
    f_mask = (f_psd >= 0.5) & (f_psd <= max_f_disp)
    if not np.any(f_mask):
        f_mask = np.ones_like(f_psd, dtype=bool)
    f_disp = f_psd[f_mask]

    ax_psd.plot(
        f_disp,
        psd_b_db[f_mask],
        "-",
        color=(0.25, 0.45, 0.75),
        lw=1.8,
        label="Original",
    )
    ax_psd.plot(
        f_disp,
        psd_a_db[f_mask],
        "-",
        color=(0.08, 0.68, 0.25),
        lw=2.0,
        label="Signal",
    )
    ax_psd.plot(
        f_disp,
        psd_n_db[f_mask],
        "-",
        color=(0.85, 0.18, 0.18),
        lw=1.8,
        label="Noise",
    )
    if f_disp.size >= 2:
        ax_psd.set_xlim(float(f_disp[0]), float(f_disp[-1]))
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=10.5)
    ax_psd.set_ylabel("Power (dB)", fontsize=10.5)
    ax_psd.set_title("ENOVA-weighted PSD over channels", fontsize=10, fontweight="bold")
    ax_psd.grid(True, alpha=0.3)
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.legend(loc="upper right", frameon=False, fontsize=8.5)

    # ── PANEL 3 (LOWER): ENOVA TOPOPLOT (OR BAR FALLBACK) ────────────────
    info_topo, valid_mask = _prepare_topomap_info(info_eval)
    plot_done = False
    if int(np.sum(valid_mask)) >= 3:
        valid_chs = [ch for ch, ok in zip(ch_names, valid_mask, strict=True) if ok]
        info_valid = mne.pick_info(
            info_topo,
            mne.pick_channels(info_topo.ch_names, include=valid_chs, ordered=True),
        )
        try:
            im, _ = mne.viz.plot_topomap(
                enova_plot_val[valid_mask],
                info_valid,
                axes=ax_topo,
                cmap="turbo",
                vlim=(0.0, 100.0),
                contours=6,
                outlines="head",
                sensors=True,
                show=False,
            )
            cb_enova = fig.colorbar(im, cax=cbar_topo_ax)
            cb_enova.set_label("ENOVA (%)", fontsize=10)
            cb_enova.set_ticks([0, 20, 40, 60, 80, 100])
            ax_topo.set_title(
                "ENOVA (Explained Noise Variance)",
                fontsize=10,
                fontweight="bold",
                pad=8,
            )
            plot_done = True
        except Exception:
            ax_topo.cla()

    if not plot_done:
        cbar_topo_ax.set_visible(False)
        ax_topo.set_position([0.708, 0.115, 0.255, 0.290])
        x_idx = np.arange(n_ch)
        ax_topo.bar(
            x_idx,
            enova_plot_val,
            color=(0.85, 0.35, 0.25),
            edgecolor="none",
        )
        step = max(1, n_ch // 20)
        tick_idx = x_idx[::step]
        ax_topo.set_xticks(tick_idx)
        ax_topo.set_xticklabels(
            [ch_names[i] for i in tick_idx],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax_topo.set_ylim(0.0, 105.0)
        ax_topo.set_ylabel("ENOVA (%)", fontsize=10.5)
        ax_topo.set_title(
            "ENOVA (Explained Noise Variance)",
            fontsize=10,
            fontweight="bold",
        )
        ax_topo.grid(True, axis="y", alpha=0.3)
        ax_topo.spines["top"].set_visible(False)
        ax_topo.spines["right"].set_visible(False)

    # ── GLOBAL TITLE ──────────────────────────────────────────────────────
    title_parts = []
    if sensai_score is not None:
        title_parts.append(f"SENSAI = {sensai_score:.1f}%")
    if mean_enova is not None:
        enova_val = mean_enova * 100 if mean_enova <= 1.0 else mean_enova
        title_parts.append(f"ENOVA = {enova_val:.1f}%")
    if title_suffix:
        title_parts.append(f"[{title_suffix}]")
    full_title = (
        ", ".join(title_parts)
        if title_parts
        else "SENSAI Visualization: Subspace Similarity vs Epoch Power"
    )

    fig.suptitle(full_title, fontsize=12, fontweight="bold", y=0.985)

    metrics = {
        "ssi_before_mean": float(np.mean(ssi_before)),
        "ssi_after_mean": float(np.mean(ssi_after)),
        "ssi_noise_mean": float(np.mean(ssi_noise)),
        "signal_silhouette": sil_signal,
        "ideal_power_target_db": ideal_power_target,
        "enova_per_channel": enova_ch,
        "psd_before_db": psd_b_db,
        "psd_after_db": psd_a_db,
        "psd_noise_db": psd_n_db,
        "psd_freqs": f_psd,
        "enova_weights": w_ch,
    }

    if show:
        plt.show()

    return fig, metrics
