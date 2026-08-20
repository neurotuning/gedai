import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eigh, svd
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score


def plot_sensai_visualization(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    reference_cov: np.ndarray | str = "leadfield",
    epoch_duration_sec: float = 1.0,
    n_pc: int = 3,
    sensai_score: float | None = None,
    mean_enova: float | None = None,
    title_suffix: str = "",
    show: bool = True,
):
    """Plot 2D SENSAI Subspace Similarity vs Epoch Power Scatter & Manifold Classification.

    Exact replica of MATLAB's SENSAI_visualization.m with side-by-side Before/After
    subspace projections, soft pastel LDA decision shading, and marginal KDE distributions.

    Parameters
    ----------
    raw_before : mne.io.BaseRaw
        Original EEG recording before denoising.
    raw_after : mne.io.BaseRaw
        Cleaned EEG recording after denoising.
    reference_cov : np.ndarray or str
        Reference leadfield covariance matrix.
    epoch_duration_sec : float
        Epoch length in seconds (default 1.0s).
    n_pc : int
        Number of principal components for SSI calculation (default 3 for EEG).
    sensai_score : float | None
        Overall SENSAI score (%) to display in title.
    mean_enova : float | None
        Mean ENOVA (%) to display in title.
    title_suffix : str
        Additional parameters string for plot title.
    show : bool
        Whether to call plt.show() or return the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created visualization figure.
    metrics : dict
        Computed subspace similarity and LDA classification metrics.
    """
    from ..covariance.covariance import _ensure_cov, _pick_cov

    # Extract common channels and data in microvolts (uV) for standard EEG dB scaling
    ch_names = [ch for ch in raw_after.ch_names if ch in raw_before.ch_names]
    data_before = raw_before.copy().pick(ch_names).get_data(verbose=False) * 1e6
    data_after = raw_after.copy().pick(ch_names).get_data(verbose=False) * 1e6
    data_noise = data_before - data_after

    sfreq = raw_after.info["sfreq"]
    epoch_samples = max(1, round(sfreq * epoch_duration_sec))
    n_ch, n_times = data_after.shape
    n_epochs = n_times // epoch_samples

    if n_epochs == 0:
        raise ValueError(f"Recording duration ({n_times/sfreq:.2f}s) is too short for epoch duration {epoch_duration_sec}s.")

    # 1. Prepare Reference Subspace
    if isinstance(reference_cov, str):
        cov = _ensure_cov(reference_cov)
        cov = _pick_cov(cov, ch_names)
        ref_cov_data = cov.data
    else:
        ref_cov_data = np.asarray(reference_cov)

    # Regularize reference covariance
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
    X_lda = np.vstack([
        np.column_stack([lpow_after, ssi_after]),
        np.column_stack([lpow_noise, ssi_noise]),
    ])
    y_lda = np.array([1] * n_epochs + [0] * n_epochs)

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(X_lda, y_lda)
        lda_accuracy = float(lda.score(X_lda, y_lda) * 100.0)
    except Exception:
        lda = None
        lda_accuracy = float("nan")

    try:
        sil_signal = float(silhouette_score(X_lda[:, [1]], y_lda))
    except Exception:
        sil_signal = float("nan")

    # 4. Determine matched plot limits
    chi2_95 = -2.0 * np.log(1.0 - 0.95)
    def _get_extents(x):
        return [np.mean(x) - np.sqrt(np.var(x) * chi2_95), np.mean(x) + np.sqrt(np.var(x) * chi2_95)]

    ext_b = _get_extents(lpow_before)
    ext_a = _get_extents(lpow_after)
    ext_n = _get_extents(lpow_noise)
    all_vals = np.concatenate([lpow_before, lpow_after, lpow_noise, ext_b, ext_a, ext_n])
    all_vals_finite = all_vals[np.isfinite(all_vals)]
    if len(all_vals_finite) == 0:
        x_lims = (-10.0, 10.0)
    else:
        x_min, x_max = float(np.min(all_vals_finite)), float(np.max(all_vals_finite))
        x_lims = (x_min - 2.0, x_max + 5.0)

    # 5. Create Figure & Layout Matching MATLAB
    fig = plt.figure(figsize=(15.5, 6.8), facecolor="white")

    # Outer split: Left panel (0.06 to 0.44), Right panel (0.52 to 0.94)
    # Left Panel
    ax1 = fig.add_axes([0.06, 0.10, 0.32, 0.70])
    cbar_ax = fig.add_axes([0.395, 0.10, 0.015, 0.70])

    # Right Panel Main + Marginals
    ax2 = fig.add_axes([0.52, 0.10, 0.33, 0.70])
    ax2_top = fig.add_axes([0.52, 0.81, 0.33, 0.07])
    ax2_right = fig.add_axes([0.855, 0.10, 0.035, 0.70])

    # Exact MATLAB Color Palette
    col_sig = np.array([0.08, 0.72, 0.22])      # Green
    col_noise = np.array([0.85, 0.13, 0.13])    # Red
    col_star = np.array([1.00, 0.88, 0.00])     # Gold
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
        N=128
    )

    # ── PANEL 1: BEFORE DENOISING ─────────────────────────────────────────
    si = np.argsort(ssi_before)
    sc1 = ax1.scatter(
        lpow_before[si], ssi_before[si], c=ssi_before[si], cmap=cmap_parula,
        vmin=0.0, vmax=1.0, s=38, edgecolors="none", alpha=0.75, zorder=3
    )
    ax1.axhline(1.0, color=col_star, linestyle="--", lw=1.5, alpha=0.6, zorder=2)
    ax1.text(
        float(np.mean(x_lims)), 1.10, "Leadfield Subspace",
        color=col_star_dark, fontsize=10, fontweight="bold", ha="center"
    )
    ax1.set_xlim(x_lims)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("Epoch Power (dB)", fontsize=11)
    ax1.set_ylabel(f"SSI (geom. mean of top-{n_pc} PC cosines)", fontsize=11)
    ax1.tick_params(direction="inout", top=False, right=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(f"Before Denoising  |  Mean SSI: {np.mean(ssi_before):.2f}", fontsize=10, pad=20)

    # Colorbar
    cb = fig.colorbar(sc1, cax=cbar_ax)
    cb.set_label("SSI (Subspace Similarity Index) relative to Leadfield", fontsize=10)
    cb.set_ticks(np.linspace(0, 1, 11))

    # ── PANEL 2: AFTER DENOISING ──────────────────────────────────────────
    # Soft Pastel LDA Shading
    if lda is not None:
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_lims[0], x_lims[1], 200),
            np.linspace(-0.05, 1.15, 200)
        )
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        try:
            probs = lda.predict_proba(grid_pts)[:, 1].reshape(grid_x.shape)
            ax2.imshow(
                probs, extent=[x_lims[0], x_lims[1], -0.05, 1.15],
                origin="lower", aspect="auto", cmap=lda_cmap, vmin=0.0, vmax=1.0, zorder=1
            )
        except Exception:
            pass

    h_noise = ax2.scatter(
        lpow_noise, ssi_noise, color=col_noise, s=38, alpha=0.40,
        edgecolors="none", zorder=3, label=f"Noise (mean SSI={np.mean(ssi_noise):.2f})"
    )
    h_sig = ax2.scatter(
        lpow_after, ssi_after, color=col_sig, s=38, alpha=0.40,
        edgecolors="none", zorder=4, label=f"Signal (mean SSI={np.mean(ssi_after):.2f})"
    )
    ax2.axhline(1.0, color=col_star, linestyle="--", lw=1.5, alpha=0.6, zorder=2)
    h_star = ax2.scatter(
        [ideal_power_target], [1.0], color=col_star, marker="*", s=250,
        edgecolors="black", linewidths=1.0, zorder=5, label="Leadfield Subspace"
    )
    ax2.text(
        ideal_power_target, 1.10, "Leadfield Subspace",
        color=col_star_dark, fontsize=10, fontweight="bold", ha="center"
    )

    ax2.set_xlim(x_lims)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("Epoch Power (dB)", fontsize=11)
    ax2.set_ylabel(f"SSI (geom. mean of top-{n_pc} PC cosines)", fontsize=11)
    ax2.tick_params(direction="inout", top=False, right=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Boxed Legend matching MATLAB
    leg = ax2.legend(
        handles=[h_star, h_sig, h_noise],
        labels=["Leadfield Subspace", f"Signal (mean SSI={np.mean(ssi_after):.2f})", f"Noise (mean SSI={np.mean(ssi_noise):.2f})"],
        loc="upper right", bbox_to_anchor=(1.40, 1.02),
        frameon=True, edgecolor="black", facecolor="white", fontsize=9
    )

    # ── PANEL 2 MARGINALS (KDE Distributions) ─────────────────────────────
    # Top Marginal (Power distribution)
    try:
        kde_pwr_sig = gaussian_kde(lpow_after)
        kde_pwr_noi = gaussian_kde(lpow_noise)
        x_grid = np.linspace(x_lims[0], x_lims[1], 300)
        
        ax2_top.fill_between(x_grid, kde_pwr_sig(x_grid), color=col_sig, alpha=0.25, edgecolor=col_sig, lw=1.2)
        ax2_top.fill_between(x_grid, kde_pwr_noi(x_grid), color=col_noise, alpha=0.25, edgecolor=col_noise, lw=1.2)
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

    sil_str = f"\nSSI Silhouette Score: {sil_signal:.2f}" if not np.isnan(sil_signal) else ""
    ax2_top.set_title(
        f"After Denoising  |  Mean SSSI: {np.mean(ssi_after):.2f}  |  Mean NSSI: {np.mean(ssi_noise):.2f}{sil_str}",
        fontsize=9, pad=10
    )

    # Right Marginal (SSI distribution)
    try:
        kde_ssi_sig = gaussian_kde(ssi_after)
        kde_ssi_noi = gaussian_kde(ssi_noise)
        y_grid = np.linspace(-0.05, 1.15, 300)

        ax2_right.fill_betweenx(y_grid, kde_ssi_sig(y_grid), color=col_sig, alpha=0.25, edgecolor=col_sig, lw=1.2)
        ax2_right.fill_betweenx(y_grid, kde_ssi_noi(y_grid), color=col_noise, alpha=0.25, edgecolor=col_noise, lw=1.2)
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

    # ── GLOBAL TITLE ──────────────────────────────────────────────────────
    title_parts = []
    if sensai_score is not None:
        title_parts.append(f"SENSAI = {sensai_score:.0f}%")
    if mean_enova is not None:
        enova_val = mean_enova * 100 if mean_enova <= 1.0 else mean_enova
        title_parts.append(f"ENOVA = {enova_val:.0f}%")
    if title_suffix:
        title_parts.append(f"[{title_suffix}]")
    full_title = ", ".join(title_parts) if title_parts else "SENSAI Visualization: Subspace Similarity vs Epoch Power"

    fig.suptitle(full_title, fontsize=12, fontweight="bold", y=0.985)

    metrics = {
        "ssi_before_mean": float(np.mean(ssi_before)),
        "ssi_after_mean": float(np.mean(ssi_after)),
        "ssi_noise_mean": float(np.mean(ssi_noise)),
        "signal_silhouette": sil_signal,
        "ideal_power_target_db": ideal_power_target,
    }

    if show:
        plt.show()

    return fig, metrics
