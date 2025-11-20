import matplotlib.pyplot as plt
import numpy as np


def plot_mne_style_overlay_interactive(raw_noisy, raw_clean, title=None, duration=4.0):
    """Plot interactive overlay of noisy and cleaned recordings."""
    noisy_data = raw_noisy.get_data()
    clean_data = raw_clean.get_data()
    times = raw_noisy.times
    n_chans = noisy_data.shape[0]
    spacing = np.max(np.ptp(noisy_data, axis=1)) * 1.2
    start_idx = 0
    end_idx = np.searchsorted(times, duration)
    plot_mode = "overlay"  # 'overlay', 'diff', 'denoised_only', or 'noisy_only'
    mag_scale = 5.0
    fig, ax = plt.subplots(figsize=(12, 0.5 * n_chans + 4))
    offsets = np.arange(n_chans) * spacing
    ylim = (-spacing, offsets[-1] + spacing)

    def plot_window():
        ax.clear()
        ax.set_ylim(ylim)
        if plot_mode == "diff":
            for ch in range(n_chans):
                diff = (
                    noisy_data[ch, start_idx:end_idx]
                    - clean_data[ch, start_idx:end_idx]
                ) * mag_scale
                ax.plot(
                    times[start_idx:end_idx],
                    diff + offsets[ch],
                    color="purple",
                    alpha=0.8,
                    linewidth=0.8,
                    label="Noisy - Cleaned" if ch == 0 else None,
                )
            ax.set_ylabel("Channels")
            ax.legend(loc="upper right")
            if title:
                ax.set_title(title + " (Difference)")
        elif plot_mode == "denoised_only":
            for ch in range(n_chans):
                ax.plot(
                    times[start_idx:end_idx],
                    clean_data[ch, start_idx:end_idx] * mag_scale + offsets[ch],
                    color="blue",
                    alpha=0.7,
                    linewidth=0.8,
                    label="Cleaned" if ch == 0 else None,
                )
            ax.set_ylabel("Channels")
            ax.legend(loc="upper right")
            if title:
                ax.set_title(title + " (Denoised only)")
        elif plot_mode == "noisy_only":
            for ch in range(n_chans):
                ax.plot(
                    times[start_idx:end_idx],
                    noisy_data[ch, start_idx:end_idx] * mag_scale + offsets[ch],
                    color="red",
                    alpha=0.7,
                    linewidth=0.8,
                    label="Noisy" if ch == 0 else None,
                )
            ax.set_ylabel("Channels")
            ax.legend(loc="upper right")
            if title:
                ax.set_title(title + " (Noisy only)")
        else:  # overlay
            for ch in range(n_chans):
                ax.plot(
                    times[start_idx:end_idx],
                    noisy_data[ch, start_idx:end_idx] * mag_scale + offsets[ch],
                    color="red",
                    alpha=0.7,
                    linewidth=0.8,
                    label="Noisy" if ch == 0 else None,
                )
                ax.plot(
                    times[start_idx:end_idx],
                    clean_data[ch, start_idx:end_idx] * mag_scale + offsets[ch],
                    color="blue",
                    alpha=0.7,
                    linewidth=0.8,
                    label="Cleaned" if ch == 0 else None,
                )
            ax.set_ylabel("Channels")
            ax.legend(loc="upper right")
            if title:
                ax.set_title(title)
        ax.set_yticks(offsets)
        ax.set_yticklabels(raw_noisy.ch_names)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(times[start_idx], times[end_idx - 1])
        fig.tight_layout()
        fig.canvas.draw()

    def on_key(event):
        nonlocal start_idx, end_idx, plot_mode, mag_scale
        window_len = end_idx - start_idx
        step = int(window_len // 4)
        if event.key == "right":
            if end_idx + step < len(times):
                start_idx += step
                end_idx += step
            else:
                start_idx = len(times) - window_len
                end_idx = len(times)
        elif event.key == "left":
            if start_idx - step >= 0:
                start_idx -= step
                end_idx -= step
            else:
                start_idx = 0
                end_idx = window_len
        elif event.key == "up":
            mag_scale *= 1.2
        elif event.key == "down":
            mag_scale /= 1.2
        elif event.key.lower() == "d":
            if plot_mode == "diff":
                plot_mode = "overlay"
            else:
                plot_mode = "diff"
        elif event.key.lower() == "n":
            if plot_mode == "denoised_only":
                plot_mode = "overlay"
            else:
                plot_mode = "denoised_only"
        elif event.key.lower() == "o":
            if plot_mode == "noisy_only":
                plot_mode = "overlay"
            else:
                plot_mode = "noisy_only"
        plot_window()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plot_window()
    plt.show(block=True)
    return fig, ax
