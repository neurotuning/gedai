from importlib import resources
from pathlib import Path


def get_data_path(filename: str) -> Path:
    """Return the absolute path of a packaged GEDAI data file."""
    return Path(resources.files("gedai").joinpath(f"data/{filename}"))


def get_leadfield_cov_path() -> Path:
    """Return the path to the bundled default leadfield covariance file."""
    return get_data_path("fsavLEADFIELD_4_GEDAI-cov.fif")


def get_simulated_clean_eeg_set_path() -> Path:
    """Return the path to the bundled clean EEG sample dataset (.set)."""
    return get_data_path("simulated_clean_EEG_2.set")


def get_contaminated_eeg_set_path() -> Path:
    """Return the path to the bundled contaminated EEG sample dataset (.set)."""
    return get_data_path(
        "SNR=0.35481 contamination=25 clean_EEG_dataset_2.set + "
        "EOG_EMG_NOISE_dataset_1.set"
    )


__all__ = [
    "get_contaminated_eeg_set_path",
    "get_data_path",
    "get_leadfield_cov_path",
    "get_simulated_clean_eeg_set_path",
]
