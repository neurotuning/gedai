import numpy as np
import mne


def _enova(original_data, cleaned_data):
    """" Computes the explained noise variance (ENOVA)"""
    var_original = np.var(original_data, ddof=1)  
    var_noise = np.var(cleaned_data - original_data, ddof=1)
    enova_per_epoch = var_noise / var_original
    return enova_per_epoch


def create_bad_enova_annotations(raw, raw_corrected, duration=2, overlap=0, threshold=0.9, annotation_description='BAD_ENOVA'):
    """Creates annotations for bad segments using ENOVA.

    The function uses a Explained Noise Variance (ENOVA) metric to evaluate 
    how much of the original signal's variance was removed by the correction 
    process. Segments with high ENOVA values indicate that a significant portion
    of the original signal's variance was removed, which may suggest that the
    original signal quality does not allow for effective correction.
    Sliding windows of duration ``duration`` with an overlap of ``overlap`` 
    seconds are built across the data, and ENOVA values are computed for each
    segment. The function then applies a threshold to the ENOVA values to identify
    bad segments. :class:`mne.Annotations` are created for the identified bad segments,
    which can be added to the original raw data.

    
    Parameters
    ----------
    raw : mne.io.Raw
        The original raw data.
    raw_corrected : mne.io.Raw
        The corrected raw data.
    duration : float
        The duration of each epoch in seconds.
    overlap : float
        The overlap between consecutive epochs in seconds.
    threshold : float, optional
        The threshold for identifying bad epochs based on ENOVA. Default is 0.9.
    annotation_description : str, optional
        The description for the bad epoch annotations. Default is 'BAD_ENOVA'.

    Returns
    -------
    annotations : mne.Annotations
        The annotations for the bad epochs.
    """
    # Get data from MNE Raw objects
    starts, stops = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs_orig = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, reject_by_annotation=False)
    epochs_corr = mne.make_fixed_length_epochs(raw_corrected, duration=duration, overlap=overlap, reject_by_annotation=False)

    enova_values = []
    for epoch_orig, epoch_corr in zip(epochs_orig, epochs_corr):
        enova_per_epoch = _enova(epoch_orig, epoch_corr)
        enova_values.append(enova_per_epoch)

    # Identify bad epochs based on ENOVA threshold
    bad_epochs = [i for i, enova in enumerate(enova_values) if enova > threshold]
    # Bad annotations
    annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.info['meas_date'])
    for bad_epoch in bad_epochs:
        start_time = bad_epoch * (duration - overlap)
        annotations.append(onset=start_time, duration=duration, description=annotation_description)
    return annotations


def _channel_enova(original_data, cleaned_data):
    """Computes the explained noise variance (ENOVA) for each channel."""
    var_original = np.var(original_data, axis=1, ddof=1)  
    var_noise = np.var(cleaned_data - original_data, axis=1, ddof=1)
    enova_per_channel = var_noise / var_original
    return enova_per_channel