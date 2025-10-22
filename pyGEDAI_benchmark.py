import mne
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore
from itertools import product
import random
import time
import warnings # Import the warnings module
import matplotlib.pyplot as plt # Import matplotlib for potential plt.close()

# Import the GEDAI denoising function from the local module
from pygedai_EEGLAB_denoiser import pygedai_denoise_EEGLAB_data
# Import the plotting function from the viz subfolder
from gedai.viz.compare import plot_mne_style_overlay_interactive

# --- Helper Functions (Translated from MATLAB) ---

def sig_to_noise(ground_truth_matrix, denoised_matrix):
    """
    Calculates Signal-to-Noise Ratio (SNR) in dB.
    Translated from MATLAB's sig_to_noise function.
    """
    original_signal_power = np.var(ground_truth_matrix.flatten())
    residual_noise_power = np.var(denoised_matrix - ground_truth_matrix)
    if residual_noise_power == 0:
        return np.inf  # Infinite SNR if no residual noise
    SNR = 10 * np.log10(original_signal_power / residual_noise_power)
    return SNR

def correlation_coefficient(ground_truth_matrix, denoised_matrix):
    """
    Calculates the Pearson correlation coefficient (R).
    """
    gt_flat = ground_truth_matrix.flatten()
    denoised_flat = denoised_matrix.flatten()

    if len(gt_flat) == 0 or len(denoised_flat) == 0:
        return 0.0

    # Check for constant data, which would lead to NaN correlation
    if np.all(gt_flat == gt_flat[0]) or np.all(denoised_flat == denoised_flat[0]):
        return 0.0 # Cannot compute correlation if data is constant

    R = np.corrcoef(gt_flat, denoised_flat)[0, 1]
    return R

def relative_RMSE(ground_truth_matrix, denoised_matrix):
    """
    Calculates Relative Root Mean Square Error (RRMSE).
    Translated from MATLAB's relative_RMSE function.
    """
    squared_error = (ground_truth_matrix - denoised_matrix)**2
    RMSE = np.sqrt(np.mean(squared_error))
    rms_ground_truth = np.sqrt(np.mean(ground_truth_matrix**2))
    if rms_ground_truth == 0:
        return np.inf  # Avoid division by zero
    RRMSE = RMSE / rms_ground_truth
    return RRMSE

def retain_exact_percentage_random_eeg_blocks(data_matrix, percentage_to_keep, min_block_size, max_block_size):
    """
    Retains NON-OVERLAPPING blocks to meet an exact target percentage.
    Translated from MATLAB's retainExactPercentageRandomEEGBlocks function.
    """
    if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2 or data_matrix.size == 0:
        raise ValueError('Input data must be a non-empty 2D NumPy array.')
    if not (0 <= percentage_to_keep <= 100):
        raise ValueError('percentage_to_keep must be between 0 and 100.')
    if not (isinstance(min_block_size, int) and min_block_size > 0):
        raise ValueError('min_block_size must be a positive integer.')
    if not (isinstance(max_block_size, int) and max_block_size > 0 and max_block_size >= min_block_size):
        raise ValueError('max_block_size must be a positive integer >= min_block_size.')

    rows, num_cols = data_matrix.shape

    kept_column_indices = np.array([], dtype=int)
    zeroed_column_indices = np.arange(num_cols)
    actual_block_sizes = []
    actual_k_num_blocks = 0
    modified_matrix = np.zeros_like(data_matrix)

    # Fast-path for 100% case to avoid complex logic and potential floating point issues
    if percentage_to_keep == 100:
        kept_column_indices = np.arange(num_cols)
        zeroed_column_indices = np.array([], dtype=int)
        actual_block_sizes = [num_cols]
        actual_k_num_blocks = 1
        return data_matrix.copy(), kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks

    if percentage_to_keep == 0:
        return modified_matrix, kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks

    target_total_columns = round(num_cols * percentage_to_keep / 100)

    if target_total_columns == 0:
        print(f'Warning: Target percentage {percentage_to_keep:.2f}% results in 0 columns to keep after rounding for {num_cols} total columns.')
        return modified_matrix, kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks

    if target_total_columns > num_cols:
        raise ValueError(f'Target percentage {percentage_to_keep:.2f}% results in {target_total_columns} columns, which is more than the available {num_cols} columns.')

    # --- Iteratively Generate Block Sizes ---
    current_total_columns = 0
    while True:
        remaining_target = target_total_columns - current_total_columns
        if remaining_target <= 0:
            break

        max_possible_next_size = min(max_block_size, remaining_target)
        min_possible_next_size = min(min_block_size, max_possible_next_size)

        if current_total_columns + 1 > num_cols:
            print(f'Warning: Cannot add more columns; stopping short of the target {target_total_columns}. Current total: {current_total_columns}.')
            break

        if current_total_columns + min_possible_next_size > num_cols:
            print(f'Warning: Cannot fit another block respecting constraints without exceeding total columns; stopping short of the target {target_total_columns}. Current total: {current_total_columns}.')
            break

        if max_possible_next_size < min_block_size:
            next_block_size = remaining_target
            if current_total_columns + next_block_size > num_cols:
                print(f'Warning: Cannot fit the final remainder block; stopping short of the target {target_total_columns}. Current total: {current_total_columns}.')
                break
        else:
            next_block_size = random.randint(min_possible_next_size, max_possible_next_size)

        actual_block_sizes.append(next_block_size)
        current_total_columns += next_block_size
        actual_k_num_blocks += 1

        if next_block_size == remaining_target:
            break

    if current_total_columns != target_total_columns:
        print(f'Warning: Could not achieve exact target of {target_total_columns} columns. Achieved {current_total_columns} columns due to space constraints.')
        target_total_columns = current_total_columns

    if actual_k_num_blocks == 0 and target_total_columns > 0:
        print('Warning: No blocks were selected despite a non-zero target. Target percentage might be too low or min_block_size too large relative to total columns.')
        return modified_matrix, kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks
    elif actual_k_num_blocks == 0 and target_total_columns == 0:
        return modified_matrix, kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks

    total_columns_needed = target_total_columns

    # --- Core Logic (Non-Overlapping Selection using Gap Method) ---
    available_gap_space = num_cols - total_columns_needed
    if available_gap_space < 0:
        raise RuntimeError(f'Internal logic error: Available gap space ({available_gap_space}) is negative. total_needed={total_columns_needed}, num_cols={num_cols}')

    k = actual_k_num_blocks
    if k == 0:
        gap_lengths = [available_gap_space]
    else:
        # Correctly handle the case where available_gap_space is 0
        # This logic mirrors MATLAB's randperm(N, K) where N = available_gap_space + k
        if available_gap_space + k > 0:
            dividers = np.sort(np.random.choice(np.arange(1, available_gap_space + k + 1), k, replace=False))
            gap_lengths = np.diff(np.concatenate(([0], dividers, [available_gap_space + k + 1]))) - 1
        else:  # No gap space (total_columns_needed == num_cols)
            gap_lengths = np.zeros(k + 1, dtype=int)

    kept_column_indices_list = [] # Use a list for dynamic appending
    current_column = 0 # Starting column position in the matrix (0-indexed)

    for i in range(k):
        # Add the gap *before* the current block
        current_column += gap_lengths[i]

        start_col = current_column
        block_size = actual_block_sizes[i]
        end_col = current_column + block_size

        if end_col > num_cols:
            raise RuntimeError(f'Internal logic error: Calculated end column {end_col} exceeds matrix dimension {num_cols} for block {i}.')
        if block_size <= 0:
            raise RuntimeError(f'Internal logic error: Calculated block size {block_size} is non-positive for block {i}.')

        kept_column_indices_list.extend(range(start_col, end_col))
        current_column = end_col

    # Add the last gap (if any)
    current_column += gap_lengths[k]
    
    kept_column_indices = np.sort(np.array(kept_column_indices_list, dtype=int))

    if len(kept_column_indices) != total_columns_needed:
        raise RuntimeError(f'Fatal internal error: Final number of kept indices ({len(kept_column_indices)}) does not match target ({total_columns_needed}).')

    zeroed_column_indices = np.setdiff1d(np.arange(num_cols), kept_column_indices)

    if len(kept_column_indices) > 0:
        modified_matrix[:, kept_column_indices] = data_matrix[:, kept_column_indices]

    return modified_matrix, kept_column_indices, zeroed_column_indices, actual_block_sizes, actual_k_num_blocks


# --- Main Benchmark Script Logic ---

if __name__ == "__main__":
    # Configuration parameters
    contaminated_signal_proportion = [100]  # percent of epochs temporally contaminated
    signal_to_noise_in_db = [-9,]  # initial data signal-to-noise ratio in decibels
    
    # Set to True to display interactive plots for each combination (will pause execution)
    generate_individual_plots = False

    # Suppress all warnings for cleaner output during benchmark
    warnings.filterwarnings('ignore')

    # Data directories (adjust these paths as needed)
    # Assuming the script is run from the pyGEDAI directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct paths relative to the pyGEDAI directory using raw strings to handle backslashes
    clean_eeg_dir = r'C:\Users\Ros\Documents\EEG data\new 4GEDAI paper\DENOISING SIMULATIONS\EMPIRICAL analysis\CLEAN EEG'
    artifact_eeg_dir = r'C:\Users\Ros\Documents\EEG data\new 4GEDAI paper\DENOISING SIMULATIONS\EMPIRICAL analysis\artifacts_test'

    # For demonstration, let's use placeholder paths if the full path doesn't exist
    # In a real scenario, these should point to actual data.
    if not os.path.exists(clean_eeg_dir):
        print(f"Warning: Clean EEG directory not found at {clean_eeg_dir}. Using a hardcoded placeholder path. Please update if incorrect.")
        clean_eeg_dir = r"C:\Users\Ros\Documents\EEG data\new 4GEDAI paper\DENOISING SIMULATIONS\EMPIRICAL analysis\CLEAN EEG"
    if not os.path.exists(artifact_eeg_dir):
        print(f"Warning: Artifact EEG directory not found at {artifact_eeg_dir}. Using a hardcoded placeholder path. Please update if incorrect.")
        artifact_eeg_dir = r"C:\Users\Ros\Documents\EEG data\new 4GEDAI paper\DENOISING SIMULATIONS\EMPIRICAL analysis\ARTIFACTS\EMG"


    just_testing = False
    random_number_of_test_files = 50

    # Initialize result DataFrames
    results_correlation = pd.DataFrame(columns=['clean_EEG_file', 'artifact_file', 'artifact', 'temporal_contamination', 'signal_to_noise', 'Algorithm', 'Correlation'])
    results_rrmse = pd.DataFrame(columns=['clean_EEG_file', 'artifact_file', 'artifact', 'temporal_contamination', 'signal_to_noise', 'Algorithm', 'RRMSE'])
    results_snr = pd.DataFrame(columns=['clean_EEG_file', 'artifact_file', 'artifact', 'temporal_contamination', 'signal_to_noise', 'Algorithm', 'SNR'])
    results_time = pd.DataFrame(columns=['clean_EEG_file', 'artifact_file', 'artifact', 'temporal_contamination', 'signal_to_noise', 'Algorithm', 'time'])

    # Load file names
    clean_eeg_files = [f for f in os.listdir(clean_eeg_dir) if f.endswith('.set')]
    artifact_eeg_files_full_paths = []
    artifact_eeg_file_names = []
    artifact_types = []

    for root, _, files in os.walk(artifact_eeg_dir):
        for f in files:
            if f.endswith('.set'):
                artifact_eeg_files_full_paths.append(os.path.join(root, f))
                artifact_eeg_file_names.append(f)
                artifact_types.append(os.path.basename(root)) # Get the folder name as artifact type

    files_in_group1 = len(clean_eeg_files)
    files_in_group2 = len(artifact_eeg_files_full_paths)

    if files_in_group1 == 0 or files_in_group2 == 0:
        print("Error: No .set files found in specified directories. Please check paths and data.")
        exit()

    # Pre-load all clean and artifact raw MNE objects to avoid repeated disk I/O inside loops
    print("Pre-loading clean EEG data...")
    ALLEEG1_raw_objects = []
    for f_name in clean_eeg_files:
        raw = mne.io.read_raw_eeglab(os.path.join(clean_eeg_dir, f_name), preload=True, verbose=False)
        ALLEEG1_raw_objects.append(raw)
    print("Pre-loading artifact EEG data...")
    ALLEEG2_raw_objects = []
    for f_path in artifact_eeg_files_full_paths:
        raw = mne.io.read_raw_eeglab(f_path, preload=True, verbose=False)
        ALLEEG2_raw_objects.append(raw)


    # Main benchmark loop
    print("Starting benchmark...")
    for n_idx, snr_db in enumerate(signal_to_noise_in_db):
        print(f"\nProcessing SNR: {snr_db} dB")
        # Convert SNR from dB (a power ratio) to a linear power ratio.
        # SNR_dB = 10 * log10(P_signal / P_noise) => P_signal / P_noise = 10^(SNR_dB / 10)
        snr_linear_power_ratio = 10**(snr_db / 10)

        # We need a scaling factor for amplitude. Since Power ~ Amplitude^2, Amplitude ~ sqrt(Power).
        # The noise_ratio is the desired ratio of noise amplitude to signal amplitude.
        noise_ratio = 1 / np.sqrt(snr_linear_power_ratio)

        for c_idx, contamination_percent in enumerate(contaminated_signal_proportion):
            print(f"  Processing Contamination: {contamination_percent}%")

            # Generate mixing combinations
            if just_testing:
                mixing_combinations = []
                for _ in range(random_number_of_test_files):
                    mixing_combinations.append((random.randint(0, files_in_group1 - 1),
                                                random.randint(0, files_in_group2 - 1)))
            else:
                mixing_combinations = list(product(range(files_in_group1), range(files_in_group2)))

            # Loop through different artifact types/clean EEG combinations
            for m_idx, (clean_idx, artifact_idx) in enumerate(mixing_combinations):
                clean_eeg_file_name = clean_eeg_files[clean_idx]
                artifact_eeg_file_name = artifact_eeg_file_names[artifact_idx]
                current_artifact_type = artifact_types[artifact_idx]

                print(f"    Processing combination {m_idx+1}/{len(mixing_combinations)}: {clean_eeg_file_name} + {artifact_eeg_file_name}")

                # Get clean EEG data
                clean_raw = ALLEEG1_raw_objects[clean_idx]
                ground_truth_data = clean_raw.get_data()
                srate = clean_raw.info['sfreq']
                ch_names = clean_raw.info['ch_names'] # Get channel names for GEDAI

                # Z-score clean data (MATLAB's zscore(data(:)) z-scores the entire flattened array)
                ground_truth_data_zscored = zscore(ground_truth_data.flatten()).reshape(ground_truth_data.shape)

                # Get artifact data
                artifact_raw = ALLEEG2_raw_objects[artifact_idx]
                artifact_raw_data = artifact_raw.get_data()
                
                # Ensure artifact data has same number of channels as clean data
                if artifact_raw_data.shape[0] != ground_truth_data.shape[0]:
                    n_channels_clean = ground_truth_data.shape[0]
                    n_channels_artifact = artifact_raw_data.shape[0]
                    if n_channels_artifact > n_channels_clean:
                        artifact_raw_data = artifact_raw_data[:n_channels_clean, :]
                    elif n_channels_artifact < n_channels_clean:
                        print(f"Warning: Artifact data '{artifact_eeg_file_name}' has fewer channels ({n_channels_artifact}) than clean data ({n_channels_clean}). Padding with zeros.")
                        padded_artifact = np.zeros((n_channels_clean, artifact_raw_data.shape[1]))
                        padded_artifact[:n_channels_artifact, :] = artifact_raw_data
                        artifact_raw_data = padded_artifact
                
                # Ensure artifact data has same number of samples as clean data
                if artifact_raw_data.shape[1] != ground_truth_data.shape[1]:
                    n_samples_clean = ground_truth_data.shape[1]
                    n_samples_artifact = artifact_raw_data.shape[1]
                    if n_samples_artifact > n_samples_clean:
                        artifact_raw_data = artifact_raw_data[:, :n_samples_clean]
                    elif n_samples_artifact < n_samples_clean:
                        print(f"Warning: Artifact data '{artifact_eeg_file_name}' has fewer samples ({n_samples_artifact}) than clean data ({n_samples_clean}). Padding with zeros.")
                        padded_artifact = np.zeros((artifact_raw_data.shape[0], n_samples_clean))
                        padded_artifact[:, :n_samples_artifact] = artifact_raw_data
                        artifact_raw_data = padded_artifact

                # Apply temporal contamination to artifact data
                min_block_size_samples = 1
                max_block_size_samples = int(1 * srate) # 1 second
                
                contaminated_artifact_data, kept_indices, _, _, _ = \
                    retain_exact_percentage_random_eeg_blocks(
                        artifact_raw_data, contamination_percent,
                        min_block_size_samples, max_block_size_samples
                    )
                
                # Inconsistency Fix: Match MATLAB's artifact scaling logic.
                # Z-score and scale ONLY the non-zero, contaminated portions of the artifact data.
                artifact_data_scaled = np.zeros_like(contaminated_artifact_data)
                if kept_indices.size > 0:
                    # Select only the non-zero parts for z-scoring
                    temp_data = contaminated_artifact_data[:, kept_indices]
                    
                    if np.std(temp_data) > 0:
                        # Z-score the flattened temp_data and reshape back
                        scaled_temp_data = noise_ratio * zscore(temp_data.flatten()).reshape(temp_data.shape)
                        # Place the scaled data back into the corresponding columns
                        artifact_data_scaled[:, kept_indices] = scaled_temp_data
                else:
                    # If there are no kept indices, artifact_data_scaled remains all zeros
                    pass

                # Mix data
                mixed_data = ground_truth_data_zscored + artifact_data_scaled

                # --- RUN GEDAI ---
                start_time = time.time()
                denoised_data_gedai = pygedai_denoise_EEGLAB_data(mixed_data, srate, ch_names)
                end_time = time.time()
                time_gedai = end_time - start_time

                # --- Plotting (if enabled) ---
                if generate_individual_plots:
                    # Create MNE RawArray objects for plotting the mixed and denoised data
                    # The info object needs to be created for each RawArray
                    info_mixed = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=["eeg"] * len(ch_names))
                    raw_mixed = mne.io.RawArray(mixed_data, info_mixed, verbose=False)

                    info_denoised = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=["eeg"] * len(ch_names))
                    raw_denoised = mne.io.RawArray(denoised_data_gedai, info_denoised, verbose=False)

                    plot_title = (f"SNR: {snr_db}dB, Contam: {contamination_percent}%, "
                                  f"Clean: {clean_eeg_file_name}, Artifact: {artifact_eeg_file_name}")
                    print(f"Displaying plot for: {plot_title}")
                    plot_mne_style_overlay_interactive(raw_mixed, raw_denoised, title=plot_title)
                    # plt.close('all') # Uncomment to automatically close plots after interaction

                # Calculate metrics
                gedai_correlation = correlation_coefficient(ground_truth_data_zscored, denoised_data_gedai)
                gedai_rrmse = relative_RMSE(ground_truth_data_zscored, denoised_data_gedai)
                gedai_snr = sig_to_noise(ground_truth_data_zscored, denoised_data_gedai)

                # Store results
                new_row_correlation = {
                    'clean_EEG_file': clean_eeg_file_name,
                    'artifact_file': artifact_eeg_file_name,
                    'artifact': current_artifact_type,
                    'temporal_contamination': str(contamination_percent),
                    'signal_to_noise': str(snr_db),
                    'Algorithm': 'GEDAI',
                    'Correlation': gedai_correlation
                }
                results_correlation = pd.concat([results_correlation, pd.DataFrame([new_row_correlation])], ignore_index=True)

                new_row_rrmse = {
                    'clean_EEG_file': clean_eeg_file_name,
                    'artifact_file': artifact_eeg_file_name,
                    'artifact': current_artifact_type,
                    'temporal_contamination': str(contamination_percent),
                    'signal_to_noise': str(snr_db),
                    'Algorithm': 'GEDAI',
                    'RRMSE': gedai_rrmse
                }
                results_rrmse = pd.concat([results_rrmse, pd.DataFrame([new_row_rrmse])], ignore_index=True)

                new_row_snr = {
                    'clean_EEG_file': clean_eeg_file_name,
                    'artifact_file': artifact_eeg_file_name,
                    'artifact': current_artifact_type,
                    'temporal_contamination': str(contamination_percent),
                    'signal_to_noise': str(snr_db),
                    'Algorithm': 'GEDAI',
                    'SNR': gedai_snr
                }
                results_snr = pd.concat([results_snr, pd.DataFrame([new_row_snr])], ignore_index=True)

                new_row_time = {
                    'clean_EEG_file': clean_eeg_file_name,
                    'artifact_file': artifact_eeg_file_name,
                    'artifact': current_artifact_type,
                    'temporal_contamination': str(contamination_percent),
                    'signal_to_noise': str(snr_db),
                    'Algorithm': 'GEDAI',
                    'time': time_gedai
                }
                results_time = pd.concat([results_time, pd.DataFrame([new_row_time])], ignore_index=True)

    #print("\nBenchmark finished. Results:")
    #print("\nCorrelation Results:")
    #print(results_correlation.head())
    #print("\nRRMSE Results:")
    #print(results_rrmse.head())
    #print("\nSNR Results:")
    #print(results_snr.head())
    #print("\nTime Results:")
    #print(results_time.head())
    print("\nBenchmark finished. Aggregated Results (Mean ± Std Dev):")

    print("\n--- Correlation ---")
    print(results_correlation.groupby(['Algorithm', 'artifact', 'temporal_contamination', 'signal_to_noise'])['Correlation'].agg(['mean', 'std']))

    print("\n--- RRMSE ---")
    print(results_rrmse.groupby(['Algorithm', 'artifact', 'temporal_contamination', 'signal_to_noise'])['RRMSE'].agg(['mean', 'std']))

    print("\n--- SNR ---")
    print(results_snr.groupby(['Algorithm', 'artifact', 'temporal_contamination', 'signal_to_noise'])['SNR'].agg(['mean', 'std']))

    print("\n--- Time (seconds) ---")
    print(results_time.groupby(['Algorithm', 'artifact', 'temporal_contamination', 'signal_to_noise'])['time'].agg(['mean', 'std']))

    # --- Add new grouping by artifact type ---
    print("\n\n--- Aggregated Results by Artifact Type (Mean ± Std Dev) ---")

    print("\n--- Correlation by Artifact ---")
    print(results_correlation.groupby(['Algorithm', 'artifact'])['Correlation'].agg(['mean', 'std']))

    print("\n--- RRMSE by Artifact ---")
    print(results_rrmse.groupby(['Algorithm', 'artifact'])['RRMSE'].agg(['mean', 'std']))

    print("\n--- SNR by Artifact ---")
    print(results_snr.groupby(['Algorithm', 'artifact'])['SNR'].agg(['mean', 'std']))

    print("\n--- Time (seconds) by Artifact ---")
    print(results_time.groupby(['Algorithm', 'artifact'])['time'].agg(['mean', 'std']))

    # Save results to CSV
    results_correlation.to_csv("gedai_benchmark_correlation.csv", index=False)
    results_rrmse.to_csv("gedai_benchmark_rrmse.csv", index=False)
    results_snr.to_csv("gedai_benchmark_snr.csv", index=False)
    results_time.to_csv("gedai_benchmark_time.csv", index=False)