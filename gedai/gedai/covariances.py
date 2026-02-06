import h5py
import numpy as np
import sklearn.metrics


def _compute_distance_cov(raw):
    ch_positions = [raw.info["chs"][i]["loc"][:3] for i in range(raw.info["nchan"])]
    ch_distance_matrix = sklearn.metrics.pairwise_distances(
        ch_positions, metric="euclidean"
    )
    cov = 1 - ch_distance_matrix
    return cov


def _compute_refcov(inst, mat):
    inst_ch_names = inst.info["ch_names"]

    with h5py.File(mat, "r") as f:
        leadfield_data = f["leadfield4GEDAI"]
        # ch_names
        leadfield_channel_data = leadfield_data["electrodes"]
        leadfield_ch_names = [
            f[ref[0]][()].tobytes().decode("utf-16le").lower()
            for ref in leadfield_channel_data["Name"]
        ]
        # leadfield matrix
        leadfield_gain_matrix = leadfield_data["gram_matrix_avref"]
        leadfield_gain_matrix = np.array(leadfield_gain_matrix).T

    # Two-pass matching: exact first, then substring
    ch_indices = []
    ch_names = []
    matched_inst_indices = set()
    match_types = []  # Track match quality for logging
    
    # Pass 1: Exact matching (case-insensitive)
    for inst_idx, inst_ch_name in enumerate(inst_ch_names):
        for leadfield_ch_index, leadfield_ch_name in enumerate(leadfield_ch_names):
            if inst_ch_name.lower() == leadfield_ch_name.lower():
                ch_indices.append(leadfield_ch_index)
                ch_names.append(leadfield_ch_name)
                matched_inst_indices.add(inst_idx)
                match_types.append("exact")
                break  # Move to next inst channel after finding exact match
    
    # Pass 2: Substring matching for unmatched channels
    for inst_idx, inst_ch_name in enumerate(inst_ch_names):
        if inst_idx in matched_inst_indices:
            continue  # Already matched exactly
        
        inst_lower = inst_ch_name.lower()
        best_match = None
        best_match_length = 0
        
        for leadfield_ch_index, leadfield_ch_name in enumerate(leadfield_ch_names):
            leadfield_lower = leadfield_ch_name.lower()
            
            # Check if leadfield name is substring of inst name (e.g., "fp1" in "fp1-ave")
            # or inst name is substring of leadfield name
            if leadfield_lower in inst_lower or inst_lower in leadfield_lower:
                # Prefer longer matches to avoid false positives (e.g., "p1" vs "fp1")
                match_length = min(len(leadfield_lower), len(inst_lower))
                if match_length > best_match_length:
                    best_match = leadfield_ch_index
                    best_match_length = match_length
        
        if best_match is not None:
            ch_indices.append(best_match)
            ch_names.append(leadfield_ch_names[best_match])
            matched_inst_indices.add(inst_idx)
            match_types.append("substring")
    
    # Validation and warnings
    n_inst_channels = len(inst_ch_names)
    n_matched = len(ch_indices)
    
    if n_matched == 0:
        raise ValueError(
            f"No electrode matches found between your data and the leadfield template.\n"
            f"Your channels: {inst_ch_names[:10]}{'...' if n_inst_channels > 10 else ''}\n"
            f"Leadfield channels: {leadfield_ch_names[:10]}{'...' if len(leadfield_ch_names) > 10 else ''}\n"
            f"Please check that your electrode names follow standard conventions (e.g., Fp1, Fp2, F3, F4)."
        )
    
    match_ratio = n_matched / n_inst_channels
    
    # Always warn if any channels didn't match
    if n_matched < n_inst_channels:
        import warnings
        unmatched = [inst_ch_names[i] for i in range(n_inst_channels) if i not in matched_inst_indices]
        n_exact = match_types.count("exact")
        n_substring = match_types.count("substring")
        
        warnings.warn(
            f"Electrode matching: {n_matched}/{n_inst_channels} channels matched "
            f"({n_exact} exact, {n_substring} substring). "
            f"Unmatched channels ({len(unmatched)}): {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}",
            UserWarning
        )
    
    refCOV = leadfield_gain_matrix[np.ix_(ch_indices, ch_indices)]
    return (refCOV, ch_names)
