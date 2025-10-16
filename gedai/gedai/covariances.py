import h5py
import numpy as np
import sklearn.metrics


def compute_distance_cov(raw):
    ch_positions = [raw.info['chs'][i]['loc'][:3] for i in range(raw.info['nchan'])]
    ch_distance_matrix = sklearn.metrics.pairwise_distances(ch_positions, metric='euclidean')
    cov = 1 - ch_distance_matrix
    return(cov)


def compute_refcov(inst, mat):
    inst_ch_names = inst.info['ch_names']

    with h5py.File(mat, 'r') as f:
        leadfield_data = f['leadfield4GEDAI']
        # ch_names
        leadfield_channel_data = leadfield_data['electrodes']
        leadfield_ch_names = [f[ref[0]][()].tobytes().decode('utf-16le').lower() for ref in leadfield_channel_data['Name']]
        # leadfield matrix
        leadfield_gain_matrix = leadfield_data['gram_matrix_avref']
        leadfield_gain_matrix = np.array(leadfield_gain_matrix).T

    ch_indices = []
    ch_names = []
    for i, inst_ch_name in enumerate(inst_ch_names):
        for l, leadfield_ch_name in enumerate(leadfield_ch_names):
            if inst_ch_name.lower() == leadfield_ch_name.lower():
                ch_indices.append(l)
                ch_names.append(leadfield_ch_name)

    refCOV = leadfield_gain_matrix[np.ix_(ch_indices, ch_indices)]
    return(refCOV, ch_names)
