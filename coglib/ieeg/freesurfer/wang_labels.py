

def get_montage_volume_labels_wang(montage, subject, subjects_dir=None,
                                   aseg='wang15_mplbl', dist=2):
    """Get regions of interest near channels from a Freesurfer parcellation.
    .. note:: This is applicable for channels inside the brain
              (intracranial electrodes).
    Parameters
    ----------
    %(montage)s
    %(subject)s
    %(subjects_dir)s
    %(aseg)s
    dist : float
        The distance in mm to use for identifying regions of interest.
    Returns
    -------
    labels : dict
        The regions of interest labels within ``dist`` of each channel.
    colors : dict
        The Freesurfer lookup table colors for the labels.
    """
    import numpy as np
    import os.path as op
    from mne.channels import DigMontage
    from mne._freesurfer import read_freesurfer_lut
    from mne.utils import get_subjects_dir, _check_fname, _validate_type
    from mne.transforms import apply_trans
    from mne.surface import _voxel_neighbors, _VOXELS_MAX
    from collections import OrderedDict

    _validate_type(montage, DigMontage, 'montage')
    _validate_type(dist, (int, float), 'dist')

    if dist < 0 or dist > 10:
        raise ValueError('`dist` must be between 0 and 10')

    import nibabel as nib
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    aseg = _check_fname(op.join(subjects_dir, subject, 'mri', aseg + '.mgz'),
                        overwrite='read', must_exist=True)
    aseg = nib.load(aseg)
    aseg_data = np.array(aseg.dataobj)

    # read freesurfer lookup table
    lut, fs_colors = read_freesurfer_lut(
        op.join(subjects_dir, 'wang2015_LUT.txt'))  # put the wang2015_LUT.txt into the freesurfer subjects dir
    label_lut = {v: k for k, v in lut.items()}

    # assert that all the values in the aseg are in the labels
    assert all([idx in label_lut for idx in np.unique(aseg_data)])

    # get transform to surface RAS for distance units instead of voxels
    vox2ras_tkr = aseg.header.get_vox2ras_tkr()

    ch_dict = montage.get_positions()
    if ch_dict['coord_frame'] != 'mri':
        raise RuntimeError('Coordinate frame not supported, expected '
                           '"mri", got ' + str(ch_dict['coord_frame']))
    ch_coords = np.array(list(ch_dict['ch_pos'].values()))

    # convert to freesurfer voxel space
    ch_coords = apply_trans(
        np.linalg.inv(aseg.header.get_vox2ras_tkr()), ch_coords * 1000)
    labels = OrderedDict()
    for ch_name, ch_coord in zip(montage.ch_names, ch_coords):
        if np.isnan(ch_coord).any():
            labels[ch_name] = list()
        else:
            voxels = _voxel_neighbors(
                ch_coord, aseg_data, dist=dist, vox2ras_tkr=vox2ras_tkr,
                voxels_max=_VOXELS_MAX)
            label_idxs = set([aseg_data[tuple(voxel)].astype(int)
                              for voxel in voxels])
            labels[ch_name] = [label_lut[idx] for idx in label_idxs]

    all_labels = set([label for val in labels.values() for label in val])
    colors = {label: tuple(fs_colors[label][:3] / 255) + (1.,)
              for label in all_labels}
    return labels, colors