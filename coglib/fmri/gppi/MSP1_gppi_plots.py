# Plots gppi group-level corrected and uncorrected stats maps on a brain surfaces
# Author: Aya Khalaf
# Email: aya.khalaf@yale.edu

from plotters import plot_brain
import os

# Separate conditions
contrast = ['face_rel-object_rel']
contrast = ['face_irr-object_irr']
# Combined conditions
#contrasts = ['face-object']
# Select seed size in voxels
n_voxels = 300
# Select seed (FFA or LOC)
# LOC
roi_folder = 'PPI_LOC_gPPI_' + str(n_voxels)
# FFA
#roi_folder = 'PPI_FFA_gPPI_' + str(n_voxels)

data_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/second level_nonparametric', roi_folder, contrast)
roi_map = {'V1d': 0, 'V1v': 0, 'V2d': 0, 'V2v': 0}

# Corrected maps
plot_brain(surface='inflated', hemi='lh', overlays=[os.path.join(data_dir, 'gppi_group_stats_map.nii')], colorbar_title='t-value',
           overlay_threshold=0, vmin=0, vmax=5, cmap='Oranges', views=['medial', 'lateral'], roi_map=roi_map, parc='wang2015_mplbl',
           roi_map_edge_color=[0, 0, 0], roi_map_transparency=0.0, overlay_method='blend', save_file=os.path.join(data_dir,'gppi_lh.png'))
#plt.show()
plot_brain(surface='inflated', hemi='rh', overlays=[os.path.join(data_dir, 'gppi_group_stats_map.nii')], colorbar_title='t-value',
           overlay_threshold=0, vmin=0, vmax=5, cmap='Oranges', views=['medial', 'lateral'], roi_map=roi_map, parc='wang2015_mplbl',
           roi_map_edge_color=[0, 0, 0], roi_map_transparency=0.0, overlay_method='blend', save_file=os.path.join(data_dir,'gppi_rh.png'))

# Uncorrected maps
plot_brain(surface='inflated', hemi='lh', overlays=[os.path.join(data_dir, 'gppi_group_stats_map_uncorr_0.01.nii')], colorbar_title='t-value',
           overlay_threshold=0, vmin=-3, vmax=3, cmap='RdYlBu_r', views=['medial', 'lateral'], roi_map=roi_map, parc='wang2015_mplbl',
           roi_map_edge_color=[0, 0, 0], roi_map_transparency=0.0, overlay_method='blend', save_file=os.path.join(data_dir,'gppi_lh_0.01.png'))
#plt.show()
plot_brain(surface='inflated', hemi='rh', overlays=[os.path.join(data_dir, 'gppi_group_stats_map_uncorr_0.01.nii')], colorbar_title='t-value',
           overlay_threshold=0, vmin=-3, vmax=3, cmap='RdYlBu_r', views=['medial', 'lateral'], roi_map=roi_map, parc='wang2015_mplbl',
           roi_map_edge_color=[0, 0, 0], roi_map_transparency=0.0, overlay_method='blend', save_file=os.path.join(data_dir,'gppi_rh_0.01.png'))



