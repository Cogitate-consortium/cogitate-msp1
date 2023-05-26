
import numpy as np
import matplotlib.pyplot as plt
from ecog_plotters import  plot_brain, plot_electrodes
import config
import pandas as pd

# get the parameters dictionary
param = config.param

#%%

# =================================================================================
# Plotting electrode activations:

# electrode activations must be Nx4 (mni, activation)    
electrode_activations = np.array([[-67.333333, -20.666667, -10.000000, 400], [-71. ,  -8.5,  -9.5, 200]])  
    
# Using fsaverage      
plot_brain(subject='fsaverage', surface='pial', hemi='lh', sulc_map='curv', parc='aparc.a2009s', 
                         views=['lateral', 'medial'], 
                         cmap='Oranges', colorbar=True, colorbar_title='Latency (ms)', colorbar_title_position='left', vmin=0.,
                         electrode_activations=electrode_activations, vertex_distance=10., vertex_scaling='linear', 
                         brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300)    
plt.show()


# Using inflated surface (preserves suface mapping by matching vertices across pial/inflated surfaces)      
plot_brain(subject='fsaverage', surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s', 
                         electrode_activations=electrode_activations, vertex_distance=10., vertex_scaling='linear', 
                         views=['lateral', 'medial'], 
                         cmap='Oranges', colorbar=True, colorbar_title='Latency (ms)', colorbar_title_position='left', vmin=0.,
                         brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300)    
plt.show()


# #%% USING CH2 this is commented out, but is for ECoG using CH2 template brain
# # Using CH2 (must be in your freesurfer subjects_dir). 
# # Is this case you need to set scanner2tkr (or set to None for auto-detection) since this is brain is offset       
# plot_brain(subject='CH2', surface='pial', hemi='lh', sulc_map='curv', parc='aparc.a2009s', 
#                           views=['lateral', 'medial'], scanner2tkr=(1.0, 17., -19.0),
#                           cmap='Oranges', colorbar=True, colorbar_title='Latency (ms)', colorbar_title_position='left', vmin=0.,
#                           electrode_activations=electrode_activations, vertex_distance=10., vertex_scaling='linear', 
#                           brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300)    
# plt.show()




#%% electrode plotting
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
elec_file = bids_root+'/sub-SF119/ses-V1/ieeg/sub-SF119_ses-V1_space-fsaverage_electrodes.tsv'
channels_coordinates = pd.read_csv(elec_file, sep='\t')
mni = np.asarray([channels_coordinates.x.values, channels_coordinates.y.values, channels_coordinates.z.values]).T * 1e3
elec_color = np.tile( [0., 0., 0., 1.], np.size(mni,0) ).reshape(np.size(mni,0),4)



#%% plot electrodes
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial')


#%% plot using a reduced cortical model. This speeds up plotting, and allows for rotating the figure more easily. However, it breaks any activation/roi_mapping and cannot be used with this option (see last demo as an example of using roi_maps on brain + electrodes)
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', reduce_model=5000, force_to_nearest_vertex=True)

#%% remove edge_color
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', reduce_model=5000, force_to_nearest_vertex=True, edge_color=None)

#%% the medial view is wonky at this point, so we will need to address this is needed [TODO: address computed_zorder=False bug]
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', reduce_model=5000, force_to_nearest_vertex=True, views=['medial'])


#%% change the elctrode size & markers
elec_color = np.tile( np.array( ([0., 0., 0., 1.],[1., 0., 0., 1.])), ( int(np.size(mni,0)/2),1) )
markers = np.tile( ['o', 'v'], int(np.size(mni,0)/2) )
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', elec_size=20, elec_marker=markers)


#%% force to nearest vertex
plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', force_to_nearest_vertex=True)


# using CH2 (needed in default freesurfer directory)
# #%% CH2
# plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', force_to_nearest_vertex=True, subject='CH2', scanner2tkr=None) #forces calculation of scanner2tkr

# #%% CH2 w/additional brain alpha
# plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', force_to_nearest_vertex=True, subject='CH2', scanner2tkr=None, brain_alpha=0.5) #forces calculation of scanner2tkr

# #%% CH2 w/additional brain alpha + additional views
# plot_electrodes( coords=mni, elec_color=elec_color, surface='pial', force_to_nearest_vertex=True, subject='CH2', scanner2tkr=None, brain_alpha=0.9, views=['lateral', (0,210,0)])



#%% use an inflated surface + roi_map to highlight ROI
# using an inflated surface automatically snaps electrodes to the close vertex (since vertices are shared across pial/inflated surfaces) ensuring accurate display (notice how all surface electrodes end up on a gryus)
# create an roi_map of IIT rois, with color value set to 1
# using overlay_method='blend' helps to make rois more transparent to see underlying brain surface
iit_rois = ['G_temporal_inf',
 'Pole_temporal',
 'G_cuneus',
 'G_occipital_sup',
 'G_oc-temp_med-Lingual',
 'Pole_occipital',
 'S_calcarine',
 'G_and_S_occipital_inf',
 'G_occipital_middle',
 'G_oc-temp_lat-fusifor',
 'G_oc-temp_med-Parahip',
 'S_intrapariet_and_P_trans',
 'S_oc_middle_and_Lunatus',
 'S_oc_sup_and_transversal',
 'S_temporal_sup']
roi_map = {k: .5 for k in iit_rois}

plot_electrodes( coords=mni, elec_color=elec_color, force_to_nearest_vertex=True, surface='inflated', views=['lateral'], brain_alpha=0.5, overlay_method='blend',
                roi_map=roi_map, cmap='Blues', vmin=0, vmax=1)
















