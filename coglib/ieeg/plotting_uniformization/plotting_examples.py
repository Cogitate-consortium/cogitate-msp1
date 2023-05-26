
import numpy as np
import matplotlib.pyplot as plt
from plotters import plot_time_series, plot_matrix, plot_rasters, plot_brain
import config

# get the parameters dictionary
param = config.param

# =================================================================================
# Plotting temporal generalization matrix:
data = np.random.normal(1, 2, size=[100, 100])
mask = data > 4
t0 = -0.2
tend = 2.0
midpoint = 0  # Center of the color map
plot_matrix(data, t0, tend, t0, tend, mask=mask, cmap=None, ax=None, ylim=None, midpoint=midpoint, transparency=1.0,
            xlabel="Time (s)", ylabel="Time (s)", cbar_label="Accuracy", filename=None, vline=0,
            title=None, square_fig=False)

plt.show()

# =================================================================================
# Plotting time frequency:
data = np.random.normal(1, 2, size=[40, 1000])
mask = np.full((40, 1000), False)
mask[4:14, 400:600] = True
x0 = -0.2
x_end = 2.0
y0 = 0
y_end = 40
midpoint = 0  # Center of the color map
plot_matrix(data, x0, x_end, y0, y_end, mask=mask, cmap=None, ax=None, ylim=None, midpoint=midpoint, transparency=0.5,
            xlabel="Time (s)", ylabel="Frequency (Hz)", cbar_label="Power", filename=None, vline=0,
            title=None, square_fig=False)

plt.show()

# =================================================================================
# Plotting time series:
data = np.random.normal(1, 2, size=[4, 1000])
# Add different intercepts:
data[1, :] = data[1, :] + 10
data[2, :] = data[2, :] + 20
data[3, :] = data[3, :] + 30
error = np.random.normal(1, 0.2, size=[4, 1000])
conditions = ["face", "object", "letter", "false"]
colors = [param["colors"][cate] for cate in conditions]
t0 = -0.2
tend = 2.0

plot_time_series(data, t0, tend, ax=None, err=error, colors=colors, vlines=None, ylim=None,
                 xlabel="Time (s)", ylabel="Activation", err_transparency=0.2,
                 filename=None, title=None, square_fig=False, conditions=conditions, do_legend=True,
                 patches=[0, 1.2], patch_color="r", patch_transparency=0.2)
plt.show()

# =================================================================================
# Plotting raster:
data = np.random.normal(1, 2, size=[40, 1000])
conditions = ["face", "object", "letter", "false"] * 10
cond_order = ["face", "object", "letter", "false"]
t0 = -0.2
tend = 2.0
plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=None, midpoint=1, transparency=1.0,
             xlabel="Time (s)", ylabel="Trials", cbar_label="Amplitude", filename=None, vlines=[0, 0.5, 1.0, 1.5],
             title=None, square_fig=False, conditions=conditions, cond_order=cond_order)
plt.show()


# =================================================================================
# Plotting brain surface:
acc_values = {'G_front_middle': 0.9, 'Pole_temporal': 0.7, 'G_occipital_middle': 0.6}
plot_brain( roi_map=acc_values, subject='fsaverage', surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s', 
                         views=['lateral', 'medial'], 
                         cmap='Oranges', colorbar=True, colorbar_title='ACC', vmin=0.5, vmax=1., overlay_method='overlay',
                         brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300)    
plt.show()


# =================================================================================
# Plotting electrode activations:

# electrode activations must be Nx4 (mni, activation)    
electrode_activations = np.array([[-67.333333, -20.666667, -10.000000, 400], [-71. ,  -8.5,  -9.5, 300]])  
    
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


#%%
# this is commented out, but is for ECoG using CH2 template brain
# Using CH2 (must be in your freesurfer subjects_dir). 
# Is this case you need to set scanner2tkr (or set to None for auto-detection) since this is brain is offset       
plot_brain(subject='CH2', surface='pial', hemi='lh', sulc_map='curv', parc='aparc.a2009s', 
                          views=['lateral', 'medial'], scanner2tkr=(1.0, 17., -19.0),
                          cmap='Oranges', colorbar=True, colorbar_title='Latency (ms)', colorbar_title_position='left', vmin=0.,
                          electrode_activations=electrode_activations, vertex_distance=10., vertex_scaling='linear', 
                          brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300)    
plt.show()












