


mni =[-0.0318   -0.0970   -0.0072;
   -0.0320   -0.0973    0.0093];

activations = [1.8; 2];


%% basic brain, update lighting as you rotate (not sure how well it will work on the hpc)
figure;
plotBrain(hemi={'lh', 'rh'}, update_lighting=true);

%% plot some electrodes
figure;
plotBrain(hemi={'lh', 'rh'}, view=[-90 0], coords=m*1e3);

%% plot some electrodes using spheres  (default = 'scatter')
figure;
plotBrain(hemi={'lh', 'rh'}, view=[-90 0], coords=m*1e3, elec_type='sphere', elec_color=[1 0 0]);


%% plot some electrodes using big red spheres
figure;
plotBrain(hemi={'lh', 'rh'}, view=[-90 0], coords=m*1e3, elec_type='sphere', elec_color=[1 0 0], elec_size=5);

%% show activations
figure;
plotBrain(hemi={'lh', 'rh'}, view=[-90 0], coords=m*1e3, activations=activations, color_map='RdYlBu_r', color_bar=true, colorbar_title='latency (ms)', activation_method=@mean, activation_show_electrodes=true);


%% show activations (inflated)
figure;
plotBrain(hemi='lh', view=[-90 0], coords=m*1e3, surface='inflated', activations=activations, color_map='RdYlBu_r', color_bar=true, colorbar_title='latency (ms)', activation_method=@mean, activation_show_electrodes=true);

%% show electrodes on top
figure;
plotBrain(hemi={'lh', 'rh'}, view=[-90 0], coords=m*1e3, activations=activations, color_map='RdYlBu_r', color_bar=true, colorbar_title='latency (ms)', activation_method=@median, activation_show_electrodes=true);