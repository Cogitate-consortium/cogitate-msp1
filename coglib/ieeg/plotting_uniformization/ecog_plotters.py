#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:44:39 2023

@author: simonhenin
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config
# for plot_brain
from mne import read_surface, decimate_surface
from mne.utils import get_subjects_dir, run_subprocess
from mne.label import _read_annot
from mne.surface import read_curvature
from mne.transforms import apply_trans
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib import cm, colors
from matplotlib.collections import PolyCollection
from scipy import spatial

# get the parameters dictionary
param = config.param

# # Loading the json dict:
# with open("/hpc/users/alexander.lepauvre/sw/github/plotting_uniformization/config.json", 'r') as fp:
#     param = json.load(fp)

fig_size = param["figure_size_mm"]
def_cmap = param["colors"]["cmap"]
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('font', size=param["font_size"])  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"])  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"])  # legend fontsize
plt.rc('figure', titlesize=param["font_size"])  # fontsize of the fi


# credit where credit is due:
# This script heavily adapted from https://github.com/greydongilmore/seeg2bids-pipeline/blob/9055da475411032a49685ab6552253843b63b042/workflow/scripts/brain4views.py

# transformations from 3D -> 2D using PolyCollection
#   see also: https://matplotlib.org/matplotblog/posts/custom-3d-engine/


def mm2inch(val):
    return val / 25.4

EPSILON = 1e-12  # small value to add to avoid division with zero
def normalize_v3(arr):
	"""Normalize a numpy array of 3 component vectors shape=(n,3)"""
	lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
	arr[:, 0] /= lens + EPSILON
	arr[:, 1] /= lens + EPSILON
	arr[:, 2] /= lens + EPSILON
	return arr


def normal_vectors(vertices, faces):
	tris = vertices[faces]
	n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
	n = normalize_v3(n)
	return n


def frustum(left, right, bottom, top, znear, zfar):
	M = np.zeros((4, 4), dtype=np.float32)
	M[0, 0] = +2.0 * znear / (right - left)
	M[1, 1] = +2.0 * znear / (top - bottom)
	M[2, 2] = -(zfar + znear) / (zfar - znear)
	M[0, 2] = (right + left) / (right - left)
	M[2, 1] = (top + bottom) / (top - bottom)
	M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
	M[3, 2] = -1.0
	return M


def perspective(fovy, aspect, znear, zfar):
	h = np.tan(0.5 * np.radians(fovy)) * znear
	w = h * aspect
	return frustum(-w, w, -h, h, znear, zfar)


def translate(x, y, z):
	return np.array(
		[[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float
	)


def xrotate(theta):
	t = np.pi * theta / 180
	c, s = np.cos(t), np.sin(t)
	return np.array(
		[[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
	)


def yrotate(theta):
	t = np.pi * theta / 180
	c, s = np.cos(t), np.sin(t)
	return np.array(
		[[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float
	)

def zrotate(theta):
	t = np.pi * theta / 180
	c, s = np.cos(t), np.sin(t)
	return np.array(
		[[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
	)

def normalize(X):
    return X/(1e-16+np.sqrt((np.array(X)**2).sum(axis=-1)))[..., np.newaxis]

def clip(V, vmin=0, vmax=1):
    return np.minimum(np.maximum(V,vmin),vmax)

def shading_intensity(vertices,faces, light = np.array([0,0,1]),shading=0.7):
    """shade calculation based on light source
       default is vertical light.
       shading controls amount of shading.
       Also saturates so top 20 % of vertices all have max intensity."""
    face_normals=normal_vectors(vertices,faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)]=1
    shading = 0.7    
    #top 20% all become fully coloured
    intensity = (1-shading)+shading*(intensity-np.min(intensity))/((np.percentile(intensity,80)-np.min(intensity)))
    #saturate
    intensity[intensity>1]=1
    
    return intensity

def lighting(F, direction=(1,1,1), color=(1,0,0), specular=False):
    """
    """
    
    # Faces center
    C = F.mean(axis=1)
    # Faces normal
    N = normalize(np.cross(F[:,2]-F[:,0], F[:,1]-F[:,0]))
    # Relative light direction
    D = normalize(C - direction)
    # Diffuse term
    diffuse = clip((N*D).sum(-1).reshape(-1,1))

    # Specular term
    if specular:
        specular = np.power(diffuse,24)
        return np.maximum(diffuse*color, specular)
    
    return diffuse*color


def plot_brain(subject='fsaverage', subjects_dir=None, surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s', roi_map=None, views=['lateral', 'medial'], 
                         cmap='Oranges', colorbar=True, colorbar_title='ACC', colorbar_title_position='top', vmin=None, vmax=None, overlay_method='overlay',
                         electrode_activations=None, vertex_distance=10., vertex_scaling='linear', vertex_scale = (1.0, 0.), vertex_method='max', 
                         scanner2tkr=(0,0,0), brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300):
    # plot brain surfaces and roi map overlays 
    #
    # options:
    #   roi_map: dict of rois and associated values (e.g. ACC), example: {'G_front_middle': 0.98, 'Pole_temporal': 0.54, 'G_cuneus': 0.75}
    #   subject: freesurfer subject [default: fsaverage]
    #   subjects_dir: freesurfer subjects directory [default: none, find default freesurfer directory]
    #   surface: surface type to use, e.g., inflated, pial [default: 'inflated' ]
    #   hemi: hemisphere to use [default='lh']
    #   sulc_map: sulcus mapping to use, default is to use the curvature file [default: lh.curv]
    #   parc: parcellation file to use for roi_map. It should include the names of the roi_map keys [default: 'aparc.a2009s']
    #   views: views to show as strings [ 'lateral', 'medial'] or list of tuples as (x,y [,z]), x=rotation around horizontal plane, y = rotation on vertical plane, z = rotation on anterior/posterior plane
    #   colorbar: Add colorbar [default: True]
    #   cmap: colormap to use for roi_map [default: 'Oranges']
    #   overlay_method: 'overlay' or 'blend' colors onto the brain [default: overlay]
    #
    #   electrode_activations:  Nx4 numpy array of electrodes coordinates (columsn 1-3) and activations (4th column)
    #   vertex_distance:        distance (in mm) of activation extent
    #   vertex_scaling:         method of activation decrease with distance (linear, gaussian, or exponential decay) [default: linear]
    #   vertex_scale:           tuple or list of float shape (scale_at_center, scale_at_distance). Extent of scaling [default: (1., 0.)]
    #   vertex_method:          Method for combining vertices [default: max]
    #    
    #   brain_cmap: colormap used for brain [default: 'Greys']
    #   brain_color_scale: range of cmap values to use [default: (0.42, 0.58) produces narrow grey range used for standrd inflated surfaces]
    #   brain_alpha: transparency of the surface
    #   figsize: sigure size
    #   save_file: filename to save figure to
    #   dpi: dpi to save with
    
    # TODO: need to update lighting for non-standard views, (e.g., ventral, dorsal)
    
    
    # some checks
    assert type(figsize) is tuple and len(figsize)==2, 'figsize must be tuple of length 2'
    if roi_map is not None:
        assert (parc is not None), 'Parcelelation file is required to map rois'
    if subjects_dir is None:
        subjects_dir = get_subjects_dir(raise_error=True)
    
    if not os.path.exists(surface):
        # generate from freesurfer folder
        surface = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+surface)
    vertices, faces = read_surface(surface)
    if scanner2tkr is None:
        # get from original mri
        mri_orig = os.path.join( subjects_dir, subject, 'mri', 'orig.mgz')
        scanner2tkr,_ = run_subprocess(['mri_info', '--scanner2tkr', mri_orig])
        scanner2tkr = np.fromstring(scanner2tkr, sep=' ').astype(float)
        scanner2tkr = scanner2tkr.reshape(4,4)
        scanner2tkr = scanner2tkr[0:3,-1]
    vertices -= scanner2tkr
    
    # read in annotation file    
    if parc is not None:
        if not os.path.exists(parc):
            parc = os.path.join( subjects_dir, subject, 'label', hemi + os.path.extsep + parc + '.annot' )
        assert os.path.exists(parc), ('could not locate parcellation file at %s' % parc)            
        annot_vert, annot_tbl, annot_anatomy  = _read_annot( parc )
    

    # parse views
    view_strings = {'lh':{'lateral': (-90, 0, 0), 'medial': (90,0, 0), 'rostral': (0,0,0), 'caudal': (180,0,0), 'dorsal': (180,0,90), 'ventral': (0,0,90)},
                    'rh':{'lateral': (90, 0, 0), 'medial': (-90,0, 0), 'rostral': (0,0,0), 'caudal': (180,0,0), 'dorsal': (180,0,90), 'ventral': (0,0,90)},
                    }
    v = []
    for vv in views:
        if type(vv) == str:
            v.append( view_strings[hemi][vv] )
        else:
            v.append( vv+(0,) ) # append tuple(0) to ensure at least 3 dims
    views = v
                
            
    ##################################            
    # eletrode activations. 
    # Needs to be done before any scaling of vertices since distances are computed
    vertex_activations = []
    if electrode_activations is not None:
        
        # Needs to be done on the pial surface since electrode distance to pial surface is required to map activation. 
        # But after that it can be applied to the inflated since the vertex indices are the same
        if os.path.splitext(surface)[1] == '.pial':
            vertices_pial = vertices
        else:
            surface = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+'pial')
            vertices_pial, _ = read_surface(surface)
            vertices_pial -= scanner2tkr
        
        # find closest vertex
        for i in range( electrode_activations.shape[0] ):
            coords = electrode_activations[i, 0:3]
            val = electrode_activations[i, 3]
            tmp = np.sqrt((vertices_pial[:,0]-coords[0])**2 + (vertices_pial[:,1]-coords[1])**2 + (vertices_pial[:,2]-coords[2])**2)
            vertex_id = np.argmin( tmp )
            tmp = np.sqrt((vertices_pial[:,0]-vertices_pial[vertex_id, 0])**2 + (vertices_pial[:,1]-vertices_pial[vertex_id, 1])**2 + (vertices_pial[:,2]-vertices_pial[vertex_id, 2])**2)
            vertex_nbrs = np.where( tmp < vertex_distance)[0]
            vertex_nbrs = vertex_nbrs[np.argsort( tmp[vertex_nbrs] )]
            
            # update this to be based on distance from center vertex
            if vertex_scaling == 'guassian':
                mx = np.max( tmp[vertex_nbrs] )
                sigma = 0.8/2
                vertex_vals = np.exp(-( ( (tmp[vertex_nbrs]/mx)**2 )/(2*sigma**2)));
            else: #linear
                vertex_vals = val*(1 - tmp[vertex_nbrs]/vertex_distance)
            vertex_activations.append( [vertex_nbrs, vertex_vals] )
        
        
        
    vert_range = max(vertices.max(0) - vertices.min(0))
    vertices = (vertices - (vertices.max(0) + vertices.min(0)) / 2) / vert_range
    
    face_normals = normal_vectors(vertices, faces)
    light = np.array([0, 0, 1])
    intensity = np.dot(face_normals, light)
    shading = 0.7  # shading 0-1. 0=none. 1=full
    # top 20% all become fully colored
    denom = np.percentile(intensity, 80) - np.min(intensity)
    intensity = (1 - shading) + shading * (intensity - np.min(intensity)) / denom
    intensity[intensity > 1] = 1

    # instantiate face colors
    face_colors = np.ones((faces.shape[0], 4))
    
    # generate sulcal map definition
    sulc = np.ones(vertices.shape[0]) * 0.5
    if sulc_map is not None:
        # read in sulcal file (typically curv file) 
        if os.path.exists( sulc_map ):
            # full path file
            sulc_file = sulc_map
        else:
            sulc_file = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+sulc_map )
        assert os.path.exists(sulc_file), ('could not locate sulc_map file at %s' % sulc_file)            
        sulc = read_curvature(sulc_file, binary=False)
    sulc_faces = np.mean(sulc[faces], axis=1)
    
    # binarize sulcal map
    if sulc_faces.min() != sulc_faces.max():
        neg_sulc = np.where(sulc_faces <= 0)
        pos_sulc = np.where(sulc_faces > 0)
        sulc_faces[neg_sulc] = 0
        sulc_faces[pos_sulc] = 1

    # mask, unused for now
    # mask = np.zeros(vertices.shape[0]).astype(bool)

    # assign greyscale colormap to sulcal map faces
    greys = plt.get_cmap(brain_cmap, 512)
    greys_narrow = ListedColormap(greys(np.linspace(brain_color_scale[0], brain_color_scale[1], 256)))
    face_colors = greys_narrow(sulc_faces)

    
    
     
    ##################################
    # roi labeling/coloring    
    label_masks  = []
    label_val = []
    if roi_map is not None:
        
        vals = [val for val in roi_map.values()]
        # figure out vmin/vmax from label_val, if vmin/vmax not provided
        if vmin is None:
            vmin = np.min(vals)
        if vmax is None:    
            vmax = np.max(vals)
            
        # colors for roi map based on cmap provided
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # reindex the colormap so it starts deeper into the colorscale (e.g. color values = 0.1-1 over the scale of vmin-vmax (ACC) of purple)
        cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))
        
        # loop through the dictionary and color rois
        for i, roi in enumerate( roi_map ):
            L_mask = np.zeros(vertices.shape[0]).astype(bool)
            idx_ = annot_anatomy.index( roi.encode() )
            vertex_val = annot_tbl[idx_, -1]
            vertex_idx = np.where( annot_vert == vertex_val )[0]
            L_mask[vertex_idx] = 1  # label vertices = 1
            label_masks.append(L_mask)
            label_val.append( roi_map[roi] )

        
        label_mask_faces = [np.median(L[faces], axis=1) for L in label_masks]
        # assign label faces to appropriate color
        for i, L in enumerate(label_mask_faces):
            L_idx = np.where(L >= 0.5)
            # blend (multiply) label color with underlying color
            # face_colors[L_idx] = face_colors[L_idx] * [1., 0., 0., 1.]
            if overlay_method == 'blend':
                face_colors[L_idx] = face_colors[L_idx] * cmap(norm(label_val[i]))
            else:
                face_colors[L_idx] = cmap(norm(label_val[i]))
    
    
    
    ##################################
    # vertex coloring for activations
    if len(vertex_activations):
        
        # figure out vmin/vmax from activation values
        if vmin is None:
            vmin = np.min( electrode_activations[:,3].min() )
        if vmax is None:    
            vmax = np.max( electrode_activations[:,3].max() )
            
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))
        # compile all vertex indices into a dict and then get mean value per index
        indices = []
        for vertex_nbrs, vertex_vals in vertex_activations:
            indices.append( vertex_nbrs )
        indices = np.concatenate( indices )
        vert_indices = {k: [] for k in indices}
            
        # now add values to them
        for vertex_nbrs, vertex_vals in vertex_activations:
            for i, idx in enumerate(vertex_nbrs):
                vert_indices[ idx ].append( vertex_vals[i] )
                    
        # finally add the face color based on these indices
        for idx in vert_indices.keys():
            idx_ = np.where(faces == idx)[0]
            if vertex_method == 'max':
                val = np.max( vert_indices[idx] )
            elif vertex_method == 'mean':
                val = np.mean( vert_indices[idx] )
            elif vertex_method == 'median':
                val = np.median( vert_indices[idx] )
                    
            if overlay_method == 'blend':
                face_colors[idx_] = face_colors[idx_] * cmap(norm( val ))
            else:
                face_colors[idx_] = cmap(norm( val ))
            
            
            
    # apply shading after the facecolors are set
    # face_colors[:, 0] *= intensity
    # face_colors[:, 1] *= intensity
    # face_colors[:, 2] *= intensity
    
    ##################################
    # Draw the plot
    fig = plt.figure(figsize=figsize)
    for i, view in enumerate(views):
        MVP = (
        	perspective(25, 1, 1, 100)
        	@ translate(0, 0, -3-(view[2]>0)*1) # adjust if z-axis roated TODO: make more dynamic
            # the coordinate system is not like in mni space (e.g. x = left-to-right, y=anterior-to-posterior, z = ventral-to-dorsal)
            # first 2 rotaions, put in back in this type of coordinate systems. The view=[rotation_around_horizontal_pane   rotation_around_vertical_plane]
            @ xrotate(-90)  
            @ zrotate(180)
            @ zrotate( view[0] ) # around horizontal
            @ yrotate( view[1] ) # around vertical
            @ xrotate( view[2] ) # around other vertical
        )
        # adapt lighting
        light = np.array([0,0,1,1]) @ yrotate( -view[2] )
        intensity=shading_intensity(vertices, faces, light=light[:3],shading=0.7)
        fcolor = face_colors.copy()
        fcolor[:,0] *= intensity
        fcolor[:,1] *= intensity
        fcolor[:,2] *= intensity
        
        
        # translate coordinates based on viewing position
        V = np.c_[vertices, np.ones(len(vertices))] @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        V = V[faces]
        
        
        # triangle coordinates
        T = V[:, :, :2]
        # get Z values for ordering triangle plotting
        Z = -V[:, :, 2].mean(axis=1)
        # sort the triangles based on their z coordinate
        Zorder = np.argsort(Z)
        T, C = T[Zorder, :], fcolor[Zorder, :]
        
        # add subplot and plot PolyCollection
        # ax = fig.add_subplot(
        #  	1,
        #  	2,
        #  	i + 1,
        #  	xlim=[-1, +1],
        #  	ylim=[-0.6, +0.6],
        #  	frameon=False,
        #  	aspect=1,
        #  	xticks=[],
        #  	yticks=[],
        # )
        ax = fig.add_axes(
            (0.025+(i)*0.45, 0, 0.5, 1), 
            xlim=[-1, +1],
         	ylim=[-0.6, +0.6],
         	frameon=False,
         	aspect=1,
         	xticks=[],
         	yticks=[],
        )
        collection = PolyCollection(
        	T, closed=True, antialiased=False, facecolor=C, edgecolor=C, linewidth=0
        )
        collection.set_alpha(brain_alpha)
        ax.add_collection(collection)
    plt.subplots_adjust(wspace=0)
            
    threshold = None
    if colorbar:
        
        our_cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        bounds = np.linspace(vmin, vmax, our_cmap.N)
        
        if threshold is None:
            ticks = [vmin, vmax]
        elif threshold == vmin:
            ticks = [vmin, vmax]
        else:
            if vmin >= 0:
                ticks = [vmin, threshold, vmax]
            else:
                ticks = [vmin, -threshold, threshold, vmax]

            cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
			# set colors to grey for absolute values < threshold
            istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
            istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
            for i in range(istart, istop):
                cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
            our_cmap = LinearSegmentedColormap.from_list(
				"Custom cmap", cmaplist, our_cmap.N
			)

		# we need to create a proxy mappable
        proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
        # proxy_mappable.set_array(overlay_faces)
        proxy_mappable.set_array(face_colors)
        cax = plt.axes([0.49, 0.55, 0.02, 0.1])
        cb = plt.colorbar(
			proxy_mappable,
			cax=cax,
			boundaries=bounds,
			ticks=ticks,
            drawedges=False,
			orientation="vertical",
		)
        cb.ax.tick_params(size=0, labelsize=8)
        if colorbar_title_position == 'top':
            cb.ax.set_title(colorbar_title, fontsize=8)
        elif colorbar_title_position == 'right':
                cb.ax.set_ylabel(colorbar_title, fontsize=8, rotation=270)    
        elif colorbar_title_position == 'left':
            cb.ax.set_ylabel(colorbar_title, fontsize=8)
            cb.ax.yaxis.set_label_position('left')
        
    if save_file is not None:
        plt.savefig(save_file, dpi=dpi)
    
    return fig








# electrode plots
def plot_electrodes(subject='fsaverage', subjects_dir=None, surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s', roi_map=None, views=['lateral'], 
                         cmap='Oranges', colorbar=True, colorbar_title='ACC', colorbar_title_position='top', vmin=0., vmax=1., overlay_method='overlay',
                         electrode_activations=None, vertex_distance=10., vertex_scaling='linear', vertex_scale = (1.0, 0.), vertex_method='max', 
                         coords=None, elec_color=[0., 0., 0.], elec_size=12, elec_marker='o', edge_color='k', force_to_nearest_vertex=False, reduce_model=None,
                         scanner2tkr=(0,0,0), brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6), save_file=None, dpi=300):
    

# plot electrodes on brain surfaces with roi map overlays using matplotlib axes3d
# options:
#   roi_map: dict of rois and associated values (e.g. ACC), example: {'G_front_middle': 0.98, 'Pole_temporal': 0.54, 'G_cuneus': 0.75}
#   subject: freesurfer subject [default: fsaverage]
#   subjects_dir: freesurfer subjects directory [default: none, find default freesurfer directory]
#   surface: surface type to use, e.g., inflated, pial [default: 'inflated' ]
#   hemi: hemisphere to use [default='lh']
#   sulc_map: sulcus mapping to use, default is to use the curvature file [default: lh.curv]
#   parc: parcellation file to use for roi_map. It should include the names of the roi_map keys [default: 'aparc.a2009s']
#   views: views to show as strings [ 'lateral', 'medial'] or list of tuples as (x,y [,z]), x=rotation around horizontal plane, y = rotation on vertical plane, z = rotation on anterior/posterior plane
#   colorbar: Add colorbar [default: True]
#   cmap: colormap to use for roi_map [default: 'Oranges']
#   overlay_method: 'overlay' or 'blend' colors onto the brain [default: overlay]
#
#   electrode_activations:  Nx4 numpy array of electrodes coordinates (columsn 1-3) and activations (4th column)
#   vertex_distance:        distance (in mm) of activation extent
#   vertex_scaling:         method of activation decrease with distance (linear, gaussian, or exponential decay) [default: linear]
#   vertex_scale:           tuple or list of float shape (scale_at_center, scale_at_distance). Extent of scaling [default: (1., 0.)]
#   vertex_method:          Method for combining vertices [default: max]
#
#   coords:                 Nx3 coordinates
#   elec_color:             scaler or Nx3 color array 
#   elec_size:              int [default: 12] 
#   elec_marker:            str or array-like of str [default: 'o'] 
#   edge_color:             str or 1x3 array color of eletrode edges
#   force_to_nearest_vertex: for coordinates to nearest surface vertex [default: False]
#   reduce_model:           int, m to reduce surface model to m faces [default: None, no reduction]
#    
#   brain_cmap: colormap used for brain [default: 'Greys']
#   brain_color_scale: range of cmap values to use [default: (0.42, 0.58) produces narrow grey range used for standrd inflated surfaces]
#   brain_alpha: transparency of the surface
#   figsize: sigure size
#   save_file: filename to save figure to
#   dpi: dpi to save with

# TODO: need to update lighting for non-standard views, (e.g., ventral, dorsal)


    # some checks
    assert type(figsize) is tuple and len(figsize)==2, 'figsize must be tuple of length 2'
    if roi_map is not None:
        assert (parc is not None), 'Parcelelation file is required to map rois'
    if subjects_dir is None:
        subjects_dir = get_subjects_dir(raise_error=True)
    if surface != 'pial':
        # enforce snap to vertex
        force_to_nearest_vertex = True
    
    
    if not os.path.exists(surface):
        # generate from freesurfer folder
        surface = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+surface)
    vertices, faces = read_surface(surface)
    if scanner2tkr is None:
        # get from original mri
        mri_orig = os.path.join( subjects_dir, subject, 'mri', 'orig.mgz')
        scanner2tkr,_ = run_subprocess(['mri_info', '--scanner2tkr', mri_orig])
        scanner2tkr = np.fromstring(scanner2tkr, sep=' ').astype(float)
        scanner2tkr = scanner2tkr.reshape(4,4)
        scanner2tkr = scanner2tkr[0:3,-1]
    vertices -= scanner2tkr
    
    # read in annotation file    
    if parc is not None:
        if not os.path.exists(parc):
            parc = os.path.join( subjects_dir, subject, 'label', hemi + os.path.extsep + parc + '.annot' )
        assert os.path.exists(parc), ('could not locate parcellation file at %s' % parc)            
        annot_vert, annot_tbl, annot_anatomy  = _read_annot( parc )
    
    
    # parse views
    view_strings = {'lh':{'lateral': (0, 180, 0), 'medial': (0, 0, 0), 'rostral': (0,0,0), 'caudal': (180,0,0), 'dorsal': (180,0,90), 'ventral': (0,0,90)},
                    'rh':{'lateral': (90, 0, 0), 'medial': (-90,0, 0), 'rostral': (0,0,0), 'caudal': (180,0,0), 'dorsal': (180,0,90), 'ventral': (0,0,90)},
                    }
    v = []
    for vv in views:
        if type(vv) == str:
            v.append( view_strings[hemi][vv] )
        else:
            v.append( vv+(0,) ) # append tuple(0) to ensure at least 3 dims
    views = v          
    
    elec_vertex_index = []
    if force_to_nearest_vertex is not None and force_to_nearest_vertex:
        # need to do this on pial surface, once vertex index is known, we can snap it to any surface
        if os.path.splitext(surface)[1] == '.pial':
            vertices_pial = vertices
        else:
            surface = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+'pial')
            vertices_pial, _ = read_surface(surface)
            vertices_pial -= scanner2tkr
            
        # find closest vertex for each electrode. We'll use these later for plotting
        tree = spatial.KDTree(vertices_pial)
        _, elec_vertex_index = tree.query(coords)
    
        # nan coordinates get mapped to out-of-range-index, so re-map to center
        idx_ = np.where( elec_vertex_index >= vertices_pial.shape[0])
        center_vrtx = np.argmin( spatial.distance.cdist( vertices_pial, np.zeros((1,3)) ))
        elec_vertex_index[idx_] = center_vrtx
        
        # set coordinates to actual surace vertices used for this plot
        coords = vertices[elec_vertex_index, :]+(-2,0,0)

            
    ##################################            
    # eletrode activations. 
    # Needs to be done before any scaling of vertices since distances are computed
    vertex_activations = []
    if electrode_activations is not None:
        
        # Needs to be done on the pial surface since electrode distance to pial surface is required to map activation. 
        # But after that it can be applied to the inflated since the vertex indices are the same
        if os.path.splitext(surface)[1] == '.pial':
            vertices_pial = vertices
        else:
            surface = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+'pial')
            vertices_pial, _ = read_surface(surface)
            vertices_pial -= scanner2tkr
        
        # find closest vertex
        for i in range( electrode_activations.shape[0] ):
            coords = electrode_activations[i, 0:3]
            val = electrode_activations[i, 3]
            tmp = np.sqrt((vertices_pial[:,0]-coords[0])**2 + (vertices_pial[:,1]-coords[1])**2 + (vertices_pial[:,2]-coords[2])**2)
            vertex_id = np.argmin( tmp )
            tmp = np.sqrt((vertices_pial[:,0]-vertices_pial[vertex_id, 0])**2 + (vertices_pial[:,1]-vertices_pial[vertex_id, 1])**2 + (vertices_pial[:,2]-vertices_pial[vertex_id, 2])**2)
            vertex_nbrs = np.where( tmp < vertex_distance)[0]
            vertex_nbrs = vertex_nbrs[np.argsort( tmp[vertex_nbrs] )]
            
            # update this to be based on distance from center vertex
            if vertex_scaling == 'guassian':
                mx = np.max( tmp[vertex_nbrs] )
                sigma = 0.8/2
                vertex_vals = np.exp(-( ( (tmp[vertex_nbrs]/mx)**2 )/(2*sigma**2)));
            else: #linear
                vertex_vals = val*(1 - tmp[vertex_nbrs]/vertex_distance)
            vertex_activations.append( [vertex_nbrs, vertex_vals] )
        
    
    
    # rescale & center -1/+1
    vert_range = max(vertices.max(0) - vertices.min(0))
    vert_diff = (vertices.max(0) + vertices.min(0)) / 2
    vertices = (vertices - vert_diff) / vert_range
    coords = (coords-vert_diff)/vert_range
    
    
    # instantiate face colors
    face_colors = np.ones((faces.shape[0], 4))
    
    # generate sulcal map definition
    sulc = np.ones(vertices.shape[0]) * 0.5
    if sulc_map is not None:
        # read in sulcal file (typically curv file) 
        if os.path.exists( sulc_map ):
            # full path file
            sulc_file = sulc_map
        else:
            sulc_file = os.path.join( subjects_dir, subject, 'surf', hemi+os.path.extsep+sulc_map )
        assert os.path.exists(sulc_file), ('could not locate sulc_map file at %s' % sulc_file)            
        sulc = read_curvature(sulc_file, binary=False)
    sulc_faces = np.mean(sulc[faces], axis=1)
    
    # binarize sulcal map
    if sulc_faces.min() != sulc_faces.max():
        neg_sulc = np.where(sulc_faces <= 0)
        pos_sulc = np.where(sulc_faces > 0)
        sulc_faces[neg_sulc] = 0
        sulc_faces[pos_sulc] = 1
    
    # mask, unused for now
    # mask = np.zeros(vertices.shape[0]).astype(bool)
    
    # assign greyscale colormap to sulcal map faces
    greys = plt.get_cmap(brain_cmap, 512)
    greys_narrow = ListedColormap(greys(np.linspace(brain_color_scale[0], brain_color_scale[1], 256)))
    face_colors = greys_narrow(sulc_faces)
    face_colors[:,-1] = brain_alpha
    
    
     
    ##################################
    # roi labeling/coloring    
    label_masks  = []
    label_val = []
    if roi_map is not None:
        
        vals = [val for val in roi_map.values()]
        # figure out vmin/vmax from label_val, if vmin/vmax not provided
        if vmin is None:
            vmin = np.min(vals)
        if vmax is None:    
            vmax = np.max(vals)
            
        # colors for roi map based on cmap provided
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # reindex the colormap so it starts deeper into the colorscale (e.g. color values = 0.1-1 over the scale of vmin-vmax (ACC) of purple)
        cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))
        
        # loop through the dictionary and color rois
        for i, roi in enumerate( roi_map ):
            L_mask = np.zeros(vertices.shape[0]).astype(bool)
            idx_ = annot_anatomy.index( roi.encode() )
            vertex_val = annot_tbl[idx_, -1]
            vertex_idx = np.where( annot_vert == vertex_val )[0]
            L_mask[vertex_idx] = 1  # label vertices = 1
            label_masks.append(L_mask)
            label_val.append( roi_map[roi] )
    
        
        label_mask_faces = [np.median(L[faces], axis=1) for L in label_masks]
        # assign label faces to appropriate color
        for i, L in enumerate(label_mask_faces):
            L_idx = np.where(L >= 0.5)
            # blend (multiply) label color with underlying color
            # face_colors[L_idx] = face_colors[L_idx] * [1., 0., 0., 1.]
            if overlay_method == 'blend':
                face_colors[L_idx] = face_colors[L_idx] * cmap(norm(label_val[i]), brain_alpha)
            else:
                face_colors[L_idx] = cmap(norm(label_val[i]), brain_alpha)
    
    
    
    ##################################
    # vertex coloring for activations
    if len(vertex_activations):
        
        # figure out vmin/vmax from activation values
        if vmin is None:
            vmin = np.min( electrode_activations[:,3].min() )
        if vmax is None:    
            vmax = np.max( electrode_activations[:,3].max() )
            
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))
        # compile all vertex indices into a dict and then get mean value per index
        indices = []
        for vertex_nbrs, vertex_vals in vertex_activations:
            indices.append( vertex_nbrs )
        indices = np.concatenate( indices )
        vert_indices = {k: [] for k in indices}
            
        # now add values to them
        for vertex_nbrs, vertex_vals in vertex_activations:
            for i, idx in enumerate(vertex_nbrs):
                vert_indices[ idx ].append( vertex_vals[i] )
                    
        # finally add the face color based on these indices
        for idx in vert_indices.keys():
            idx_ = np.where(faces == idx)[0]
            if overlay_method == 'blend':
                face_colors[idx_] = face_colors[idx_] * cmap(norm( np.max( vert_indices[idx] ) ), brain_alpha)
            else:
                face_colors[idx_] = cmap(norm( np.max( vert_indices[idx] ) ), brain_alpha)
    
    
    # reduce the mesh?
    if reduce_model is not None:
        # this will currently break roi_maps
        vert, fac = decimate_surface(vertices, faces, reduce_model)
        # much faster way to do cdist on large arrays
        tree = spatial.KDTree(vertices)
        _, reduced_i = tree.query(vert)
        
        
        reduced_sulc = sulc[ reduced_i ]
        reduced_sulc_faces = np.mean( reduced_sulc[fac], axis=1)
        
        # binarize sulcal map
        if reduced_sulc_faces.min() != reduced_sulc_faces.max():
            neg_sulc = np.where(reduced_sulc_faces <= 0)
            pos_sulc = np.where(reduced_sulc_faces > 0)
            reduced_sulc_faces[neg_sulc] = 0
            reduced_sulc_faces[pos_sulc] = 1
        
        face_colors = greys_narrow(reduced_sulc_faces)
        vertices = vert
        faces = fac
    
            
    
    face_normals = normal_vectors(vertices, faces)
    light = np.array([0, 0, 1])
    intensity = np.dot(face_normals, light)
    shading = 0.7  # shading 0-1. 0=none. 1=full
    # top 20% all become fully colored
    denom = np.percentile(intensity, 80) - np.min(intensity)
    intensity = (1 - shading) + shading * (intensity - np.min(intensity)) / denom
    intensity[intensity > 1] = 1
            
            
    # apply shading after the facecolors are set
    face_colors[:, 0] *= intensity
    face_colors[:, 1] *= intensity
    face_colors[:, 2] *= intensity
    
    # face_colors = lighting(vertices[faces], direction=(-1,0,0.25),
    #                       color=greys_narrow(0), specular=True)
    
    ##################################
    # Draw the plot
    fig = plt.figure(figsize=figsize)
    axwidth = 1/len(views)
    for i, view in enumerate(views):
    
        # ax = fig.add_axes( (0+(i*axwidth), 0, axwidth, 1), projection='3d',  xlim=[-0.5, +0.5], 	ylim=[-0.6, +0.6],    zlim=[-0.45,0.45], computed_zorder=False)
        ax = fig.add_axes( (0+(i*axwidth), 0, axwidth, 1), projection='3d',  xlim=[np.min(vertices[:,0]), np.max(vertices[:,0])], ylim=[np.min(vertices[:,1]), np.max(vertices[:,1])], zlim=[np.min(vertices[:,2]), np.max(vertices[:,2])], computed_zorder=False)
        # ax = fig.add_subplot( 1, len(views), i+1, projection='3d',  xlim=[-0.6, +0.6], ylim=[-0.6, +0.6],    zlim=[-0.45,0.45])
        # ax = fig.add_subplot( 1, len(views), i+1, projection='3d')
        
        ax.view_init(elev=view[0], azim=view[1], roll=view[2])
        ax.set_proj_type('ortho')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create an instance of a LightSource and use it to illuminate the surface.
        # ls = colors.LightSource(90, 0)
        # illuminated_surface = ls.shade_normals( face_normals, fraction=0.8)
        
        # p = ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces, color=greys_narrow(1), linewidth=0.2, antialiased=True)
        # p.set_facecolor( face_colors )
        
        p = ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces, color=greys_narrow(1), linewidth=0.2, antialiased=True, alpha=brain_alpha, rasterized=True)
        p.set_facecolor( face_colors.tolist() )
        # pc = Poly3DCollection(vertices[faces], facecolors=face_colors)
        # ax.add_collection(pc)
        
        if len(elec_marker)>1:
            for m in np.unique(elec_marker):
                idx_ = np.where( elec_marker == m)[0]
                ax.scatter( coords[idx_,0], coords[idx_,1], coords[idx_,2], color=elec_color[idx_,:], edgecolor=edge_color, zorder=0, s=elec_size, marker=m, rasterized=True, depthshade=True)
        else:
            ax.scatter( coords[:,0], coords[:,1], coords[:,2], color=elec_color, edgecolor=edge_color, s=elec_size, marker=elec_marker, rasterized=True, depthshade=True)
        

        # # plot the eletrodes    
        # radius = 3
        # for center, color in zip( coords, elec_color):
        #     # draw sphere for each coordinate
        #     u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        #     x = radius*np.cos(u)*np.sin(v)+center[0]
        #     y = radius*np.sin(u)*np.sin(v)+center[1]
        #     z = radius*np.cos(v)+center[2]
        
        #     ax.plot_surface(x, y, z, color=color, alpha=1)
        


