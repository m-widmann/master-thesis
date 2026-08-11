import numpy as np
import pandas as pd
import pathlib
import nrrd
from general_util_functions import pickle_load_object, pickle_save_object
from classify_cells_with_regressors import correlation_to_regressors
from plotting_functions import \
    group_plotter_stimuli_separated, add_scale_bar_y, add_scale_bar_x, style_axis
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tslearn.barycenters import softdtw_barycenter
from scipy.stats import wasserstein_distance, mannwhitneyu, false_discovery_control
from joblib import Parallel, delayed
from datetime import datetime
today = datetime.now().strftime('%Y-%m-%d')

color_palette = [
    "#1F77B4FF",
    "#FF7F0EFF",
    "#2CA02CFF",
    "#D62728FF",
    "#9467BDFF",
    "#8C564BFF",
    "#E377C2FF",
    "#7F7F7FFF",
    "#BCBD22FF",
    "#17BECFFF",
]
cell_type_colors = {
    'Luminance Integrator': '#E69F00',
    'Bright': '#F748A5',
    'Dark': '#9F0162',
    'Change': '#D55E00',
    'Motion Onset': '#5CC8F3',
    'Motion Integrator': '#359B73',
    'Slow Motion Integrator': '#2271B2',
}


#%% do correlation with lumi_mot stim to find MI cells
data_path = pathlib.Path(r"data\WARP")
lumi_mot_traces = pd.read_csv(data_path / '2026-06-04_trial-clustering_dots-lumi-stimulus_k6-th03-frac03.tsv',
                              sep='\t', index_col=list(range(8)))
warp_traces = pd.read_csv(data_path / '2026-06-04_trial-clustering_WARP-stimulus_k6-th02-frac03.tsv',
                          sep='\t', index_col=list(range(8)))
segmentation_matches = pickle_load_object(data_path / 'segmentation_matches_dict.pkl')

# keep only cells which have a matched segment for both stimuli sets
keep_tuples = []
cell_map = {}
for fish, plane_dict in segmentation_matches.items():
    for plane, cell_array in plane_dict.items():
        z_plane_val = int(plane)
        for cells in cell_array:
            keep_tuples.append((fish, z_plane_val, cells[0]))
            cell_names = [f'f{fish}r00t000z{plane.zfill(3)}c{cell + 10000}' for cell in cells]
            cell_map[cell_names[0]] = cell_names[1]

keep_index = pd.MultiIndex.from_tuples(keep_tuples, names=['fish_id', 'z_plane', 'cell_number'])
traces_index = lumi_mot_traces.index.names
df_reset = lumi_mot_traces.reset_index().set_index(['fish_id', 'z_plane', 'cell_number'])
filtered_lumi_mot_traces = df_reset[df_reset.index.isin(keep_index)].reset_index().set_index(traces_index)


regressors_omr = (np.load(r"data/ideal_responses/motion_regressors.npy") * 100)[:3]

regressor_corr_df = []
for fish,grouped_df in filtered_lumi_mot_traces.groupby('fish_id'):
    regressor_corr_df.append(
        correlation_to_regressors(
        traces_df=grouped_df.xs('Rhombencephalon', level='region'),
        regressors=regressors_omr[:,None,:],
        min_correlation=0.7,
        cell_type_difference_threshold=0.1,
        quantile_cutoff=0.0,
        stimuli=['dots_ipsi'],
        correlation_method='pearson',
        stimulus_resampling_params = {
            'reg_stim_start':20,
            'reg_stim_end':100,
            'pre_stim_len':0,
            'stim_len':50,
            'post_stim_len':20,
        },
        cell_type_names=['Motion Onset', 'Motion Integrator', 'Slow Motion Integrator']
    ))
regressor_corr_df = pd.concat(regressor_corr_df)
regressor_corr_df.dropna(how='any', inplace=True)
# Fig 14 A
group_plotter_stimuli_separated(
    trace_df=lumi_mot_traces,
    group_df=regressor_corr_df,
    col_name='cell_type',
    colors=cell_type_colors,
    stim_start=0,
    stim_end=50,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_individual_traces=False,
    ylim_percentiles=(10, 95),
    y_scale_length=.1,
    ylim_pad=.2,
    x_scale_length=20,
    x_scale_label='10s',
    stimuli=['lumi_ipsi', 'lumi_contra', 'dots_ipsi', 'dots_contra'],
    max_stim_cols=1,
    show_stim_names=False,
    show_panel_title=False,
    title='Hindbrain Motion Cell Types',
    fig_size=(3,12),
)
#%% Fig S3 A
lumi_mot_regressors = pd.read_csv(r"data/ideal_responses/lumi_mot_regressors.tsv", sep="\t",
                               index_col=[0, 1])  # ['Luminance', 'Bright', 'Dark', 'Motion', 'Difference', 'Drive']
regressors = lumi_mot_regressors.to_numpy().reshape((6, 6, 120))[
    :, [1, 3, 4, 5]]  # lumi ipsi, lumi contra, dots ipsi, dots contra
pure_lumi_regressors = regressors[[0, 1, 2, 4]]

correlation_dfs = []
total_cells = {a:0 for a in ['3dpf', '4dpf', '5dpf']}
for (region,fish),grouped_df in filtered_lumi_mot_traces.groupby(['fish_id','region']):
    if region == "Telencephalon": continue
    corr_df = correlation_to_regressors(
        traces_df=grouped_df,
        regressors=pure_lumi_regressors,
        min_correlation=0.4,
        cell_type_difference_threshold=0.05,
        quantile_cutoff=0.,
        fdr_alpha=.05,
        correlation_method="pearson",
        stimulus_resampling_params={
            'reg_stim_start': 20,
            'reg_stim_end': 100,
            'pre_stim_len': 0,
            'stim_len': 50,
            'post_stim_len': 20
        },
        stimuli=['lumi_ipsi', 'lumi_contra', 'dots_ipsi', 'dots_contra'],
        cell_type_names=['Luminance Integrator', 'Bright', 'Dark', 'Change'],
        n_jobs=8
    )
    correlation_dfs.append(corr_df)
lumi_corr_df = pd.concat(correlation_dfs)

group_plotter_stimuli_separated(
    trace_df=filtered_lumi_mot_traces,
    group_df=lumi_corr_df,
    col_name='cell_type',
    colors=cell_type_colors,
    stim_start=0,
    stim_end=50,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_individual_traces=False,
    ylim_percentiles=(10, 98),
    y_scale_length=.1,
    ylim_pad=.2,
    x_scale_length=20,
    x_scale_label='10s',
    stimuli=['lumi_ipsi', 'lumi_contra', 'dots_ipsi', 'dots_contra'],
    max_stim_cols=1,
    show_stim_names=False,
    show_panel_title=False,
    title='Luminance Cell Types',
    fig_size=(3,12),
)

#%% Fig 15 A
positions_df = pickle_load_object(
    r"data\3to5dpf\2026-06-26_3to5dpf_cell_positions.pkl")
mi_cells =  regressor_corr_df.loc[regressor_corr_df['cell_type'] == 'Motion Integrator'].index.get_level_values("cell_name").values
mi_positions = positions_df.loc[np.isin(positions_df.index.get_level_values("cell_name"), mi_cells)]
mi_coords = mi_positions.loc[:,['z','y','x']].to_numpy()
brain_volume = nrrd.read(r"Z:\Zebrafish atlases\z_brain_atlas\volume_stacks\Elavl3-H2BRFP.nrrd")[0].transpose(
    (2, 1, 0))  # z,y,x
brain_image = brain_volume[60:90,:-206,41:-40].mean(axis=0)

fig,ax = plt.subplots(figsize=(8,15))
ax.imshow(brain_image, cmap='gray')
ax.scatter(y=mi_coords[:,1], x=mi_coords[:,2]-41, c='#359B73', alpha=.6)
style_axis(ax)
plt.show()
#%% plot warp cells and get the barycenters (Fig S3 B)
corr_df_warp_cells = copy(pd.concat([regressor_corr_df, lumi_corr_df])).reset_index()
corr_df_warp_cells['cell_name'] = corr_df_warp_cells['cell_name'].map(cell_map)
corr_df_warp_cells.set_index(regressor_corr_df.index.names, inplace=True)

group_plotter_stimuli_separated(
    trace_df=warp_traces.iloc[:,:30],
    group_df=corr_df_warp_cells,
    col_name='cell_type',
    colors=cell_type_colors,
    stim_start=0,
    stim_end=12,
    max_stim_cols=1,
    show_individual_traces=False,
    stimuli=None,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    x_scale_length=12,
    x_scale_label="3s",
    y_scale_length=.1,
    ylim_pad=.7,
    show_stim_names=True,
    show_panel_title=False,
    title='WARP Responses of Cell Types',
    fig_size=(6,12),
)


#%% Fig 14 D
#get regressors for the Janelia data
mi_cells = corr_df_warp_cells.loc[corr_df_warp_cells['cell_type'] == 'Motion Integrator'].index.get_level_values('cell_name').unique()
mi_warp_barycenters = []
stim_index = []
for stim, df in warp_traces.loc[warp_traces.index.get_level_values('cell_name').
            isin(mi_cells)].iloc[:, :-1].groupby(level='stimulus'):
    mi_warp_barycenters.append(softdtw_barycenter(df.to_numpy(), gamma=.8, max_iter=50))
    stim_index.append(stim)
mi_warp_barycenters = np.array(mi_warp_barycenters).reshape((7,-1))

fig, axes = plt.subplots(2,4,sharex=True, sharey=True)
axes = axes.flatten()
for i,(stim,trace) in enumerate(zip(stim_index, mi_warp_barycenters)):
    ax = axes[i]
    ax.axvspan(0,12,color='lightgrey', alpha=0.3)
    ax.hlines(y=0, color='grey', linestyle=':', xmin=0, xmax=30)
    ax.plot(trace[:30], color=color_palette[4], linewidth=2)
    ax.set_title(str(stim).replace("_", " "))
for i,ax in enumerate(axes):
    if i==0:
        add_scale_bar_y(ax, y_scale_length=.1, y_scale_label=r'0.1$\Delta$F/F0')
        add_scale_bar_x(ax, x_scale_length=12, x_scale_label='3s')
    style_axis(ax)
plt.show()

pickle_save_object((mi_warp_barycenters, stim_index), f'{today}_motion_integrator_warp_responses.pkl')
#%%
lumi_mot_regressors = pd.read_csv(r"data/ideal_responses/lumi_mot_regressors.tsv", sep="\t",
                               index_col=[0, 1])  # ['Luminance', 'Bright', 'Dark', 'Motion', 'Difference', 'Drive']
regressors = lumi_mot_regressors.to_numpy().reshape((6, 6, 120))[
    :, [1, 3, 4, 5]]  # lumi ipsi, lumi contra, dots ipsi, dots contra
pure_lumi_regressors = regressors[[0, 1, 2, 4]]

regressor_corr_df = []
for fish,grouped_df in filtered_lumi_mot_traces.groupby('fish_id'):
    regressor_corr_df.append(
        correlation_to_regressors(
        traces_df=grouped_df,
        regressors=pure_lumi_regressors,
        min_correlation=0.1,
        cell_type_difference_threshold=0.05,
        quantile_cutoff=0.8,
        stimuli=['lumi_ipsi', 'lumi_contra', 'dots_ipsi', 'dots_contra'],
        correlation_method='pearson',
        stimulus_resampling_params={
            'reg_stim_start':20,
            'reg_stim_end':100,
            'pre_stim_len':0,
            'stim_len':50,
            'post_stim_len':20,
        },
        cell_type_names=['Luminance Integrator', 'Bright', 'Dark', 'Change']
    ))
lumi_corr_df = pd.concat(regressor_corr_df)
lumi_corr_df.dropna(how='any', inplace=True)
# Fig S3 A
group_plotter_stimuli_separated(
    trace_df=lumi_mot_traces,
    group_df=lumi_corr_df,
    col_name='cell_type',
    colors=cell_type_colors,
    stim_start=0,
    stim_end=50,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_individual_traces=False,
    ylim_percentiles=(5, 95),
    y_scale_length=.1,
    ylim_pad=.5,
    x_scale_length=20,
    x_scale_label='10s',
    stimuli=['lumi_ipsi', 'lumi_contra', 'dots_ipsi', 'dots_contra'],
    title='Luminance Cell Types'
)
#%% Fig S3 B
lumi_warp_cells = copy(lumi_corr_df).reset_index()
lumi_warp_cells['cell_name'] = lumi_warp_cells['cell_name'].map(cell_map)
lumi_warp_cells.set_index(lumi_corr_df.index.names, inplace=True)


group_plotter_stimuli_separated(
    trace_df=warp_traces.iloc[:,:30],
    group_df=lumi_warp_cells,
    col_name='cell_type',
    colors=cell_type_colors,
    stim_start=0,
    stim_end=12,
    max_stim_cols=7,
    show_individual_traces=False,
    stimuli=None,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    x_scale_length=4,
    x_scale_label="2s",
    y_scale_length=.1,
    ylim_pad=.7,
    title='Lumi WARP Responses of Clusters',
)


#%% load janelia data
warp_data_dir = pathlib.Path(r"warp_data\postprocessed")
# https://figshare.com/s/d1d19b105c4f74865c32
dff_traceAllavg = np.load(warp_data_dir / 'dff_traceAllavg.npy')
stim_timings = np.load(warp_data_dir / 'stim_timings_ds.npy')
# stimuli in open-loop were the following: forward visual motion, backward visual motion, rightward visual motion,
# leftward visual motion, light flash, dark flash, right loom (expanding dot) and a left loom
# but this order does not match with the representation in the figures
np.median(stim_timings[:,:,1] - stim_timings[:,:,0], axis=1)
# the darkflash seems shorter (5 samples vs 7 samples then the other stimuli), so will exclude it for now
stim_timings = np.concat([stim_timings[:4], stim_timings[5:]])
# pre_duration = 5, stim_duration = 7, post_duration = 10 from https://github.com/Zebrafish-WARP/WARP/blob/main/WARP/stimulus_response_utils.py
# inbetween 7s (= 14 frames) of closed loop
stim_start_times = [19 + i * 17  for i in range(8) if i != 4]

#filter cells with activity
non_nan_non_zero = ~(np.isnan(dff_traceAllavg).all(axis=1) + (dff_traceAllavg == 0.).all(axis=1))
traces_stim_separated = np.array([
    dff_traceAllavg[:,start:start+17] for start in stim_start_times
]).transpose(1,0,2)  # cells x stimuli x timepoints
activity_thresholds = np.quantile(np.max(np.abs(traces_stim_separated[non_nan_non_zero]), axis=2), .5, axis=0)
mask = copy(non_nan_non_zero)
mask[non_nan_non_zero] = ((np.max(np.abs(traces_stim_separated[non_nan_non_zero]), axis=2) >= activity_thresholds).sum(axis=1) >= 2)


#load gene expression data
gene_expression_data = np.load(warp_data_dir / 'genes_df_All.npy')[mask]
# index 16 is cx43 marker for glia cells
glia_cells =  gene_expression_data[:,16].astype(bool)
#update mask to not include gila
mask[mask] = ~glia_cells


# load coordinates for cells and remove cells which don't have any
coordinates = np.load(warp_data_dir / 'Coords_All.npy') #z,x,y

# mapzebrain midline is ~285
midline = 284.
hemisphere = []
for coords in coordinates:
    if np.isnan(coords).any():
        out = '-'
        hemisphere.append(out)
    else:
        diff = coords[1] - midline
        if abs(diff) < 5:
            out = '-'
        else:
            out = 'left' if diff < 0 else 'right'
        hemisphere.append(out)
hemisphere = np.array(hemisphere)

mask = np.bitwise_and(mask, hemisphere != '-')
filt_hemisphere = hemisphere[mask]

#load expression cluster data
cluster_labels = np.load(warp_data_dir / 'cluster_labelsAll2.npy')[mask]
cluster_names = np.concat([[None],np.load(warp_data_dir / 'good_cls_names.npy')])
cell_numbers = np.where(mask)[0]

stim_names = ['motion_forward', 'motion_backward', 'motion_right', 'motion_left', 'bright', 'loom_right', 'loom_left']
index_tuples = []
for cell,ipsi,clstr in zip(cell_numbers, filt_hemisphere, cluster_labels):
    for stim in stim_names:
        contra = 'right' if ipsi=='left' else 'left'
        if stim != 'bright':
            stim = stim.replace(ipsi, 'ipsi').replace(contra, 'contra')
        index_tuples.append((cell, clstr, stim, ))
janelia_traces = pd.DataFrame(traces_stim_separated[mask].reshape((-1,17)))
janelia_traces.index = pd.MultiIndex.from_tuples(index_tuples, names=['cell_number', 'cluster', 'stimulus'])

# pickle_save_object((janelia_traces,mask), rf"data\WARP\{today}_janelia_trace_df.pkl")
# janelia_traces,mask = pickle_load_object(r"data\WARP\2026-07-24_janelia_trace_df.pkl")
warp_MI_regressors, stimuli = pickle_load_object(r"data\WARP\\2026-07-15_motion_integrator_warp_responses.pkl")

brain_regions_arr = np.load(warp_data_dir / 'BrainRegions_All.npy')
region_names = np.load(warp_data_dir.parent / "Fish1/region_names.npy")
hb_cells = np.where(brain_regions_arr[:,np.where(region_names == 'rhombencephalon_(hindbrain)')[0][0]])[0]
coordinates = np.load(warp_data_dir / 'Coords_All.npy')
mi_y_bounds = (460,630)
mi_x_bounds = (170,400)
hb_coords = coordinates[hb_cells]
in_bounds = [mi_x_bounds[0] < x < mi_x_bounds[1] and mi_y_bounds[0] < y < mi_y_bounds[1]  for x,y in hb_coords[:,1:]]
selected_hb_cells = hb_cells[in_bounds]
warp_corr_df_full = correlation_to_regressors(
    traces_df=janelia_traces,
    regressors=warp_MI_regressors[None,-4:,:],
    cell_id_level_name='cell_number',
    min_correlation=0.,
    cell_type_difference_threshold=0.,
    quantile_cutoff=0.,
    stimuli=stimuli[-4:],
    correlation_method='pearson',
    cell_type_names=['Motion Integrator'],
    stimulus_resampling_params = {
        'reg_stim_start':0,
        'reg_stim_end':12,
        'pre_stim_len':0,
        'stim_len':7,
        'post_stim_len':10,
    },
)
warp_corr_df_full.dropna(how='any', inplace=True)
warp_corr_df = copy(warp_corr_df_full.loc[warp_corr_df_full["correlation_value"] >= 0.7])
warp_mi_cells = np.intersect1d(warp_corr_df.index.values, selected_hb_cells)
coordinates = np.load(warp_data_dir / 'Coords_All.npy') #z,x,y

# Fig 15 B

# warp cells are registered to mapzebrain but plot them on the zbrain to compare
# so there is some shifting to be done
mapzebrain_vol = nrrd.read(r"Z:\Zebrafish atlases\mapzebrain_atlas2024\volume_stacks\T_AVG_elavl3GCaMP6s.nrrd")[0] #x,y,z
mzb_img = mapzebrain_vol[:,:,150:300].mean(axis=2).transpose(1,0)
mzb_cropped = mzb_img[2:-32,65:-(37+65)]
brain_volume = nrrd.read(r"Z:\Zebrafish atlases\z_brain_atlas\volume_stacks\Elavl3-H2BRFP.nrrd")[0].transpose(
    (2, 1, 0))  # z,y,x
brain_image = brain_volume[60:90,:-206,41:-40].mean(axis=0)
# almost the same crop
x_scale = brain_image.shape[1] / mzb_cropped.shape[1]
y_scale = brain_image.shape[0] / mzb_cropped.shape[0]
shifted_coordinates = np.array([(coordinates[:,1] - 65) * x_scale, (coordinates[:,2] - 2) * y_scale]).T #x,y

warp_mi_coords = shifted_coordinates[warp_corr_df.index.values,]
hb_mi_coords = shifted_coordinates[warp_mi_cells]

fig, ax = plt.subplots(figsize=(8,15))
ax.imshow(brain_image, cmap='gray')
ax.scatter(x=warp_mi_coords[:,0], y=warp_mi_coords[:,1], alpha=0.6, c=color_palette[0])
ax.scatter(x=hb_mi_coords[:,0], y=hb_mi_coords[:,1], alpha=0.6, c=color_palette[1])
style_axis(ax)
plt.show()

#%% Fig 15 C
group_plotter_stimuli_separated(
    trace_df=janelia_traces,
    group_df=warp_corr_df.loc[~np.isin(warp_corr_df.index, warp_mi_cells)],
    col_name='cell_type',
    cell_id_index_name='cell_number',
    stim_start=0,
    stim_end=7,
    max_stim_cols=4,
    show_individual_traces=False,
    stimuli=['motion_forward', 'motion_backward', 'motion_ipsi', 'motion_contra', 'loom_ipsi', 'loom_contra'],
    use_same_axes_for_all_plots=True,
    show_scale_bar=True,
    colors=[color_palette[0],],
    x_scale_length=2,
    x_scale_label="1s",
    y_scale_length=.1,
    ylim_percentiles=[2,95],
    ylim_pad=.25,
    title='all WARP Cells correlated with MI cells',
    fig_size=(8,3)
)

#%% Fig 15 D
group_plotter_stimuli_separated(
    trace_df=janelia_traces,
    group_df=warp_corr_df.loc[warp_mi_cells],
    col_name='cell_type',
    cell_id_index_name='cell_number',
    stim_start=0,
    stim_end=7,
    max_stim_cols=4,
    show_individual_traces=False,
    stimuli=['motion_forward', 'motion_backward', 'motion_ipsi', 'motion_contra', 'loom_ipsi', 'loom_contra'],
    use_same_axes_for_all_plots=True,
    show_scale_bar=True,
    colors=[color_palette[1],],
    x_scale_length=2,
    x_scale_label="1s",
    y_scale_length=.1,
    ylim_percentiles=[2,95],
    ylim_pad=.25,
    title='selected WARP Cells correlated with MI cells',
    fig_size=(8,3)
)

#%% Fig 16
gene_expression_data = np.load(warp_data_dir / 'genes_df_All.npy')
gene_names = np.array([
    'cart2', 'glyt2', 'tac1', 'pvalb7', 'npb', 'grm1b', 'irx1b', 'dat', 'net', 'calb1',
    'penka', 'penkb', 'eomesa', 'emx3', 'cfos', 'gad1b', 'cx43', 'vglut2a', 'sst', 'uts1',
    'pou4f2', 'cort', 'nr4a2a', 'cckb', 'tph2', 'chata', 'calb2a', 'npy', 'gfra1a', 'dmbx1a',
    'gbx2', 'crhb', 'nefma', 'chodl', 'pyya', 'zic2a', 'th', 'pdyn', 'tbr1b', 'otpa', 'esrrb'
    ])
subset_gene_data = gene_expression_data[warp_mi_cells]
hindbrain_gene_data = gene_expression_data[selected_hb_cells]

subset_gene_data = np.delete(subset_gene_data, 16, 1)
hindbrain_gene_data = np.delete(hindbrain_gene_data, 16, 1)
gene_names = np.delete(gene_names, 16)

def _wasserstein_parallel(x,n,rng):
    rng.shuffle(x)
    dist = wasserstein_distance(x[:n], x)
    return dist

def permutation_pvalue_wasserstein(subset, full, n_perm=1000, seed=0, do_parallel=False):
    rng = np.random.default_rng(seed)
    observed = wasserstein_distance(subset, full)
    n = len(subset)

    if do_parallel:
        null_dists = Parallel(n_jobs=-1)(
            delayed(_wasserstein_parallel)(full, n, rng)
        for i in range(n_perm))
    else:
        null_dists = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(full)
            null_dists[i] = wasserstein_distance(full[:n], full)
    pval = (null_dists >= observed).mean()
    return observed, pval, np.median(null_dists)


u_test = []
diff_in_mean = []
for subset,full,gene in zip(subset_gene_data.transpose(1,0), hindbrain_gene_data.transpose(1,0),gene_names):
    u_test.append(mannwhitneyu(subset,full,alternative='two-sided'))
    diff_in_mean.append(np.nanmedian(full[full > 0]) - np.nanmedian(subset[subset > 0]))
u_test = np.array(u_test)
u_test_direction =  np.array(["higher" if x > 0 else "lower" for x in diff_in_mean])


wasserstein = []
for i in range(len(gene_names)):
    full = hindbrain_gene_data[:,i]
    subset = subset_gene_data[:,i]
    wasserstein.append(permutation_pvalue_wasserstein(subset, full, n_perm=10000, seed=0, do_parallel=True))
wasserstein = np.array(wasserstein)
wasserstein[:,1] = false_discovery_control(wasserstein[:,1], method='bh')

p_values = false_discovery_control(u_test[:,1], method='bh')
neg_log10_p = -np.log10(p_values)
effect_sizes = np.array([e * -1 if d == 'lower' else e for e,d in zip(np.log1p(wasserstein[:,0]), u_test_direction)])

signif_mask = np.bitwise_and(abs(wasserstein[:,0] - wasserstein[:,2]) >= 1.0, p_values < 0.05)
up_index = np.where(np.bitwise_and(signif_mask, u_test_direction == 'higher'))[0]
down_index =  np.where(np.bitwise_and(signif_mask, u_test_direction == 'lower'))[0]

fig,ax = plt.subplots()
ax.scatter(x=effect_sizes[~signif_mask], y=neg_log10_p[~signif_mask], c='grey')
ax.scatter(x=effect_sizes[up_index], y=neg_log10_p[up_index], c=color_palette[3])
ax.scatter(x=effect_sizes[down_index], y=neg_log10_p[down_index], c=color_palette[0])
max_x = np.max(abs(effect_sizes)) * 1.05
max_y = np.max(neg_log10_p) * 1.05
ax.set_xlim([-max_x, max_x])
ax.set_yticks(list(range(0,int(round(max_y,-1) + 1),10)))
ax.set_xlabel('log(1 + effect size)')
ax.set_ylabel(r'-log$_{10}$(fdr)')
for i in np.concat([up_index, down_index], axis=0):
    label = gene_names[i]
    x = effect_sizes[i]
    y = neg_log10_p[i] + 1
    ax.text(x, y, label, ha='center', va='center', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%%
subset_gene_counts_up = np.log1p(subset_gene_data[:, up_index])
non_zeros = subset_gene_counts_up != 0
for i in range(subset_gene_counts_up.shape[1]):
    values = subset_gene_counts_up[non_zeros[:,i], i]
    subset_gene_counts_up[non_zeros[:, i], i] = (values - values.min()) / (values.max() - values.min()) + 0.1
cmap = plt.cm.plasma.copy()
cmap.set_under('gray')
norm = mcolors.Normalize(vmin=0.09, vmax=subset_gene_counts_up.max())

# do binary heatmap first to get nice clustering of cells
cg = sns.clustermap(
    subset_gene_counts_up.astype(bool),
    row_cluster=True,
    col_cluster=False,
    cmap=cmap,
    norm=norm,)
cell_order = np.array(cg.dendrogram_row.reordered_ind)
plt.close()
# Fig 17
cg = sns.clustermap(
    subset_gene_counts_up[cell_order,:],
    row_cluster=False,
    col_cluster=False,
    cmap=cmap,
    norm=norm,)
plt.setp(cg.ax_heatmap.set_xticklabels(gene_names[up_index]), rotation=0, horizontalalignment='center')

#%%
up_genes = np.array(['gad1b', 'vglut2a', 'otpa', 'irx1b', 'npb', 'tac1'])
up_index = np.where(np.isin(gene_names, up_genes))[0]
from itertools import combinations
index_pairs = list(combinations(up_index,2))
expression_mask = hindbrain_gene_data[:,up_index] != 0
expression_selected_genes = selected_hb_cells[expression_mask.any(axis=1)]
subset_df = pd.DataFrame({"cell_number":selected_hb_cells})
for i in up_index:
    subset_df[gene_names[i]] = hindbrain_gene_data[:,i] != 0
subset_df.set_index('cell_number', inplace=True)
n_exp = subset_df.sum(axis=1)
mult_exp = {}
for i in range(2,7):
    for t in combinations(subset_df.columns,i):
        t = list(t) #tuples don't work as column keys
        # all genes in t are expressed
        mask = subset_df[t].all(axis=1)
        # no additional genes are expressed
        mask &= (n_exp == i)
        mult_exp[",".join(t)] = mask
mult_exp_df = pd.DataFrame(mult_exp)
#
#%% Fig 18
correlation_data = {}
for gene in mult_exp_df.columns:
    corr_values = warp_corr_df_full.loc[np.isin(warp_corr_df_full.index, mult_exp_df.loc[mult_exp_df[gene]].index),'correlation_value'].values
    if len(corr_values) >= 20:
        correlation_data[gene] = corr_values

corr_quantiles = np.quantile(warp_corr_df_full['correlation_value'].values, q=[0.25, .5, .75, .9])
all_group_medians = np.array([np.median(v)for v in correlation_data.values()])
label_order = np.array(list(correlation_data.keys()))[np.argsort(all_group_medians)[::-1]]
fig, ax = plt.subplots()
for y,c in zip(corr_quantiles, ["#001889", "#000000", "#BF2D72", "#F7A72B"]):
    ax.axhline(y, xmin=0, xmax=len(correlation_data) + 1, color=c, linestyle="--", alpha=0.8, lw=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
rng = np.random.default_rng(0)


for i, group in enumerate(label_order, start=1):
    values = correlation_data[group]
    # horizontal jitter
    x = i + rng.normal(0, 0.05, len(values))
    ax.scatter(
        x,
        values,
        c=color_palette[0],
        linewidths=.5,
        s=10,
        alpha=0.4,
    )
    # median
    med = np.median(values)
    ax.plot([i - 0.2, i + 0.2], [med, med],
            color="red", lw=3)

ax.set_xticks(range(1, len(correlation_data) + 1))
ax.set_xticklabels([k + f"\n{len(correlation_data[k])} cells" for k in label_order], rotation=90, ha="right")
# One-vs-rest tests
pvals = []
for group in label_order:
    x = correlation_data[group]
    group_cells = mult_exp_df.index.values[mult_exp_df[group]]
    y = warp_corr_df_full.loc[~np.isin(warp_corr_df_full.index, group_cells),'correlation_value'].dropna().values
    _, p = mannwhitneyu(x, y, alternative="greater")
    pvals.append(p)

# Benjamini-Hochberg FDR correction
pvals_corr = false_discovery_control(pvals, method="bh")

# Height just above each box
ymin, ymax = ax.get_ylim()
offset = 0.03 * (ymax - ymin)
for i, (group, p) in enumerate(zip(label_order, pvals_corr), start=1):
    if p < 0.0005:
        txt = "***"
    elif p < 0.005:
        txt = "**"
    elif p < 0.0025:
        txt = "*"
    else:
        continue
    ymax_group = max(correlation_data[group])
    ax.text(
        i,
        ymax_group + offset,
        txt,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylabel(r"correlation coefficient $\rho$")

plt.tight_layout()
plt.show()
#%% Fig S4
category_df = pd.DataFrame(mult_exp_df.idxmax(axis=1), columns=['category',])
group_plotter_stimuli_separated(
    trace_df=janelia_traces,
    group_df=category_df.loc[np.isin(category_df["category"], label_order[:6])],
    col_name='category',
    cell_id_index_name='cell_number',
    stim_start=0,
    stim_end=7,
    max_stim_cols=1,
    show_individual_traces=False,
    stimuli=None,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_stim_names=True,
    show_scale_bar=True,
    colors=None,
    x_scale_length=2,
    x_scale_label="1s",
    y_scale_length=.1,
    ylim_percentiles=[2,95],
    ylim_pad=.25,
    title=None,
    fig_size=(3,10)
)

