import pandas as pd
import numpy as np
import pathlib
import h5py
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from copy import deepcopy
from itertools import product, chain
from scipy.ndimage import gaussian_filter
from analysis_helpers.analysis.personal_dirs.Max_W.two_p_image_analysis.classify_cells_with_regressors \
    import correlation_to_regressors
from analysis_helpers.analysis.personal_dirs.Max_W.utils.general_toolbox import pickle_load_object, pickle_save_object
from analysis_helpers.analysis.personal_dirs.Max_W.utils.plotting_functions import \
    group_plotter_stimuli_separated, plot_active_regions_on_brain, plot_boxplots_corr_values
from analysis_helpers.analysis.personal_dirs.Max_W.two_p_image_analysis.response_correlation_analysis import \
    HdbscanCellClusterer, StimulusCrossCorrelation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from joblib.externals.loky import get_reusable_executor
import gc
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')

plot_dir = pathlib.Path(r"Z:\Bahl lab member directories\Max W\2026\master_thesis_plots")

# load data
full_traces_df, cell_to_age = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-24_3to5dpf_trace_data_reliability_filtered.pkl")

cell_type_colors = {
    'Luminance Integrator': '#E69F00',
    'Bright': '#F748A5',
    'Dark': '#9F0162',
    'Change': '#D55E00',
    'Slow Motion Integrator': '#2271B2',
    'Motion Onset': '#5CC8F3',
    'Motion Integrator': '#359B73',
}
age_colors = {
    '3dpf': '#FFD300',
    '4dpf': '#8400CD',
    '5dpf': '#2D2D34'
}

# %% corr with lumi cell types
katja_regressors = pd.read_csv(r"C:\Users\ag-bahl\Desktop\Max\master_thesis\Katjas_model_regressors.tsv", sep="\t",
                               index_col=[0, 1])  # ['Luminance', 'Bright', 'Dark', 'Motion', 'Difference', 'Drive']
regressors = katja_regressors.to_numpy().reshape((6, 6, 120))[
    :, [1, 3, 4, 5]]  # lumi ipsi, lumi contra, dots ipsi, dots contra
pure_lumi_regressors = regressors[[0, 1, 2, 4]]

correlation_dfs = []
total_cells = {a:0 for a in ['3dpf', '4dpf', '5dpf']}
for (age, region), grouped_df in full_traces_df.groupby(level=('age', 'region')):
    if region == "Telencephalon": continue
    print('\n-----------------------------------')
    print(f'{age} - {region}')
    n_cells = len(grouped_df.index.get_level_values('cell_name').unique())
    total_cells[age] += n_cells
    print(f"{n_cells} cells")
    corr_df = correlation_to_regressors(
        traces_df=grouped_df,
        regressors=pure_lumi_regressors,
        min_correlation=0.1,
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
        stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', 'lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        cell_type_names=['Luminance Integrator', 'Bright', 'Dark', 'Change'],
        n_jobs=8
    )
    corr_df['age'] = age
    corr_df['region'] = region
    correlation_dfs.append(corr_df)

lumi_corr_df = pd.concat(correlation_dfs)
lumi_corr_df.set_index(['region', 'age'], append=True, inplace=True)
lumi_corr_df.dropna(how='any', inplace=True)

# %% plotting lumi cells
fig = group_plotter_stimuli_separated(
    trace_df=full_traces_df,
    group_df=lumi_corr_df,
    col_name='cell_type',
    index_level_name='age',
    stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', ],
    stim_start=0,
    stim_end=50,
    colors=cell_type_colors,
    display_number_of_cells=False,
    display_fraction_of_cells=True,
    total_number_of_cells=total_cells,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_stim_names=False,
    show_panel_title=False,
    max_stim_cols=2,
    fig_size=(8, 4),
    x_scale_label='10s',
    x_scale_length=20,
    y_scale_length=0.1,
    y_scale_unit=r'$\Delta$F/$F_{0}$',
    ylim_pad=.5,
    title=None,
)
fig.savefig(plot_dir / 'lumi_traces.svg')

# %% correlation with pure motion hb
regressors_flo = np.load(r"C:\Users\ag-bahl\Desktop\Max\master_thesis\Flo_kmeans_regressors.npy") * 100
motion_regressors = regressors_flo[:-1, None, :]  # ['motion_onset', 'motion_integrator_1', 'slow_motion_integrator']

correlation_dfs = []
total_cells = {a:0 for a in ['3dpf', '4dpf', '5dpf']}
for (age, region), grouped_df in full_traces_df.groupby(level=('age', 'region')):
    if region == "Rhombencephalon":
        print('\n-----------------------------------')
        print(f'{age} - {region}')
        n_cells = len(grouped_df.index.get_level_values('cell_name').unique())
        total_cells[age] += n_cells
        print(f"{n_cells} cells")
        corr_df = correlation_to_regressors(
            traces_df=grouped_df,
            regressors=motion_regressors,
            min_correlation=0.1,
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
            stimuli=['lumi_off_dots_ipsi'],
            cell_type_names=['Motion Onset', 'Motion Integrator', 'Slow Motion Integrator'],
            n_jobs=8
        )
        corr_df['region'] = region
        corr_df['age'] = age
        correlation_dfs.append(corr_df)

motion_corr_df = pd.concat(correlation_dfs)
motion_corr_df.set_index(['region', 'age'], append=True, inplace=True)
motion_corr_df.dropna(how='all', inplace=True)


# %% plot correlations as box plots
combined_corr_df = pd.concat([lumi_corr_df, motion_corr_df])
plot_boxplots_corr_values(combined_corr_df, colors=cell_type_colors, do_stat_test=True, print_pvalues=True)


# %% plot reliability as box plots
reliability_dict ={k:v for k,v in zip(
    full_traces_df.index.get_level_values('cell_name').values,
    full_traces_df['reliability'].values)
                   }

combined_corr_df['reliability'] = combined_corr_df.index.get_level_values('cell_name').map(reliability_dict)
plot_boxplots_corr_values(
    combined_corr_df,
    data_col_name='reliability',
    colors=cell_type_colors,
    do_stat_test=False,
)


#%% get correlation values

last_c = None
for (c,a),df in combined_corr_df.groupby(['cell_type', 'age']):
    if last_c != c:
        print(f'\n{c}')
    last_c = c
    corr_val = np.round(np.nanmedian(df['correlation_value'].values),2)
    print(f'\t{a} {corr_val}')


# %% plotting the motion correlated cells
fig = group_plotter_stimuli_separated(
    trace_df=full_traces_df,
    group_df=motion_corr_df,
    col_name='cell_type',
    index_level_name='age',
    stimuli=['lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
    stim_start=0,
    stim_end=50,
    colors=cell_type_colors,
    display_number_of_cells=False,
    display_fraction_of_cells=True,
    total_number_of_cells=total_cells,
    use_same_axes_for_all_plots=False,
    use_same_axes_for_one_cell_type=True,
    show_panel_title=False,
    show_stim_names=False,
    max_stim_cols=2,
    fig_size=(8, 3),
    x_scale_label='10s',
    x_scale_length=20,
    y_scale_length=0.1,
    y_scale_unit=r'$\Delta$F/$F_{0}$',
    ylim_pad=.5,
    title=None,
)
fig.savefig(plot_dir / 'motion_traces.svg')
# %% plots cells as heatmaps on brain
brain_volume = nrrd.read(r"Z:\Zebrafish atlases\z_brain_atlas\volume_stacks\Elavl3-H2BRFP.nrrd")[0].transpose(
    (2, 1, 0))  # z,y,x
brain_image = brain_volume[60:90,325:-431,41:-40].mean(axis=0)
positions_df = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-26_3to5dpf_cell_positions.pkl")
# set index to just cell_name to extract coordinates more easily
coordinate_df = deepcopy(positions_df)
coordinate_df.reset_index(drop=True, inplace=True)
coordinate_df['cell_name'] = positions_df.index.get_level_values('cell_name').values
coordinate_df.set_index('cell_name', inplace=True)
coordinate_df['x'] = coordinate_df['x'] - 41
coordinate_df['y'] = coordinate_df['y'] - 325


xmin,xmax = 0, 540
ymin,ymax = 0, 650
pixel_size = 6
ybins = (ymax-ymin)//pixel_size
xbins = (xmax-xmin)//pixel_size
#%%
fig,axes = plt.subplots(nrows=7, ncols=3)
for i,(age,age_df) in enumerate(combined_corr_df.groupby('age')):
    for j,(cell_type,df) in enumerate(age_df.groupby('cell_type')):
        cell_names = df.index.get_level_values('cell_name').unique().values
        coordinates = coordinate_df.loc[cell_names]
        counts, _, _ = np.histogram2d(
            coordinates['x'].values, coordinates['y'].values, bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]]
        )
        smooth_2dhist = gaussian_filter(counts, sigma=.5).T #flip to y,x for imshow
        base_color = cell_type_colors[cell_type]
        rgb = np.array(mcolors.to_rgb(base_color))
        bright_rgb = rgb + (1 - rgb) * 0.5
        cmap = LinearSegmentedColormap.from_list(
            f"custom_cmap_{cell_type}",
            [
                (0.0, "black"),
                (0.7, base_color),  # reach the base color at 70%
                (1.0, bright_rgb),  # brightest color at the top
            ]
        )
        vmin, vmax = np.quantile(smooth_2dhist[smooth_2dhist > 0], .3), np.quantile(smooth_2dhist, .99)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(smooth_2dhist))  # (H, W, 4), alpha channel currently all 1
        max_alpha = 0.8  # overall opacity cap
        is_zero = smooth_2dhist < np.quantile(smooth_2dhist[smooth_2dhist > 0], .5)
        rgba[..., 3] = np.where(is_zero, 0.0, max_alpha)  # set alpha to 0 for these bins

        ax = axes[j,i]
        ax.imshow(brain_image, cmap='gray')
        heat = ax.imshow(
            rgba,
            extent=[xmin, xmax, ymax, ymin, ],
            origin='upper',
            zorder=1,
        )
        for s in ax.spines:
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
fig.subplots_adjust(hspace=0., wspace=0.)
plt.show()

# %% active cell fractions
cell_numbers = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-24-cell_numbers_3to5dpf.pkl")
# Prepare data storage
data_dict = {}
for (age, region), df in full_traces_df.groupby(['age', 'region']):
    if region == "Telencephalon":
        continue
    # get the total number of planes
    index_levels = [df.index.names.index(lvl) for lvl in ['fish_id', 'z_plane', 'repeat']]
    num_planes = len(np.unique([np.array(i)[index_levels] for i in df.index], axis=0))

    # Calculate total cells for this region and age
    total_cells = cell_numbers.loc[
        cell_numbers.index.get_level_values('age') == age
        ].xs(region, level='region').sum()

    # Normalize by the number of planes
    normalized_total = total_cells / num_planes
    num_active_cells = len(df.index.get_level_values('cell_name').unique())
    normalized_active = num_active_cells / num_planes
    fraction = num_active_cells / total_cells  # Fraction remains unchanged

    data_dict[(age, region)] = {
        'normalized_total': normalized_total,
        'normalized_active': normalized_active,
        'fraction': fraction
    }

# Create figure with 3 subplots (one for each region)
regions = ['Diencephalon', 'Mesencephalon', 'Rhombencephalon']
fig, axes = plt.subplots(1, 3, sharey=False)
fig.suptitle('Active Cells per Plane by Region and Age', fontsize=16)

# Plot each region in its own subplot
for i, region in enumerate(regions):
    ax = axes[i]
    ax.set_title(region, fontsize=14)

    # Filter data for this region
    region_data = {age: data_dict[(age, region)] for (age, reg) in data_dict if reg == region}

    # Sort ages
    ages = sorted(region_data.keys())
    x = np.arange(len(ages))
    width = 0.6

    # Plot normalized total cells (shaded)
    totals = [region_data[age]['normalized_total'] for age in ages]
    ax.bar(x, totals, width, color='lightgray', alpha=0.5)

    # Plot normalized active cells (solid, on top)
    actives = [region_data[age]['normalized_active'] for age in ages]
    ax.bar(x, actives, width, color=[age_colors[age] for age in ages])

    # Add fraction labels on top of bars
    ymin, ymax = ax.get_ylim()
    for j, age in enumerate(ages):
        fraction = region_data[age]['fraction']
        ax.text(j, totals[j] + ymax * .01, f'{fraction:.2f}', ha='center', fontsize=10)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if region == "Diencephalon":
        ax.set_ylabel('#Cells / #Planes')

    if region == "Rhombencephalon":
        # After plotting all bars
        legend_elements = [
            mpatches.Patch(facecolor='lightgray', alpha=0.5, label='Total cells per plane'),
            mpatches.Patch(facecolor=age_colors['3dpf'], label='Active fraction (3dpf)'),
            mpatches.Patch(facecolor=age_colors['4dpf'], label='Active fraction (4dpf)'),
            mpatches.Patch(facecolor=age_colors['5dpf'], label='Active fraction (5dpf)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
plt.show()

# %% load masks and plot areas that change

mask_path = r"Z:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\all_masks_indexed.hdf5"
with h5py.File(mask_path, 'r') as f:
    brain_region_masks = {k: np.array(f[k]['ind_mask_volume']) for k in f.keys()}  # 3 arrays z,y,x
brain_volume = nrrd.read(r"Z:\Zebrafish atlases\z_brain_atlas\volume_stacks\Elavl3-H2BRFP.nrrd")[0].transpose(
    (2, 1, 0))  # z,y,x

# %% get cell numbers
cell_numbers = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-26-cell_numbers_3to5dpf_(fine_regions).pkl")
cell_numbers_df = pd.DataFrame(cell_numbers, columns=['total_cells'])
cell_numbers_df['active_cells'] = 0
all_region_df = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-26_3to5dpf_cell_positions.pkl")
all_region_df.reset_index(drop=False, inplace=True)
all_region_df.set_index(['age', 'fish_id', 'repeat', 'z_plane', 'cell_name'], inplace=True)
active_cell_regions = all_region_df.loc[np.isin(all_region_df.index.get_level_values('cell_name'),
                                                full_traces_df.index.get_level_values('cell_name').unique())]

for (f, r, z), grouped_df in cell_numbers_df.groupby(level=('fish_id', 'repeat', 'z_plane')):
    cell_identifier = f"f{f}r0{r}t000z00{z}"
    active_cell_map = dict(active_cell_regions.iloc[[cell_identifier in i for i in
                                                     active_cell_regions.index.get_level_values('cell_name')]].groupby(
        'fine_region').size())
    cell_numbers_df.loc[grouped_df.index, 'active_cells'] = grouped_df.index.get_level_values('fine_region').map(
        active_cell_map)
cell_numbers_df['active_cells'] = [int(x) if not np.isnan(x) else 0 for x in cell_numbers_df['active_cells']]
cell_numbers_df_filtered = deepcopy(cell_numbers_df.loc[cell_numbers_df['total_cells'] > 20,])
cell_numbers_df_filtered.reset_index(drop=False, inplace=True)

age_array = np.zeros(len(cell_numbers_df_filtered)).astype(str)
id_to_age = {}
# for overnight imaging first 2/3 repeats get first age last 2 a day later
for k in ['20260305-000', '20260305-001', '20260427-000', '20260427-001',
          ('20260427-002', (0, 1)), ('20260427-003', (0, 1)), ('20260504-004', (0, 1)), ('20260504-005', (0, 1))]:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '3dpf'
for k in ['20260306-000', ('20260427-002', (2, 3, 4)), ('20260427-003', (2, 3)), ('20260504-004', (2, 3)),
          ('20260504-005', (2, 3)), ('20260505-006', (0, 1, 2)), ('20260505-007', (0, 1, 2)),
          ('20260511-008', (0, 1, 2))]:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '4dpf'
for k in ['20260307-001', '20260512-010', ('20260505-006', (3, 4)),
          ('20260505-007', (3, 4)), ('20260511-008', (3, 4)), '20260423-002', '20260601-000', '20260601-001',
          '20260601-002', '20260601-003']:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '5dpf'

for k, v in id_to_age.items():
    fish = k[0]
    reps = k[1]
    mask = np.bitwise_and(cell_numbers_df_filtered['fish_id'] == fish, cell_numbers_df_filtered['repeat'].isin(reps))
    age_array[mask] = v
cell_numbers_df_filtered['age'] = age_array
cell_numbers_df_filtered['fraction'] = cell_numbers_df_filtered['active_cells'] / cell_numbers_df_filtered[
    'total_cells']

cell_numbers_df_filtered.set_index(['age', 'fish_id', 'repeat', 'z_plane', 'fine_region'], inplace=True)

pickle_save_object(cell_numbers_df_filtered,
                   rf'C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\{today}_cell_numbers_activity_fractions.pkl')

# %%
cell_numbers_df_filtered = pickle_load_object(
    r"C:\Users\ag-bahl\Desktop\Max\master_thesis\2p_functional_imaging\dataframes\3to5dpf\2026-06-28_cell_numbers_activity_fractions.pkl")
age_differences = []
for region, region_df in cell_numbers_df_filtered.groupby('fine_region'):
    active_fractions = region_df.groupby('age').median().loc[:, 'fraction']
    if len(active_fractions) == 3:
        age_differences.append([
            region,
            active_fractions['4dpf'] - active_fractions['3dpf'],
            active_fractions['5dpf'] - active_fractions['4dpf'],
            active_fractions['5dpf'] - active_fractions['3dpf'],
        ])
age_activity_diff_df = pd.DataFrame(age_differences, columns=['region', '3to4', '4to5', '3to5'])

# %% Hindbrain small regions
mask = np.array([('Rhombencephalon' in r and not 'Rhombomere' in r and not 'Cerebellum' in r and not 'Neuropil' in r)
                 for r in age_activity_diff_df['region']])
hb_activity_diff_df = age_activity_diff_df.loc[mask]
glob_max = np.round(hb_activity_diff_df.values[:, 1:].max(), 2)
glob_min = np.round(hb_activity_diff_df.values[:, 1:].min(), 2)
for key in age_activity_diff_df.columns:
    if key == 'region': continue
    plot_active_regions_on_brain(
        hb_activity_diff_df, key, brain_region_masks, brain_volume,
        y_crop=(550, 900), x_crop=(138, 498),
        custom_color_range=(glob_min, glob_max), color_map='Spectral_r')

#%% plot traces of most changed regions
most_changing_regions = hb_activity_diff_df.loc[
    hb_activity_diff_df['3to5'] > np.nanmedian(hb_activity_diff_df['3to5']), 'region'].values
cell_subset_df = all_region_df.loc[all_region_df['fine_region'].isin(most_changing_regions),:]
cell_subset_df.reset_index(drop=False, inplace=True)
cell_subset_df.set_index(['age','fish_id', 'repeat', 'z_plane', 'cell_name',], inplace=True)
cell_subset_df['fine_region'] = [s.split(' - ')[1] for s in cell_subset_df['fine_region']]
cell_subset_df.sort_values(by=['age','fine_region'], ascending=True, inplace=True)
fig = group_plotter_stimuli_separated(
    full_traces_df,
    cell_subset_df,
    col_name='fine_region',
    index_level_name='age',
    stim_start=0,
    stim_end=50,
    show_individual_traces=False,
    colors=age_colors,
    color_by_index=True,
    stimuli=['lumi_off_dots_ipsi',],
    use_same_axes_for_all_plots=True,
    display_number_of_cells=True,
    show_stim_names=False,
    show_panel_title=True,
    ylim_pad=.25,
    ylim_percentiles=[20,95],
    x_scale_label='10s',
    x_scale_length=20,
    title=None,
    y_scale_length=.5,
    outer_wspace_hspace=[0,.75]
)
#%% plot changed regions

plot_active_regions_on_brain(
    hb_activity_diff_df.loc[np.isin(hb_activity_diff_df['region'].values, most_changing_regions)],
    '3to5',
    brain_region_masks, brain_volume,
    slices=[(70,90),],
    y_crop=(550, 900), x_crop=(138, 498), show_region_labels=True)

#%% correlation to motion regressors boxplots
correlation_dfs = []
for region, grouped_df in full_traces_df.groupby('fine_region'):
    if region in most_changing_regions:
        corr_df = correlation_to_regressors(
            traces_df=grouped_df,
            regressors=motion_regressors,
            min_correlation=0.1,
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
            stimuli=['lumi_off_dots_ipsi'],
            cell_type_names=['Motion Onset', 'Motion Integrator', 'Slow Motion Integrator'],
            n_jobs=8
        )
        corr_df['region'] = region
        correlation_dfs.append(corr_df)

changing_region_corr_df = pd.concat(correlation_dfs)
changing_region_corr_df['age'] = changing_region_corr_df.index.get_level_values('cell_name').map(cell_to_age)
changing_region_corr_df.set_index(['region', 'age'], append=True, inplace=True)
changing_region_corr_df.dropna(how='any', inplace=True)

for i,(region,reg_df) in enumerate(changing_region_corr_df.groupby('region')):
    short_name = region.split(' - ')[1].replace('.','').replace(' ', '')[:15]
    print(short_name)
    fig = plot_boxplots_corr_values(
        reg_df,
        colors=age_colors,
        n_rows_cols=(1,3),
        color_by_index=True,
        do_stat_test=True,
        print_pvalues=True,
        figsize=(5,1.5),
        title=region,
        hide_labels=False,
        y_offset_value=0.2,
        median_color='red',
        tick_values=[.0,.5,])
    #fig.savefig(plot_dir / f'region_boxplots/{short_name}_corr_boxplot.svg', transparent=True)

#%% print corr values
last_r = None
last_c = None
for (r,c,a),df in changing_region_corr_df.groupby(['region','cell_type', 'age']):
    if last_r != r:
        print(f'\n{r}')
    if last_c != c:
        print(f'\t{c}')
    last_r = r
    last_c = c
    corr_val = np.round(np.nanmedian(df['correlation_value'].values),2)
    print(f'\t\t{a} {corr_val}')



#%% barplot for fractions

age_region_fractions = cell_numbers_df_filtered.groupby(['age','fine_region']).median().loc[:,'fraction']
age_region_fractions = age_region_fractions.reset_index().set_index('age')
mask = np.array([('Rhombencephalon' in r and not 'Rhombomere' in r and not 'Cerebellum' in r and not 'Neuropil' in r)
                 for r in age_region_fractions['fine_region']])
hb_age_fractions = age_region_fractions[mask]
hb_age_fractions['fine_region'] = [s.split(' - ')[1].replace('Anterior Cluster of nV Trigeminal Motorneurons','Ant. Clust. of nV Trigem. Motorn.'
                                                             ) for s in hb_age_fractions['fine_region']]
hb_age_fractions = hb_age_fractions.reset_index().set_index(['fine_region', 'age'])
plot_df = hb_age_fractions['fraction'].unstack('age')
plot_df.dropna(how='any', inplace=True)
plot_df.sort_values(by='5dpf', ascending=False, inplace=True)

ax = plot_df.plot(kind='bar', figsize=(10, 6), color=age_colors)
ax.set_ylabel('Fraction of active cells')
ax.legend(title='', ncol=3, loc='best', frameon=False)
for spine in ax.spines:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_yticks([.25, .5, .75])
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.xlabel(None)
plt.tight_layout()
plt.show()

#%% get stimulus correlation data
datasets = {"hindbrain":{},}
for age,grouped_df in full_traces_df.groupby('age'):
    crosscorr_class = StimulusCrossCorrelation(
        grouped_df,
        stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', 'lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        do_smoothing=True,
        smoothing_window=7,
        smoothing_polyorder=3,
        use_trace_metrics=True,
        n_jobs=1,
    )
    crosscorr_df = crosscorr_class.process_df()
    datasets[age] = crosscorr_df
    hb_cells = grouped_df.index.get_level_values('cell_name')[
        grouped_df.index.get_level_values('region') == 'Rhombencephalon'].unique()
    datasets['hindbrain'][age] = deepcopy(crosscorr_df.loc[hb_cells,
                                          [c for c in crosscorr_df.columns if not 'dots_off' in c]])

hb_df = deepcopy(full_traces_df.xs('Rhombencephalon', level='region'))
hb_df = hb_df.loc[[i in ['lumi_off_dots_ipsi', 'lumi_off_dots_contra'] for i in hb_df.index.get_level_values("stimulus")]]

#%% different clustering parameters
ages = ['3dpf', '4dpf', '5dpf']
metrics = ['euclidean', 'correlation', 'cosine']
min_sample_size = range(3,14,2)
min_cluster_size = range(10,56,5)
max_cluster_size = [None, 100, 200]
selection_method = ['eom', 'leaf']

results = {a:{'full':[], 'hb':[]} for a in ages}
i = 0
for age in ages:
    cluster_class = HdbscanCellClusterer(
        full_traces_df,
        ordered_stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', 'lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        n_jobs=16,
    )

    cluster_class.load_and_scale_data(datasets[age])
    hb_clust_class = HdbscanCellClusterer(
        hb_df,
        ordered_stimuli=['lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        n_jobs=16,
    )
    hb_clust_class.load_and_scale_data(datasets['hindbrain'][age])

    for dist_met,min_sample,min_clust,max_clust,sel_met in product(
            metrics,min_sample_size,min_cluster_size,max_cluster_size,selection_method):

        i += 1
        print(f"\n-------------------------------\n"
              f"\tIteration {i} of 3240"
              f"\n-------------------------------\n")

        print((age,dist_met,min_sample,min_clust,max_clust,sel_met))
        try:
            cluster_class.run_HDBSCAN_clustering(
                parameter_dict={
                    'metric': dist_met,
                    'min_cluster_size': min_clust,
                    'min_samples': min_sample,
                    'max_cluster_size': max_clust,
                    'cluster_selection_method': sel_met,
                    'cluster_selection_epsilon': 0.,
                }
            )
            cluster_class.prune_clusters()
            cluster_class.consolidate_barycenters(max_iterations=10)
            n_clusters = cluster_class.barycenters.shape[0]
        except:
            n_clusters = None
        print(n_clusters)
        results[age]['full'].append((
        (dist_met,min_sample,min_clust,max_clust,sel_met),
            n_clusters
        ))

        try:
            hb_clust_class.run_HDBSCAN_clustering(
                parameter_dict={
                    'metric': dist_met,
                    'min_cluster_size': min_clust,
                    'min_samples': min_sample,
                    'max_cluster_size': max_clust,
                    'cluster_selection_method': sel_met,
                    'cluster_selection_epsilon': 0.,
                }
            )
            hb_clust_class.prune_clusters()
            hb_clust_class.consolidate_barycenters(max_iterations=10)
            n_clusters = hb_clust_class.barycenters.shape[0]
        except:
            n_clusters = None
        print(n_clusters)
        results[age]['hb'].append((
            (dist_met, min_sample, min_clust, max_clust, sel_met),
            n_clusters
        ))
        #save after every 10 steps and clean parallel backend
        if i % 10 == 0:
            pickle_save_object(results, fr'C:\Users\ag-bahl\Desktop\Max\master_thesis\objects\{today}_clustering_1080_parameters_results.pkl')
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        gc.collect()

#%% plot results of different parameters
clustering_results = pickle_load_object(r'C:\Users\ag-bahl\Desktop\Max\master_thesis\objects\2026-07-08_clustering_1080_parameters_results.pkl')
parameters = [t[0] for t in clustering_results['3dpf']['full']]
num_of_clusters = {
    f"{k}_{m}": [t[1] for t in w]
    for k,v in clustering_results.items()
    for m,w in v.items()}
clustering_results_df = pd.DataFrame(num_of_clusters)
tuples = [col.split("_") for col in clustering_results_df.columns]
clustering_results_df.columns = pd.MultiIndex.from_tuples(tuples, names=["age", "type"])


ages = ['3dpf', '4dpf', '5dpf']
regions = ['full', 'hb']
zero_clusters = {r:{} for r in regions}
fig, axes = plt.subplots(1,2, figsize=(6,4))
for region,age in product(regions,ages):
    series = clustering_results_df.loc[:, (age, region)].values.astype(float)
    num_zeros = sum(series == 0)
    zero_clusters[region][age] = num_zeros
    x_sorted = np.sort(series)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax = axes[regions.index(region)]
    ax.plot(x_sorted, y, drawstyle="steps-post", color=age_colors[age], lw=2)
    ax.fill_between(
        x_sorted,
        y,
        0,
        step="post",  # match the ECDF step style
        alpha=0.5,
        color=age_colors[age]
    )
    ax.set_xscale("symlog")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('# non-noise clusters')
    ax.set_ylabel('Cumulative probability')
    ax.set_yticks([0.,1.])
    ax.set_title(region)
plt.show()


fig,ax = plt.subplots()
x = np.array([0,2])
width = 0.4
bars_per_age = []
for i, age in enumerate(ages):
    vals = [zero_clusters[r][age] for r in regions]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=age, facecolor=age_colors[age], edgecolor='black')
    bars_per_age.append(bars)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.sort(list(chain.from_iterable([x + (i - 1) * width for i in range(3)]))))
ax.set_xticklabels(ages + ages)
ax.set_yticks([])
ax.set_ylabel('Number of parameter settings\nthat produced no clusters')
for bars in bars_per_age:
    ax.bar_label(bars, padding=2)
plt.show()

#%% plot for regressors
katja_regressors = pd.read_csv(r"C:\Users\ag-bahl\Desktop\Max\master_thesis\Katjas_model_regressors.tsv", sep="\t",
                               index_col=[0, 1])  # ['Luminance', 'Bright', 'Dark', 'Motion', 'Difference', 'Drive']
regressors = katja_regressors.to_numpy().reshape((6, 6, 120))[
    :, [1, 3, 4, 5]]  # lumi ipsi, lumi contra, dots ipsi, dots contra
pure_lumi_regressors = regressors[[0, 1, 2, 4]]

regressors_flo = np.load(r"C:\Users\ag-bahl\Desktop\Max\master_thesis\Flo_kmeans_regressors.npy") * 100
motion_regressors = regressors_flo[:-1, None, :]  # ['motion_onset', 'motion_integrator_1', 'slow_motion_integrator']

fig = plt.figure()
all_axes = []
outer = GridSpec(nrows=2, ncols=2)
for (i,j),cell_type,arr in zip(product(range(2),range(2)),
                              ['Luminance Integrator', 'Bright', 'Dark', 'Change'],
                              pure_lumi_regressors):
    inner = outer[i,j].subgridspec(nrows=2, ncols=2)
    for (k,l),stim_trace in zip(product(range(2),range(2)),arr):
        ax = fig.add_subplot(inner[k,l])
        all_axes.append(ax)
        ax.plot(stim_trace[20:], color=cell_type_colors[cell_type])
        ax.axvspan(0, 80, color='lightgrey', alpha=0.3)
for ax in all_axes:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim((-.2,1.2))
plt.show()

fig, axes = plt.subplots(1,3, sharey=True, figsize=(9,3))
for ax,cell_type,trace in zip(axes,
                            ['Motion Onset', 'Motion Integrator', 'Slow Motion Integrator'],
                            motion_regressors[:,0,:]):
    ax.plot(trace[20:], color=cell_type_colors[cell_type])
    ax.axvspan(0, 80, color='lightgrey', alpha=0.3)
for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()


#%% create some meaningful clusters for 5, 4, and 3dpf (5)
cluster_class = HdbscanCellClusterer(
        full_traces_df,
        ordered_stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', 'lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        n_jobs=16,
)
cluster_class.load_and_scale_data(datasets['5dpf'])
print(len(datasets['5dpf']))
cluster_class.run_HDBSCAN_clustering(
    metric='cosine',
    min_samples=10,
    min_cluster_size=50,
    max_cluster_size=None,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.1,
)
cluster_class.prune_clusters()
cluster_class.consolidate_barycenters(max_iterations=20)
cluster_class.plot_cluster_traces()
cluster_df5 = cluster_class.cluster_df


#%% (4)
cluster_class.load_and_scale_data(datasets['4dpf'])
print(len(datasets['4dpf']))
cluster_class.run_HDBSCAN_clustering(
    metric='cosine',
    min_samples=8,
    min_cluster_size=40,
    max_cluster_size=None,
    cluster_selection_method='leaf',
    cluster_selection_epsilon=0.,
)
cluster_class.prune_clusters()
cluster_class.consolidate_barycenters(max_iterations=20)
cluster_class.plot_cluster_traces()
cluster_df4 = cluster_class.cluster_df

#%% (3)
cluster_class.load_and_scale_data(datasets['3dpf'])
print(len(datasets['3dpf']))
cluster_class.run_HDBSCAN_clustering(
    metric='cosine',
    min_samples=5,
    min_cluster_size=30,
    max_cluster_size=None,
    cluster_selection_method='leaf',
    cluster_selection_epsilon=0.,
)
cluster_class.prune_clusters()
cluster_class.consolidate_barycenters(max_iterations=20)
cluster_class.plot_cluster_traces()
cluster_class.manually_remove_clusters([1])
cluster_df3 = cluster_class.cluster_df

#%%
colors = plt.get_cmap("Set2").colors
for title, df in zip([str(i + 3) + " dpf" for i in range(3)],[cluster_df3, cluster_df4, cluster_df5]):
    group_plotter_stimuli_separated(
        trace_df=full_traces_df,
        group_df=df,
        col_name='cluster_label_pruned',
        cell_id_index_name='cell_name',
        stimuli=['lumi_ipsi_dots_off', 'lumi_contra_dots_off', 'lumi_off_dots_ipsi', 'lumi_off_dots_contra'],
        max_stim_cols=1,
        stim_start=0,
        stim_end=50,
        show_individual_traces=True,
        show_panel_title=False,
        display_number_of_cells=True,
        show_stim_names=False,
        use_same_axes_for_all_plots=False,
        use_same_axes_for_one_cell_type=True,
        x_scale_label='10s',
        x_scale_length=20,
        y_scale_length=0.1,
        y_scale_unit=r'$\Delta$F/$F_{0}$',
        ylim_pad=.5,
        title=title,
        colors=colors,
        fig_size=(3, 12)
    )