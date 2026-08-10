# %%
import numpy as np
import pandas as pd
from two_p_image_analysis_tools \
    import get_segmented_signals_multiindex_df, get_experiment_information
from dtw_trial_processing \
    import dtw_clustering_to_get_response_trials
from pathlib import Path
import warnings
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from datetime import datetime
today = datetime.now().strftime('%Y-%m-%d')
#%%

def process_trials_of_one_cell_one_stim(arr, window, k, threshold, min_fraction_active):
    min_active = int(arr.shape[0] * min_fraction_active)
    f0 = np.nanmedian(arr[:, window[0]:window[1]], axis=1)
    arr_centered = arr - f0[:, None]
    arr_norm = arr_centered / f0[:, None]
    if threshold is None:  # skip activity detection
        return arr_norm, True
    is_active = sum(np.abs(arr_norm[:, window[1]:]).max(axis=1) > threshold) >= min_active # hard df/f0 threshold
    if is_active:
        sigma = np.nanstd(arr_centered[:, window[0]:window[1]])  # std over all zeroed baseline periods
        is_active = sum(np.abs(arr_centered[:,window[1]:]).max(axis=1) > k * sigma) >= min_active
    return arr_norm, is_active


def _process_group(c, s, group_values, i, j, window, k, threshold, min_fraction_active):
    # group_values: numpy array for this (cell, stim)
    processed_trials, is_active = process_trials_of_one_cell_one_stim(
        group_values, window, k, threshold, min_fraction_active
    )
    return i, j, processed_trials, is_active


def normalize_and_detect_activity(df, window, threshold, min_fraction_active, k=1):
    print(f"Normalizing dataframe of shape {df.shape}")
    cell_names = sorted(df.index.get_level_values('cell_name').unique())
    stim_names = sorted(df.index.get_level_values('stimulus').unique())
    cell_idx = {c: i for i, c in enumerate(cell_names)}
    stim_idx = {s: i for i, s in enumerate(stim_names)}

    # create array to apply trail clustering to

    max_no_trials = df.groupby(['cell_name', 'stimulus']).size().max()
    n_cells = len(cell_idx)
    n_stims = len(stim_idx)
    n_time = df.shape[1]
    data_arr = np.full(
        shape=(n_cells, n_stims, max_no_trials, n_time),
        fill_value=np.inf
    )

    # prepare work units
    work = []
    for (c, s), group in df.groupby(['cell_name', 'stimulus'], sort=True):
        i, j = cell_idx[c], stim_idx[s]
        work.append((c, s, group.to_numpy(), i, j))

    # parallel processing of all groups
    results = Parallel(n_jobs=-1)(
        delayed(_process_group)(c, s, g_values, i, j, window, k, threshold, min_fraction_active)
        for (c, s, g_values, i, j) in tqdm(work)
    )

    active_indices = []
    for i, j, processed_trials, is_active in results:
        n_trials = processed_trials.shape[0]
        data_arr[i, j, :n_trials, :] = processed_trials
        if is_active:
            active_indices.append((i, j))

    # rebuild df
    index = pd.MultiIndex.from_product(
        [cell_names, stim_names, np.arange(max_no_trials)],
        names=["cell_name", "stimulus", "trial"]
    )
    output_arr = deepcopy(data_arr)
    C, S, T, t = data_arr.shape
    norm_df = pd.DataFrame(
        data_arr.reshape(C * S * T, t),
        index=index
    )
    # clean extra rows
    norm_df.replace(np.inf, np.nan, inplace=True)
    norm_df.dropna(how='all', inplace=True)
    norm_df.sort_index(inplace=True)

    active_indices = np.array(active_indices)
    # filter for at least two appearances of a cell (so active in at least 2 stimuli)
    values, counts = np.unique(active_indices[:, 0], return_counts=True)
    valid_values = values[counts >= 2]
    mask = np.isin(active_indices[:, 0], valid_values)
    active_indices = active_indices[mask]

    return norm_df, output_arr, active_indices


def cleaning_and_trial_clustering(arr, window):
    # removes nan introduced earlier
    arr_clean = arr[~np.isinf(arr)]
    arr_clean = arr_clean.reshape((arr_clean.shape[0] // arr.shape[1], arr.shape[1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_signal = dtw_clustering_to_get_response_trials(
            arr_clean,
            window=window,
            n_clusters=2,
            metric='dtw',
            metric_parameters={
                "global_constraint": "sakoe_chiba",
                "sakoe_chiba_radius": 4},
            max_iter_barycenter=10,
            random_state=42,
            n_init=5)

    return result_signal


def run_trial_clustering(df, data_arr, indices=None, samples=None, output_length=None, window=None, n_jobs=-1):
    if output_length is None:
        output_length = data_arr.shape[-1] - window[0] + 1
    if indices is None:
        cells = np.arange(len(df.index.get_level_values("cell_name").unique()))
    else:
        cells = np.unique(indices[:, 0])
    if samples is not None:
        cells = np.random.choice(cells, samples, replace=False)
    no_stim = len(df.index.get_level_values("stimulus").unique())
    work_indices = np.array(list(product(cells, range(no_stim))))
    total = len(cells) * no_stim
    print(f"Running trial clustering for {total} samples")
    result_arr = np.full(
        (data_arr.shape[0], data_arr.shape[1], output_length),
        np.nan,
    )
    # run parallel processing as generator to reduce memory overhead
    parallel = Parallel(
        n_jobs=n_jobs,
        timeout=1e9,
        backend="loky",
        return_as="generator",
    )
    # results need to be filled in on the fly
    for (i, j), res in zip(
            work_indices,
            parallel(
                delayed(cleaning_and_trial_clustering)(data_arr[i, j], window=window)
                for i, j in tqdm(work_indices, total=total)
            ),
    ):
        result_arr[i, j] = res

    # recreate index (all values should be sorted)
    cell_index = df.index.get_level_values("cell_name").unique()
    stim_index = df.index.get_level_values("stimulus").unique()
    mux = pd.MultiIndex.from_product([cell_index, stim_index], names=["cell_name", "stimulus"])

    clustered_df = pd.DataFrame(
        result_arr.reshape(len(cell_index) * len(stim_index), result_arr.shape[-1]),
        index=mux
    )
    # remove rows which were not processed
    clustered_df = clustered_df.dropna(how='all')
    clustered_df.sort_index(inplace=True)

    return clustered_df


def main(traces_df, norm_window, k_stds, activity_threshold, min_fraction_active, region_df=None, n_jobs=-1):
    # get trial clustered traces, threshold=None for all cells
    norm_df, data_arr, active_indices = normalize_and_detect_activity(traces_df, window=norm_window, k=k_stds, threshold=activity_threshold, min_fraction_active=min_fraction_active)
    clustered_df = run_trial_clustering(norm_df, data_arr, indices=active_indices, window=(norm_window[1], None), n_jobs=n_jobs)
    clustered_df.rename(columns={clustered_df.columns[-1]: 'reliability'}, inplace=True) #meaning what percent of trials where in the response cluster

    if region_df is not None:
        coarse_region_df = region_df.loc[:, ['Diencephalon -', 'Mesencephalon -', 'Rhombencephalon -', 'Telencephalon -']]
        coarse_region_df.columns = ['Diencephalon', 'Mesencephalon', 'Rhombencephalon', 'Telencephalon']
        coarse_region_series = (
            coarse_region_df[['Diencephalon', 'Mesencephalon', 'Rhombencephalon', 'Telencephalon']]
            .idxmax(axis=1).
            where(coarse_region_df[['Diencephalon', 'Mesencephalon', 'Rhombencephalon', 'Telencephalon']].any(axis=1), np.nan)
        )
        region_df.drop(columns=['Diencephalon -', 'Mesencephalon -', 'Rhombencephalon -', 'Telencephalon -'], inplace=True)
        fine_region_series = (region_df.iloc[:,3:].idxmax(axis=1).where(region_df.iloc[:,3:].any(axis=1), np.nan))

        clustered_df['region'] = clustered_df.index.get_level_values('cell_name').map(coarse_region_series)
        clustered_df['fine_region'] = clustered_df.index.get_level_values('cell_name').map(fine_region_series)

    clustered_df.reset_index(inplace=True)
    clustered_df['z_plane'] = [int(s.split('z')[1][:3]) for s in clustered_df['cell_name']]
    clustered_df['fish_id'] = [s.split('r')[0][1:] for s in clustered_df['cell_name']]
    clustered_df['repeat'] = [int(s.split('r')[1][:2]) for s in clustered_df['cell_name']]
    clustered_df['cell_number'] = np.array([int(s.split('c')[1]) for s in clustered_df['cell_name']]) - 10000

    if 'region' in clustered_df.columns:
        clustered_df.set_index(
            ['fish_id', 'region', 'fine_region', 'repeat', 'z_plane', 'cell_number', 'cell_name', 'stimulus'],
            inplace=True)
    else:
        clustered_df.set_index(
        ['fish_id', 'repeat', 'z_plane', 'cell_number', 'cell_name', 'stimulus']
            , inplace=True)

    return clustered_df

#%%
if __name__ == "__main__":
    base_dir = Path(r'Y:\M11 2P microscopes\Max W\master_thesis_imaging\WARP_stimulus\functional')
    data = {'WARP_stimulus':[], '2P_dots_lumi_4combinations':[]}
    for file in base_dir.iterdir():
        traces, regions = get_segmented_signals_multiindex_df(
            file,
            do_flipping_based_on_hemisphere=True,
            do_segmentaion_shape_cleaning=True,
            return_brain_region_assignments=True,
            registration_type="ZBRAIN"
        )
        info = get_experiment_information(file)
        stimulus = next(file.glob('*py')).name.replace('.py','')
        plane = info['fish_comment'].split('#')[1][0]
        t_idx = traces.index.names
        traces.reset_index(inplace=True, drop=False)
        traces['cell_name'] = [s.replace('z000', f'z{plane.zfill(3)}') for s in traces['cell_name']]
        traces.set_index(t_idx, inplace=True)
        r_idx = regions.index.names
        regions.reset_index(inplace=True, drop=False)
        regions['cell_name'] = [s.replace('z000', f'z{plane.zfill(3)}') for s in regions['cell_name']]
        regions.set_index(r_idx, inplace=True)
        data[stimulus].append([traces, regions])
    traces = {}
    regions = {}
    for k, v in data.items():
        traces[k] = pd.concat([d[0] for d in v])
        regions[k] = pd.concat([d[1] for d in v])

    for stim_type in ['2P_dots_lumi_4combinations',]:
        if stim_type == '2P_dots_lumi_4combinations':
            norm_window = (4, 10)
            activity_threshold = .3
            stim_name = 'dots-lumi-stimulus'
        else:
            norm_window = (2,12)
            activity_threshold = .2
            stim_name = 'WARP-stimulus'

        clustered_df = main(
            traces_df=traces[stim_type],
            norm_window=norm_window,  # what part of the stimulus to use as baseline period
            k_stds=6,  # k x stds peak height to be deemed active
            activity_threshold=activity_threshold, # min dF/F0 val to be deemed active if this is None, no activity detection will be performed and all cells will be clustered
            min_fraction_active=0.3,  # fraction of trials that has to pass activity check for the cell to be deemed active
            n_jobs=16,
            region_df=regions[stim_type],
        )
        clustered_df.to_csv(
            fr".\data\WARP\{today}_trial-clustering_{stim_name}_k6-th{str(activity_threshold).replace('.','')}-frac03.tsv",
            sep='\t',
        )

    base_dir = Path(r'Y:\M11 2P microscopes\Max W\master_thesis_imaging\3to5dfp\functional')
    for file in base_dir.iterdir():
        traces, regions = get_segmented_signals_multiindex_df(
            file,
            do_flipping_based_on_hemisphere=True,
            do_segmentaion_shape_cleaning=True,
            return_brain_region_assignments=True,
            registration_type="ZBRAIN"
        )
        tag = file.name[5:10].replace('-','') + "_fish" + file.name.split('fish')[1][:3]
        print(tag)
        clustered_df = main(
            traces_df=traces,
            norm_window=(4,10),  # what part of the stimulus to use as baseline period
            k_stds=6,  # k x stds peak height to be deemed active
            activity_threshold=.3,
            # min dF/F0 val to be deemed active if this is None, no activity detection will be performed and all cells will be clustered
            min_fraction_active=0.3,
            # fraction of trials that has to pass activity check for the cell to be deemed active
            n_jobs=-1,
            region_df=regions,
        )
        clustered_df.to_csv(
            fr".\data\3to5dpf\{today}_trial-clustering_{tag}_k6-th03-frac03.tsv",
            sep='\t',
        )