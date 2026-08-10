import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, false_discovery_control
import warnings


def resample_regressors(regressor, reg_stim_start, reg_stim_end,
                        target_pre_stim_len, target_stim_len, target_post_stim_len):
    """
    Resample the regressor to match the target stimulus and post-stimulus lengths.

    Parameters:
    - regressor: The original regressor array.
    - reg_stim_start: The start index of the stimulus period in the original regressor.
    - reg_stim_end: The end index of the stimulus period in the original regressor.
    - target_stim_len: The desired length of the stimulus period.
    - target_post_stim_len: The desired length of the post-stimulus period

    Returns:
    The resampled regressor array with length target_pre_stim_len + target_stim_len + target_post_stim_len.

    """

    if target_pre_stim_len != 0:
        pre_stim = np.interp(
            np.linspace(0, 1, target_pre_stim_len),
            np.linspace(0, 1, reg_stim_start),
            regressor[:reg_stim_start])
    else:
        pre_stim = np.array([])

    if target_stim_len != 0:
        stim = np.interp(
            np.linspace(0, 1, target_stim_len),
            np.linspace(0, 1, reg_stim_end - reg_stim_start),
            regressor[reg_stim_start:reg_stim_end])
    else:
        stim = np.array([])

    if target_post_stim_len != 0:
        post_stim = np.interp(
            np.linspace(0, 1, target_post_stim_len),
            np.linspace(0, 1, regressor.shape[0] - reg_stim_end),
            regressor[reg_stim_end:])
    else:
        post_stim = np.array([])

    return np.concatenate([pre_stim, stim, post_stim])

def compute_correlation_with_regressor_stack(arr, regressors, method='pearson'):
    """
    Compute correlation between a single trace and a stack of regressors.

    Parameters
    ----------
    arr : 1D array
        Single measured trace.
    regressors : 2D array, shape (n_regressors, n_timepoints)
        Stack of regressors to compare against arr.
    method : {'pearson', 'spearman'}
        Correlation method.

    Returns
    -------
    corr : 1D array
        Correlation coefficient for each regressor.
    pvals : 1D array
        P-value for each regressor.
        one-sided p-value for H0: rho <= 0 vs H1: rho > 0
        as anti correlation is not wanted
    """

    out_corr = np.full(regressors.shape[0], np.nan, dtype=float)
    out_pval = np.full(regressors.shape[0], np.nan, dtype=float)

    # if signal range is very small return nan
    if arr.max() - arr.min() < 1e-10:
        return out_corr, out_pval

    arr = (arr - arr.min()) / (arr.max() - arr.min()) # 0 to 1 normalization

    for i, reg in enumerate(regressors):
        if np.std(arr) < 1e-10 or np.std(reg) < 1e-10:
            continue

        if method == 'pearson':
            res = pearsonr(arr, reg, alternative='greater')
        elif method == 'spearman':
            res = spearmanr(arr, reg, alternative='greater')
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        out_corr[i] = res.statistic
        out_pval[i] = res.pvalue

    return out_corr, out_pval

def get_best_cell_type(correlation, threshold, regressor_names):
    # assumes shape of correlation is (n_cells, n_regressors)
    if correlation.shape[1] == 1:
        # only one possible cell type - no runner-up to compare against,
        # so everyone gets that type; cutoff_for_cell_type / FDR filtering downstream
        # still apply the min_correlation and significance thresholds
        top_reg_name = np.full(correlation.shape[0], regressor_names[0])
        top_corr_val = correlation[:, 0]
        return pd.DataFrame({'cell_type': top_reg_name, 'correlation_value': top_corr_val})

    sorted_idx = np.argsort(correlation, axis=1)

    top_reg_name = np.array([regressor_names[i] for i in sorted_idx[:, -1]])
    top_corr_val = np.take_along_axis(correlation, sorted_idx[:, -1][:, None], axis=1)[:, 0]
    # remove all cells without clear type
    mask = (np.diff(correlation[np.arange(correlation.shape[0])[:, None], sorted_idx[:, -2:]],
                   axis=1) > threshold).flatten()
    top_reg_name[~mask] = 'UNCLASSIFIED'
    top_corr_val[~mask] = np.nan
    return pd.DataFrame({'cell_type': top_reg_name, 'correlation_value': top_corr_val})

def cutoff_for_cell_type(grouped_df, q, min_cutoff):
    if grouped_df.name == 'UNCLASSIFIED': return grouped_df
    cutoff = grouped_df['correlation_value'].quantile(q)
    if cutoff < min_cutoff: cutoff = min_cutoff
    grouped_df['passes_cutoff'] = grouped_df['correlation_value'] >= cutoff
    return grouped_df

def get_cell_types_from_corr_df(corr_df, min_corr, diff_th, cutoff_q, level_id):
    #expects corr_df to have shape (n_cells, n_regressors) and index with 'level_id' as one of the levels
    arr = corr_df.to_numpy()
    if len(arr.shape) == 1: #if only one cell type
        arr = arr[:, None]
    regressor_names = list(corr_df.columns)
    cell_names = np.array(corr_df.index.get_level_values(level_id))
    results = get_best_cell_type(arr, diff_th, regressor_names)
    results[level_id]=cell_names
    results.set_index(['cell_type',level_id], inplace=True)
    results['passes_cutoff'] = False
    results = results.groupby('cell_type',group_keys=False).apply(cutoff_for_cell_type, q=cutoff_q, min_cutoff=min_corr)
    for (type, bool), df in results.groupby(['cell_type', 'passes_cutoff']):
        if type == 'UNCLASSIFIED' or not bool: continue
        print(f"{type}, Count: {len(df)}")
    return results

def add_fdr_columns_to_pval_df(pval_df, fdr_alpha):
    """
    For each regressor (column), apply BH FDR correction across cells.
    Adds columns: 'pval_fdr' and 'passes_fdr'.
    """
    for reg in pval_df.columns:
        p_raw = pval_df[reg].to_numpy()

        # skip NaNs in correction
        finite = np.isfinite(p_raw)
        p_corr = np.full_like(p_raw, np.nan)
        if finite.sum() > 0:
            p_fdr = false_discovery_control(p_raw[finite], method='bh')
            p_corr[finite] = p_fdr
        fdr_col = f"{reg}_pval_fdr"
        pass_col = f"{reg}_passes_fdr"

        pval_df[fdr_col] = p_corr
        pval_df[pass_col] = p_corr < fdr_alpha

    return pval_df

def apply_fdr_cutoff(grouped_df, pval_df, valid_names):
    if grouped_df.name not in valid_names: return grouped_df
    cell_type = grouped_df.name
    fdr_col = f"{cell_type}_pval_fdr"
    pass_col = f"{cell_type}_passes_fdr"
    cell_index = grouped_df.index.get_level_values(level=1)
    grouped_df['p_value'] = pval_df.loc[cell_index, cell_type].values
    grouped_df['fdr_p'] = pval_df.loc[cell_index, fdr_col].values
    grouped_df['passes_fdr'] = pval_df.loc[cell_index, pass_col].values
    return grouped_df


def correlation_to_regressors(
        traces_df,
        regressors,
        cell_id_level_name='cell_name',
        min_correlation=None,
        cell_type_difference_threshold=None,
        quantile_cutoff=None,
        fdr_alpha=.05,
        correlation_method='pearson',
        stimulus_resampling_params=None,
        stimuli=None,
        cell_type_names=None,
        n_jobs=-1):
    # standard parameters (very loose)

    if min_correlation is None:
        min_correlation = 0.2
    if cell_type_difference_threshold is None:
        cell_type_difference_threshold = 0.1
    if quantile_cutoff is None:
        quantile_cutoff = 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        traces_df.drop(columns=['reliability'], inplace=True, errors='ignore')

    if stimulus_resampling_params is not None:
        num_regressors, num_stimuli, _ = regressors.shape
        regressors = np.array([
            resample_regressors(r,
                                stimulus_resampling_params['reg_stim_start'],
                                stimulus_resampling_params['reg_stim_end'],
                                stimulus_resampling_params['pre_stim_len'],
                                stimulus_resampling_params['stim_len'],
                                stimulus_resampling_params['post_stim_len'])
            for r in regressors.reshape(num_regressors*num_stimuli, -1)
        ]).reshape(num_regressors, num_stimuli, -1)

    # concatenate down to (type x stimuli in order)
    if len(regressors.shape) > 2:
        regressors = np.concat(regressors.transpose((1, 0, 2)), axis=1)

    if stimuli is None:
        stimuli = ['lumi_ipsi_dots_ipsi', 'lumi_ipsi_dots_off', 'lumi_contra_dots_contra', 'lumi_contra_dots_off',
                   'lumi_off_dots_ipsi', 'lumi_off_dots_contra']

    if cell_type_names is None or len(cell_type_names) != regressors.shape[0]:
        cell_type_names = [f'Cluster{i}' for i in range(regressors.shape[0])]

    filter = np.array([s in stimuli for s in traces_df.index.get_level_values('stimulus')])
    if traces_df.shape[0] == 0:
        raise ValueError('stimuli are not present in the dataframe')
    traces_df = traces_df[filter]

    unique_cells = traces_df.index.get_level_values(cell_id_level_name).unique()
    cell_to_idx = {cell: i for i, cell in enumerate(unique_cells)}
    stim_to_idx = {stim: i for i, stim in enumerate(stimuli)}

    # build array: (num_cells, num_stimuli, num_features)
    num_cells, num_stimuli, num_features = len(unique_cells), len(stimuli), traces_df.shape[1]
    trace_arr = np.full((num_cells, num_stimuli, num_features), np.nan)

    # Fill it
    cell_index_level = traces_df.index.names.index(cell_id_level_name)
    stim_index_level = traces_df.index.names.index('stimulus')

    for index_levels, row in traces_df.iterrows():
        c_idx = cell_to_idx[index_levels[cell_index_level]]
        s_idx = stim_to_idx[index_levels[stim_index_level]]
        trace_arr[c_idx, s_idx, :] = row.values

    # reshape to concatenated stimuli traces
    trace_arr = np.concat(trace_arr.transpose((1, 0, 2)), axis=1)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation_with_regressor_stack)(
            trace, regressors, method=correlation_method
        )
        for trace in tqdm(
            trace_arr[np.array([cell_to_idx[c] for c in unique_cells]), :],
            desc="Computing correlations with regressors"
        )
    )
    # results is a list of (corr_vec, pval_vec)
    corr = np.array([r[0] for r in results])
    pvals = np.array([r[1] for r in results])

    corr_df = pd.DataFrame(corr, columns=cell_type_names)
    corr_df[cell_id_level_name] = unique_cells
    corr_df.set_index(cell_id_level_name, inplace=True)

    cell_types = get_cell_types_from_corr_df(
        corr_df,
        min_corr=min_correlation,
        diff_th=cell_type_difference_threshold,
        cutoff_q=quantile_cutoff,
        level_id=cell_id_level_name)

    pval_df = pd.DataFrame(pvals, columns=cell_type_names)
    pval_df[cell_id_level_name] = unique_cells
    pval_df.set_index(cell_id_level_name, inplace=True)
    pval_df = add_fdr_columns_to_pval_df(pval_df, fdr_alpha=fdr_alpha)

    cell_types = cell_types.groupby('cell_type').apply(apply_fdr_cutoff, pval_df=pval_df, valid_names=cell_type_names)
    cell_types = cell_types[
        (cell_types['passes_cutoff']) & (cell_types['passes_fdr'])
        ].copy()
    # Drop the redundant 'cell_type' column if it exists
    if sum(np.array(cell_types.index.names) == 'cell_type') > 1:
        cell_types = cell_types.droplevel(0)
    cell_types.reset_index(inplace=True) # now cell_type is a column again
    cell_types.set_index(cell_id_level_name, inplace=True)
    output_df = pd.concat(
        [corr_df.reindex(unique_cells), cell_types.reindex(unique_cells)],
        axis=1)
    output_df = output_df.loc[:,['cell_type', 'correlation_value', 'p_value', 'fdr_p']]
    if cell_id_level_name == 'cell_name':
        output_df.reset_index(inplace=True)
        output_df['z_plane'] = [int(s.split('z')[1][:3]) for s in output_df['cell_name']]
        output_df['fish_id'] = [s.split('r')[0][1:] for s in output_df['cell_name']]
        output_df['repeat'] = [int(s.split('r')[1][:2]) for s in output_df['cell_name']]
        output_df['cell_number'] = np.array([int(s.split('c')[1]) for s in output_df['cell_name']]) - 10000
        output_df.set_index(['fish_id', 'repeat', 'z_plane', 'cell_name', 'cell_number'], inplace=True)
    return output_df