import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import copy, deepcopy
from tqdm import tqdm
import warnings
from itertools import product
from tslearn.barycenters import softdtw_barycenter
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from umap import UMAP
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp, false_discovery_control
import matplotlib.pyplot as plt
from plotting_functions import group_plotter_stimuli_separated, get_spaced_colors
from general_util_functions import split_array_along_axis

#%%
def _is_valid_ts(ts):
    return len(ts) > 0 and not np.all(np.isnan(ts))

def _clean_ts(ts):
    # remove NaNs (simple strategy)
    ts = ts[~np.isnan(ts)]
    return ts

class StimulusCrossCorrelation:
    def __init__(
            self,
            full_df,
            stimuli,
            do_smoothing=False,
            smoothing_window=5,
            smoothing_polyorder=3,
            activity_threshold=None,
            n_trace_splits=None,
            index_name="cell_name",
            use_trace_metrics=True,
            n_jobs=1
            ):

        self.full_df = full_df
        self.trace_arr = None
        self.crosscorr_vector_array = None
        self.unique_cells = None
        self.cell_index_lookup = None
        self.index_name = index_name
        self.n_splits = n_trace_splits if isinstance(n_trace_splits, int) else 1
        self.split_traces = None
        self.stimuli = stimuli
        self.n_stimuli = len(self.stimuli)
        self.split_stim_names = [f'{s}_split{i}' for s,i in product(self.stimuli, range(self.n_splits))]
        self.do_smoothing = do_smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.threshold = activity_threshold

        if self.threshold is None:
            self.threshold = 0.

        self.n_jobs = n_jobs

        # cell metrics
        self.cell_id = None
        self.cell_reliability = None
        self.cell_arr = None
        self.cell_smoothed_arr = None
        self.cell_90p = None
        self.cell_return_empty_output = False
        self.cell_first_crossing = None
        self.cell_duration = None
        self.cell_activity_arr = None
        self.cell_norm_correlations = None
        self.cell_output = None
        self.result = []
        self.use_trace_metrics = use_trace_metrics
        
        # adjust output array if cell metrics are used
        if not self.use_trace_metrics:
            added_columns = []
        else:
            added_columns = [t + s for t in
                             ['90th_percentile_', 't_to_90p_', 't_above_70p_after_frist_crossing_', 'reliability_']
                             for s in self.stimuli]
        # stimuli^2 values for cross correlation + cell metic values in output array
        if self.n_splits > 1:
            n = self.n_splits * self.n_stimuli
        else:
            n = self.n_stimuli
        self.output_colnames = [None] * ((n * (n - 1)) // 2) + added_columns
        self.output_df = None

    def get_trace_array(self):

        self.unique_cells = sorted(self.full_df.index.get_level_values('cell_name').unique())

        # Mappings to index back
        self.cell_index_lookup = {cell: i for i, cell in enumerate(self.unique_cells)}
        stim_to_idx = {stim: i for i, stim in enumerate(self.stimuli)}

        # Build array: (num_cells, num_stimuli, num_features)
        num_cells, num_stimuli, num_features = len(self.unique_cells), len(self.stimuli), self.full_df.shape[1]
        self.trace_arr = np.full((num_cells, num_stimuli, num_features), np.nan)

        # Fill it
        index_level_cell = self.full_df.index.names.index('cell_name')
        index_level_stim = self.full_df.index.names.index('stimulus')
        for idx_tuple, row in self.full_df.iterrows():
            cell = idx_tuple[index_level_cell]
            stim = idx_tuple[index_level_stim]
            if stim not in self.stimuli: continue
            c_idx = self.cell_index_lookup[cell]
            s_idx = stim_to_idx[stim]
            self.trace_arr[c_idx, s_idx, :] = row.values

        # See if there is a reliability column else reliability is -1
        try:
            reliability_index = self.full_df.columns.get_loc("reliability")
            self.all_cell_reliability = self.trace_arr[:, :, reliability_index]
            self.trace_arr = np.delete(self.trace_arr, reliability_index, axis=2)
        except:
            self.output_colnames = np.array(
                [colname for colname in self.output_colnames if not 'reliability' in str(colname)])
            self.all_cell_reliability = np.zeros(self.trace_arr.shape[0])
            self.all_cell_reliability.fill(-1)

        if self.n_splits > 1:
            # trace arr has shape cells,stimuli,time
            # split it into cell,n*stimuli,time/n
            cells, stim, _ = self.trace_arr.shape
            self.split_traces = split_array_along_axis(
                self.trace_arr, self.n_splits, axis=2).reshape(
                (cells, stim * self.n_splits, -1))  # function adds a dimension, so collapse it again


    @staticmethod
    def corrcoef_safe_batch_einsum(A, default=np.nan):
        """
        Compute safe correlation matrices for a batch of square matrices A.

        A: shape (batch, n, n)
        Returns: shape (batch, n, n)
        """
        A = np.asarray(A, dtype=float)
        batch, n, _ = A.shape

        # center each row
        mean = A.mean(axis=2, keepdims=True)
        Xm = A - mean

        # row norms (std)
        norms = np.linalg.norm(Xm, axis=2, keepdims=True)  # shape (batch, n, 1)
        denom = norms @ norms.transpose(0, 2, 1)  # shape (batch, n, n)
        denom[denom == 0] = 1e-8

        # covariance / numerator
        cov = np.einsum('bij,bkj->bik', Xm, Xm)  # batch matrix multiplication

        # safe division
        corr = np.divide(cov, denom, out=np.full_like(cov, default), where=denom != 0)

        # force diagonal = 1
        diag_idx = np.arange(n)
        corr[:, diag_idx, diag_idx] = 1.0

        return corr

    def compute_cell_signal_metrics(self):
        """
        finds the time to reach 90% of max response and time to return to under 70%
        for each stimulus
        """
        self.cell_90p = np.percentile(self.cell_arr, q=90, axis=1)
        if np.all(np.abs(self.cell_90p) < self.threshold):
            # if the cell is not active at all return empty output
            self.cell_return_empty_output = True
            return
        above_90p = self.cell_arr >= self.cell_90p[:, None]
        self.cell_first_crossing = np.argmax(above_90p, axis=1)
        below_70p = self.cell_arr < np.percentile(self.cell_arr, q=70, axis=1, keepdims=True)
        drops = below_70p & (np.arange(self.cell_arr.shape[1]) >= self.cell_first_crossing[:, None])
        first_drop = np.where(
            drops.any(axis=1),
            drops.argmax(axis=1),
            self.cell_arr.shape[1]
        ) - 1
        self.cell_duration = first_drop - self.cell_first_crossing

    def create_cell_output(self):
        if self.cell_return_empty_output:
            self.cell_output = np.full((len(self.output_colnames),), np.nan)

        else:
            if self.use_trace_metrics:
                trace_metrics = [
                    self.cell_90p,
                    self.cell_first_crossing,
                    self.cell_duration,
                    self.cell_reliability
                ]
            else:
                trace_metrics = []

            output_list = trace_metrics

            if any([x is None for x in output_list]):
                raise ValueError("Something went wrong!\nSome cell metrics have not been computed yet.")

            # remove reliability output if it wasn't found (reliability == -1)
            if self.use_trace_metrics and np.all(self.cell_reliability == -1):
                _ = output_list.pop(-1)


            self.cell_output = np.concatenate(
                [self.crosscorr_vector_array[self.cell_index_lookup[self.cell_id]]] + output_list
            )

        self.cell_activity_arr = None
        self.cell_norm_correlations = None
        self.cell_90p = None
        self.cell_first_crossing = None
        self.cell_duration = None
        self.cell_reliability = None

    def _process_single_cell(self, cell_id):
        # create a shallow copy so parallel workers don't touch shared state
        obj = copy(self)

        obj.cell_id = cell_id

        if obj.use_trace_metrics:
            obj.cell_reliability = obj.all_cell_reliability[obj.cell_index_lookup[obj.cell_id]]

        obj.cell_arr = obj.trace_arr[obj.cell_index_lookup[obj.cell_id]]

        if np.isnan(obj.cell_arr).any():
            obj.cell_return_empty_output = True
        else:
            obj.compute_cell_signal_metrics()

        obj.create_cell_output()

        df = pd.DataFrame(
            obj.cell_output[None, :],
            index=[obj.cell_id],
            columns=obj.output_colnames
        )
        return df

    def process_df(self):
        self.get_trace_array()
        if self.do_smoothing:
            self.trace_arr = savgol_filter(
                self.trace_arr,
                window_length=self.smoothing_window,  # must be odd
                polyorder=self.smoothing_polyorder,  # usually 2 or 3
                axis=2
            )
        if self.split_traces is not None:
            self.crosscorr_vector_array = self.corrcoef_safe_batch_einsum(self.split_traces)
        else:
            self.crosscorr_vector_array = self.corrcoef_safe_batch_einsum(self.trace_arr)
        tri = np.triu_indices(self.crosscorr_vector_array.shape[-1],1) # take upper triangle above diagonal
        self.crosscorr_vector_array = self.crosscorr_vector_array[:, tri[0], tri[1]]

        self.output_colnames = np.array(self.output_colnames)
        self.output_colnames[self.output_colnames == None] = [f"({self.split_stim_names[i]} x {self.split_stim_names[j]})" for i, j in
                                                              zip(tri[0], tri[1])]
        if self.n_jobs == 1:
            results = [self._process_single_cell(cell_id) for cell_id in self.cell_index_lookup.keys()]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                            delayed(self._process_single_cell)(cell_id)
                            for cell_id in tqdm(list(self.cell_index_lookup.keys()), desc="Processing cells")
                        )

        self.output_df = pd.concat(results, axis=0)
        return self.output_df

class HdbscanCellClusterer:
    def __init__(
            self,
            trace_df,
            ordered_stimuli,
            cell_id_column='cell_name',
            columns_to_drop=None,
            plotting_colors=None,
            n_jobs=-1
    ):
        self.full_traces_df = deepcopy(trace_df.loc[[s in ordered_stimuli for s in trace_df.index.get_level_values('stimulus')]])
        trace_cols = [c for c in trace_df.columns if c.isdigit()]

        self.trace_df = (
            trace_df.reset_index()
            .loc[lambda d: np.isin(d['stimulus'], ordered_stimuli), :]
            .groupby([cell_id_column, 'stimulus'], sort=False)[trace_cols]
            .mean()  # collapse repeats -> one trace per cell+stimulus
            .reset_index()
            .set_index(cell_id_column)
            .sort_index()
        )

        if columns_to_drop is None: #drop everything besides cell_name and the actual trace
            columns_to_drop = [s for s in self.full_traces_df.index.names + self.full_traces_df.columns.to_list() if (s!='cell_name' and not s.isdigit())]
        self.trace_df = deepcopy(trace_df.reset_index())
        self.trace_df = deepcopy(self.trace_df.loc[np.isin(self.trace_df['stimulus'],ordered_stimuli),:]
                                 .set_index(cell_id_column)
                                 .drop(columns = columns_to_drop)
                                 .sort_index()
                     )
        self.stimuli = ordered_stimuli
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.data_full = None
        self.data = None
        self.cell_names = None
        self.clustering_parameters = None
        self.model = None
        self.n_labels = None
        self.labels = None
        self.cluster_df = None
        self.umap = None
        self.barycenters = None
        self.smoothing_window = (self.trace_df.shape[1] // 8) * 2 + 1 # basically one forth of the trace length but make it odd
        self.plotting_colors = plotting_colors
        self.stim_start = self.stim_end = None

    def load_and_scale_data(self, data):
        self.data_full = data
        self.data = self.scaler.fit_transform(data)
        self.cell_names = np.array(data.index)

    def run_HDBSCAN_clustering(self, metric=None, min_cluster_size=None, min_samples=None, max_cluster_size=None, cluster_selection_method=None,
                        cluster_selection_epsilon=0.0, parameter_dict=None):
        print('running HDBSCAN clustering')
        if parameter_dict is not None:
            self.clustering_parameters = parameter_dict
        else:
            self.clustering_parameters = {
                'metric': metric,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'max_cluster_size': max_cluster_size,
                'cluster_selection_method': cluster_selection_method,
                'cluster_selection_epsilon': cluster_selection_epsilon,
            }

        self.model = HDBSCAN(
            metric=self.clustering_parameters['metric'],
            min_cluster_size=self.clustering_parameters['min_cluster_size'],
            min_samples=self.clustering_parameters['min_samples'],
            max_cluster_size=self.clustering_parameters['max_cluster_size'],
            cluster_selection_method=self.clustering_parameters['cluster_selection_method'],
            cluster_selection_epsilon=self.clustering_parameters['cluster_selection_epsilon'],
            n_jobs=self.n_jobs,
            copy=False
        )
        self.model.fit(self.data)
        self.n_labels = self.model.labels_.max() + 1
        self.cluster_df = pd.DataFrame({'cell_name': self.cell_names, 'cluster_label': self.model.labels_, 'cluster_label_pruned': self.model.labels_}).set_index('cell_name')
        self.cluster_df['cluster_label_pruned'] = self.cluster_df['cluster_label_pruned'].astype(str)
        self.labels = np.sort(self.cluster_df['cluster_label_pruned'].unique())[1:] # remove -1 label
        # reset colors because there might be a different number of clusters
        self._get_colors()
        print(f'done - number of non-noise clusters: {self.n_labels}')

    def _calculate_barycenters(self):

        def _compute_one(grouped_df):
            cluster_traces = trace_df.loc[grouped_df.index].to_numpy()
            reshaped = cluster_traces.reshape(
                (cluster_traces.shape[0] // n_stimuli, n_stimuli, cluster_traces.shape[1])
            )
            return softdtw_barycenter(reshaped, gamma=1.0, max_iter=10)

        if self.n_labels < 10:
            barycenters = []
            for cluster_id, grouped_df in self.cluster_df.groupby('cluster_label_pruned'):
                if cluster_id == '-1': continue
                cluster_traces = self.trace_df.loc[grouped_df.index].to_numpy()
                barycenters.append(softdtw_barycenter(
                    cluster_traces.reshape(
                        (cluster_traces.shape[0] // len(self.stimuli), len(self.stimuli), cluster_traces.shape[1])),
                    gamma=1.0, max_iter=20))
        else:
            groups = [
                (cluster_id, grouped_df)
                for cluster_id, grouped_df in self.cluster_df.groupby('cluster_label_pruned')
                if cluster_id != '-1'
            ]

            n_stimuli = len(self.stimuli)
            trace_df = self.trace_df  # local ref, avoids re-pickling `self` repeatedly

            barycenters = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_one)(grouped_df) for _, grouped_df in groups
            )

        return np.array(barycenters)

    def prune_clusters(self, verbose=True, threshold_quantile=.1, recursive=False):

        """
        Removes clusters that are noise (setting their label to -1 in cluster_df)

        First filter: Clusters are flat across all stimuli
          Use absolute peak - std
          Threshold is the 10th percentile of all cluster traces 90th percentile
          Remove cluster if the peaks of the cluster median is below the threshold across all stimuli.

        Second filter: Clusters that have huge signal spikes up and down
          T-test if interquartile range of cluster is higher than all clusters for each stimulus.
          If median p-value across all stimuli is < .05 cluster is removed.
        """

        non_noise_cells = np.array(self.cluster_df.index[self.cluster_df['cluster_label_pruned'] != '-1'])
        all_non_noise_traces = self.trace_df.loc[non_noise_cells].to_numpy()
        if all_non_noise_traces.shape[0] == 0:
            if verbose: print("No non-noise clusters found, skipping pruning.")
            return
        peak_threshold = np.quantile(np.quantile(np.abs(all_non_noise_traces), q=.9, axis=1), q=threshold_quantile, axis=0)
        interquartile_ranges = []
        remove_cluster = []
        for label,grouped_df in self.cluster_df.groupby('cluster_label_pruned'):
            if label == '-1':
                continue
            trace_subset = self.trace_df.loc[grouped_df.index].to_numpy()
            trace_subset = trace_subset.reshape((-1, len(self.stimuli), trace_subset.shape[1]))

            quartiles = np.quantile(trace_subset, q=(.25, .5, .75), axis=0)
            interquartile_ranges.append(np.subtract(quartiles[2], quartiles[0]).max(axis=1)) #maximum interquartile range for each stimulus

            peak_values = np.quantile(np.abs(quartiles[1]), .9, axis=1) #90th percentile absolute median signal value for each stimulus
            remove_cluster.append((peak_values < peak_threshold).all())

        interquartile_ranges = np.array(interquartile_ranges)
        remove_cluster = np.concat([
            np.array(remove_cluster)[:, None],
            np.repeat(np.False_, len(remove_cluster))[:, None]],
            axis=1)
        if len(interquartile_ranges) > 1:
            for i,iq_ranges in enumerate(interquartile_ranges):
                x = np.concatenate((interquartile_ranges[:i], interquartile_ranges[i + 1:]))
                _, p_vals = ttest_1samp(x, iq_ranges, alternative='less')
                p_vals = false_discovery_control(p_vals)
                remove_cluster[i, 1] = (p_vals < .05).all()

        clusters_to_remove = np.array([(i,remove_cluster[i].argmax()) for i in range(len(remove_cluster)) if remove_cluster[i].any()])

        if len(clusters_to_remove) > 0:
            if verbose:
                for (cluster_no, cause) in clusters_to_remove:
                    if cause == 0:
                        print(f"removing cluster {self.labels[cluster_no]} - low activity across all stimuli")
                    else:
                        print(f"removing cluster {self.labels[cluster_no]} - high variance across cell traces")
            self.cluster_df.loc[[str(l) in self.labels[clusters_to_remove[:,0]] for l in self.cluster_df['cluster_label_pruned']], 'cluster_label_pruned'] = '-1'
        else:
            #break recursive loop
            return
        self.labels = np.sort(self.cluster_df['cluster_label_pruned'].unique())[1:]  # remove -1 label
        self.n_labels = len(self.labels)
        if recursive:
            self.prune_clusters(threshold_quantile=threshold_quantile, verbose=verbose, recursive=recursive)

    def manually_remove_clusters(self, cluster_labels):
        print(f'\nremoving cluster {cluster_labels}')
        if type(cluster_labels) == str or type(cluster_labels) == int:
            cluster_labels = [cluster_labels]
        filter = self.cluster_df['cluster_label'].isin(cluster_labels)
        if filter.sum() == 0:
            filter = self.cluster_df['cluster_label_pruned'].isin(cluster_labels)
            if filter.sum == 0:
                return
        self.cluster_df.loc[filter, 'cluster_label_pruned'] = '-1'
        self.labels = np.sort(self.cluster_df['cluster_label_pruned'].unique())[1:]  # remove -1 label

    def refine_cluster(self, cluster_label):
        cluster_label = str(cluster_label)
        print(f'\nrefining cluster {cluster_label}')
        X = self.data_full.loc[
            self.cluster_df.index[self.cluster_df['cluster_label_pruned'] == cluster_label]]
        X = self.scaler.fit_transform(X)
        model = HDBSCAN(
            metric=self.clustering_parameters['metric'],
            min_cluster_size=int(np.round(X.shape[0] * .05)),
            min_samples=3,
            max_cluster_size=None,
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.0,
            n_jobs=self.n_jobs,
            copy=False
        )
        model.fit(X)
        subcluster_df = pd.DataFrame(
            {'cell_name': self.cluster_df.index[self.cluster_df['cluster_label_pruned'] == cluster_label],
             'cluster_label': model.labels_})
        subcluster_df = subcluster_df.set_index('cell_name')
        for label, grouped_df in subcluster_df.groupby('cluster_label'):
            if label == -1:
                replacement = '-1'
            else:
                replacement = str(cluster_label) + "." + str(label)
            self.cluster_df.loc[grouped_df.index, 'cluster_label_pruned'] = replacement
        self.labels = np.sort(self.cluster_df['cluster_label_pruned'].unique())[1:]  # remove -1 label
        self._get_colors()
        print(f'\trefinded cluster {cluster_label} into {max(model.labels_)} subclusters')

    def calculate_umap(self, data=None):
        if data is None:
            data = self.data
            self.cells_umap = self.cell_names
        else:
            data = self.scaler.fit_transform(data)
            self.cells_umap = np.array(data.index)

        self.umap = UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.01,
            n_jobs=-1,
            spread=5.0,
            metric='cosine',
        ).fit_transform(data)

    def _get_colors(self):
        colors = get_spaced_colors(len(self.labels))
        self.plotting_colors = {x: colors[i] for i, x in enumerate(self.labels)}

    def plot_cluster_scatter(self):
        if self.plotting_colors is None: self._get_colors()
        if self.umap is None: self.calculate_umap()
        umap_clusters_mask = [cell in self.cluster_df.index[self.cluster_df['cluster_label_pruned'] != '-1'] for cell in self.cells_umap]
        umap_noise_mask = [cell in self.cluster_df.index[self.cluster_df['cluster_label_pruned'] == '-1'] for cell in self.cells_umap]
        umap_clusters = self.umap[umap_clusters_mask]
        umap_noise = self.umap[umap_noise_mask]

        fig, ax = plt.subplots()
        ax.scatter(umap_noise[:, 0], umap_noise[:, 1], c='gray', alpha=0.2)  # noise umap
        ax.scatter(umap_clusters[:, 0], umap_clusters[:, 1], c=[self.plotting_colors[label] for label in self.cluster_df[self.cluster_df['cluster_label_pruned'] != '-1']['cluster_label_pruned']], )
        ax.set_title(f'{len(self.cluster_df['cluster_label_pruned'].unique())-1} clusters, {umap_clusters.shape[0]} of {self.cluster_df.shape[0]} cells')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def plot_cluster_traces(self,
                            stimuli=None, stim_start=None, stim_end=None, show_stim_name=True,
                            show_cluster_name=True,
                            show_scale_bar=True, show_all_cells=True):
        if stimuli is None:
            stimuli = self.stimuli
        if stim_start is not None:
            if self.stim_start is None:
                self.stim_start = stim_start
        if stim_end is not None:
            if self.stim_end is None:
                self.stim_end = stim_end

        ylimits = (2,98) if show_all_cells else (15,90)

        if len(self.labels) == 0:
            print('no clusters')
            return

        if self.plotting_colors is None:
            self._get_colors()
        group_plotter_stimuli_separated(
            trace_df=self.full_traces_df,
            group_df=self.cluster_df,
            col_name='cluster_label_pruned',
            colors=self.plotting_colors,
            stimuli=stimuli,
            stim_start=self.stim_start,
            stim_end=self.stim_end,
            show_stim_names=show_stim_name,
            show_panel_title=show_cluster_name,
            show_scale_bar=show_scale_bar,
            show_individual_traces=show_all_cells,
            use_same_axes_for_one_cell_type=True,
            use_same_axes_for_all_plots=False,
            y_scale_length=.1,
            ylim_percentiles=ylimits,
            ylim_pad=0.3,

        )

    def _compare_barycenters_recursively(self,iteration,hard_threshold=.9):
        print(f'\n\titeration {iteration}')
        if len(self.labels) <= 1:
            print('\tstopping consolidation because there is only one cluster left')
            return
        elif iteration >= self.max_iterations_consolidation:
            print('\tstopping consolidation because max iterations were reached')
            return # stop recursion after max iterations
        if self.consolidated_barycenters is None:
            barycenters = self.barycenters
        else:
            barycenters = self.consolidated_barycenters

        smooth_barycenters = savgol_filter(barycenters, window_length=self.smoothing_window, polyorder=3, axis=2)
        smooth_barycenters = smooth_barycenters - smooth_barycenters[:, :, 0][
            :, :, None]  # put all starting points at 0
        concat_barycenters = np.concat(smooth_barycenters.transpose(1, 0, 2), axis=1)
        arr_min = concat_barycenters.min(axis=1)
        arr_max = concat_barycenters.max(axis=1)
        concat_barycenters = (concat_barycenters - arr_min[:, None]) / (arr_max - arr_min)[:, None]  # min 0 max 1
        cross_corr = np.corrcoef(concat_barycenters)
        tri = np.triu_indices(cross_corr.shape[0], k=1)

        # get highly correlated barycenters
        corr_arr = cross_corr[tri[0], tri[1]]
        # threshold mean + 2 std same for all following iterations
        if len(barycenters) > 3:
            if self.barycenter_correlation_threshold is None:
                correlation_threshold = np.nanmean(corr_arr) + 2 * np.nanstd(corr_arr)
                if correlation_threshold > hard_threshold:
                    correlation_threshold = hard_threshold
                self.barycenter_correlation_threshold = correlation_threshold
            else:
                correlation_threshold = self.barycenter_correlation_threshold
        else:
            # if there are 3 or fewer clusters use hard threshold
            correlation_threshold = hard_threshold
        tri = np.array(tri)
        high_corr_barycenters_indices = [i for i in np.where(corr_arr > correlation_threshold)[0]]
        if len(high_corr_barycenters_indices) == 0:
            print('\tstopping consolidation because no barycenters are correlated')
            return
        else:
            # get the labels of the clusters that correspond to the highly correlated barycenters
            # only consolidate the highest correlated pair of highly correlated barycenters to avoid consolidating too many clusters at once
            highest_corr_barycenters = tri[:, corr_arr.argmax()]
            labels_to_consolidate = [self.labels[i] for i in highest_corr_barycenters]
            print(f'\thighly correlated barycenters: {labels_to_consolidate}',)

        barycenters_to_consolidate = smooth_barycenters[highest_corr_barycenters]
        cell_indices = self.cluster_df.loc[
            self.cluster_df['cluster_label_pruned'].isin(labels_to_consolidate)
        ].index
        trace_subset = self.trace_df.loc[cell_indices]
        counts = trace_subset.groupby(level=0).size()
        if (counts != len(self.stimuli)).any():
            bad_cells = counts[counts != len(self.stimuli)]
            raise ValueError(
                f"Expected {len(self.stimuli)} trace rows per cell, "
                f"found irregular counts for cells: {bad_cells.to_dict()}"
            )
        traces = trace_subset.to_numpy().reshape(
            len(cell_indices), len(self.stimuli), trace_subset.shape[1]
        )
        stim_diffs = barycenters_to_consolidate[0] - barycenters_to_consolidate[1]
        stim_diffs_means = np.mean(stim_diffs, axis=1)
        stim_filter = np.abs((stim_diffs_means - np.mean(stim_diffs_means)) / np.std(stim_diffs_means)) >= 2
        if any(stim_filter):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TimeSeriesKMeans(
                    n_clusters=2,
                    metric='dtw',
                    metric_params={
                        "global_constraint": "sakoe_chiba",
                        "sakoe_chiba_radius": self.trace_df.shape[1] // 20},
                    max_iter_barycenter=5,
                    random_state=0,
                    n_init=3
                )
                labels = model.fit_predict(np.concat(traces[:, stim_filter].transpose(1, 0, 2), axis=1))
            new_labels = ['_x_'.join(labels_to_consolidate) + f'_split{i}' for i in labels]
        else:
            new_labels = ['_x_'.join(labels_to_consolidate) + '_combined' for _ in range(traces.shape[0])]
        print('\tchanging labels to: ', np.unique(new_labels))
        self.cluster_df.loc[cell_indices, 'cluster_label_pruned'] = new_labels
        self.labels = np.sort(self.cluster_df['cluster_label_pruned'].unique())[1:]  # remove -1 label
        self.consolidated_barycenters = self._calculate_barycenters()
        self._compare_barycenters_recursively(iteration=iteration+1)

    def consolidate_barycenters(self, max_iterations=10):
        print("\nconsolidating barycenters")
        self.consolidated_barycenters = None
        self.barycenter_correlation_threshold = None
        self.barycenters = self._calculate_barycenters()
        self.max_iterations_consolidation = max_iterations
        self._compare_barycenters_recursively(iteration=0)
        if self.consolidated_barycenters is not None:
            print(f'\treduced barycenters from {len(self.barycenters)} to {len(self.consolidated_barycenters)}')
            self.barycenters = self.consolidated_barycenters
            self._get_colors()
        else:
            print('\tno change in barycenters')

    def return_barycenters(self, recalculate=True):
        if recalculate:
            self.barycenters = self._calculate_barycenters()
        return self.barycenters
