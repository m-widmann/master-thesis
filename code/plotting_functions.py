from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.colors import TwoSlopeNorm, Normalize
from skimage.measure import find_contours
from general_util_functions import best_rectangle
from itertools import product, combinations
from scipy.stats import mannwhitneyu, false_discovery_control


def get_spaced_colors(n):
    colors = cm.jet(np.linspace(0, 1, n))
    half = (n + 1) // 2
    a = np.arange(half)
    b = np.arange(n // 2) + half

    order = np.ravel(np.column_stack((a[:len(b)], b)))
    if len(a) > len(b):
        order = np.append(order, a[-1])
    colors = colors[order]
    return colors

# n = 10
# data = np.random.normal(size=n*50)
# data = data.reshape(n,-1)
# row,cols = best_rectangle(n)
# fig,axes = plt.subplots(nrows=row,ncols=cols)
# axes = axes.flatten()
# for c,ax,y in zip(get_spaced_colors(n),axes,data):
#     ax.plot(y, c=c, lw=2, clip_on=False)
# for ax in axes:
#     for s in ax.spines:
#         ax.spines[s].set_visible(False)
#         ax.set_yticks([])
#         ax.set_xticks([])
# plt.show()


def group_ylim(traces, pad, co, q):
    if q is None:
        traces = np.nanpercentile(traces, q=[20, 80], axis=0)
        lo = np.min(traces[0])
        hi = np.max(traces[1])
    else:
        lo, hi = np.nanpercentile(traces, q=q)

    if co is not None:
        if hi > co:
            hi = co
        if lo < -co:
            lo = -co

    p = (abs(hi) + abs(lo)) * pad
    return lo - p, hi + p


def prepare_trace_df(trace_df, stimuli=None, stim_start=None, stim_end=None):
    try:
        trace_df = trace_df.drop(columns='reliability')
    except KeyError:
        pass

    if stim_start is None:
        stim_start = 0
    if stim_end is None:
        stim_end = trace_df.shape[1]

    if stimuli is None:
        stimuli = list(trace_df.index.get_level_values('stimulus').unique())
    else:
        stimuli = [s for s in stimuli if s in trace_df.index.get_level_values('stimulus').unique()]
        trace_df = deepcopy(trace_df.loc[
                                trace_df.index.get_level_values(level='stimulus').isin(stimuli), :
                            ])

    other_levels = [
        trace_df.index.get_level_values(i)
        for i in range(trace_df.index.nlevels)
        if i != trace_df.index.names.index('stimulus')
    ]
    other_level_names = [n for n in trace_df.index.names if n != 'stimulus']
    new_stim_level = pd.Categorical(
        trace_df.index.get_level_values('stimulus'),
        categories=stimuli,
        ordered=True
    )
    new_index = pd.MultiIndex.from_arrays(
        [*other_levels, new_stim_level],
        names=[*other_level_names, 'stimulus']
    )
    trace_df.index = new_index

    return trace_df, stimuli, stim_start, stim_end


def prepare_colors(colors, n_groups):
    colors_is_dict = isinstance(colors, dict)
    if colors is None:
        colors = get_spaced_colors(n_groups)
    return colors, colors_is_dict


def style_axis(ax, title=None, stim_text=None):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, loc='left', fontsize=10)
    if stim_text:
        ax.text(0.05, 0.95, stim_text,
                transform=ax.transAxes, fontsize=8, va='top', ha='left')


def add_scale_bar_y(ax, y_scale_length, y_scale_label):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_pos = xmin
    y_start = 0
    y_end = y_scale_length

    ax.plot([x_pos, x_pos], [y_start, y_end], color="black", lw=2, clip_on=False)
    ax.text(
        x_pos - 0.04 * (xmax - xmin), (y_start + y_end) / 2,
        y_scale_label, ha="right", va="center", fontsize=8, color="black", rotation=90
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_scale_bar_x(ax, x_scale_length, x_scale_label):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_start, x_end = 0, x_scale_length
    y_pos = ymin

    ax.plot([x_start, x_end], [y_pos, y_pos], color="black", lw=2, clip_on=False)
    ax.text(
        (x_start + x_end) / 2, y_pos - 0.03 * (ymax - ymin),
        x_scale_label, ha="center", va="top", fontsize=8, color="black"
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def group_plotter_stimuli_separated(
        trace_df,
        group_df,
        col_name="cell_type",
        index_level_name=None,
        cell_id_index_name='cell_name',
        stim_start=None,
        stim_end=None,
        skip_values=None,
        colors=None,
        color_by_index=False,
        stimuli=None,
        fig_size=None,
        show_individual_traces=False,
        use_same_axes_for_all_plots=True,
        use_same_axes_for_one_cell_type=False,
        show_scale_bar=True,
        x_scale_length=10,
        y_scale_length=0.2,
        x_scale_label="5s",
        y_scale_unit=r"$\Delta$F/F0",
        title=None,
        hide_all_titles=False,
        show_stim_names=True,
        show_panel_title=True,
        display_number_of_cells=True,
        display_fraction_of_cells=False,
        total_number_of_cells=None,
        max_stim_cols=2,
        ylim_pad=0.3,
        ylim_co=None,
        ylim_percentiles=None,
        outer_wspace_hspace=(.2,.2),
):
    y_scale_label = str(y_scale_length) + y_scale_unit

    if skip_values is None:
        skip_values = ['nan', 'None', '', '-', '-1']

    trace_df, stimuli, stim_start, stim_end = prepare_trace_df(
        deepcopy(trace_df), stimuli, stim_start, stim_end
    )
    n_stimuli = len(stimuli)

    if hide_all_titles:
        show_panel_title = False
        show_stim_names = False
        display_fraction_of_cells = False
        display_number_of_cells = False
        title = None

    if show_individual_traces and ylim_percentiles is None:
        ylim_percentiles = (10, 90)

    if display_fraction_of_cells and total_number_of_cells is None:
        display_fraction_of_cells = False



    cell_types = [x for x in group_df[col_name].unique() if str(x) not in skip_values]
    n_cell_types = len(cell_types)

    if colors is None or index_level_name is None:
        color_by_index = False
    colors, colors_is_dict = prepare_colors(colors, n_cell_types)

    if index_level_name is None:
        outer_rows, outer_columns = best_rectangle(n_cell_types)
        index_values = None
    else:
        index_values = list(group_df.index.get_level_values(index_level_name).unique())
        outer_rows = n_cell_types
        outer_columns = len(index_values)

    if index_level_name is None:
        grouped_df_iterator = group_df.groupby(col_name, sort=False)
    else:
        grouped_df_iterator = group_df.groupby([index_level_name, col_name], sort=False)

    key_lookup = [[None for _ in range(outer_columns)] for _ in range(outer_rows)]
    cell_name_list = [[None for _ in range(outer_columns)] for _ in range(outer_rows)]

    for key, df in grouped_df_iterator:
        if index_level_name is None:
            if str(key) in skip_values:
                continue
            i = cell_types.index(key)
            c = i // outer_rows
            r = i % outer_rows
        else:
            if str(key[1]) in skip_values:
                continue
            r = cell_types.index(key[1])
            c = index_values.index(key[0])

        key_lookup[r][c] = key
        cell_name_list[r][c] = list(df.index.get_level_values(cell_id_index_name).unique())

    trace_by_cell_stim = {
        (cell_name, stim): df.to_numpy()
        for (cell_name, stim), df in trace_df.groupby([cell_id_index_name, 'stimulus'], sort=False, observed=False)
    }

    panel_trace_lookup = {}
    for i, j in product(range(outer_rows), range(outer_columns)):
        key = key_lookup[i][j]
        cells = cell_name_list[i][j]

        if key is None or cells is None:
            continue

        stim_dict = {}
        for stim in stimuli:
            arrs = [
                trace_by_cell_stim[(cell_name, stim)]
                for cell_name in cells
                if (cell_name, stim) in trace_by_cell_stim
            ]
            if len(arrs) == 0:
                continue
            stim_dict[stim] = np.vstack(arrs)

        panel_trace_lookup[(i, j)] = stim_dict

    global_ylim = None
    type_ylims = None

    if use_same_axes_for_all_plots:
        all_arrays = []
        for stim_dict in panel_trace_lookup.values():
            for arr in stim_dict.values():
                all_arrays.append(arr)
        if len(all_arrays) > 0:
            global_ylim = group_ylim(
                np.vstack(all_arrays),
                pad=ylim_pad, co=ylim_co, q=ylim_percentiles,
            )
    elif use_same_axes_for_one_cell_type:
        type_ylims = {}
        for ct in cell_types:
            all_arrays = []
            for i, j in product(range(outer_rows), range(outer_columns)):
                key = key_lookup[i][j]
                if key is None:
                    continue

                this_ct = key if index_level_name is None else key[1]
                if this_ct != ct:
                    continue

                stim_dict = panel_trace_lookup.get((i, j), {})
                for arr in stim_dict.values():
                    all_arrays.append(arr)

            if len(all_arrays) > 0:
                type_ylims[ct] = group_ylim(
                    np.vstack(all_arrays),
                    pad=ylim_pad, co=ylim_co, q=ylim_percentiles,
                )

    fig = plt.figure(figsize=fig_size)
    outer = GridSpec(outer_rows, outer_columns, figure=fig, wspace=outer_wspace_hspace[0], hspace=outer_wspace_hspace[1])
    all_axes = []

    if n_stimuli == 2:
        inner_rows, inner_columns = 1, 2
    else:
        inner_rows, inner_columns = best_rectangle(n_stimuli, max_cols=max_stim_cols)

    if colors_is_dict:
        color_lookup = colors
    else:
        color_lookup = {ct: colors[i] for i, ct in enumerate(cell_types)}

    for i, j in product(range(outer_rows), range(outer_columns)):
        key = key_lookup[i][j]
        if key is None:
            continue

        if index_level_name is None:
            cell_type = key
            index_level = None
        else:
            index_level, cell_type = key

        ylim = None
        if global_ylim is not None:
            ylim = global_ylim
        elif type_ylims is not None:
            ylim = type_ylims.get(cell_type, None)

        trace_stim_dict = panel_trace_lookup.get((i, j), {})
        if len(trace_stim_dict) == 0:
            continue

        inner = outer[i, j].subgridspec(inner_rows, inner_columns, wspace=0.05, hspace=0.05)
        if color_by_index:
            clr = color_lookup.get(index_level, 'black')
        else:
            clr = color_lookup.get(cell_type, 'black')

        first_n_cells = None

        for k, stim in enumerate(stimuli):
            if stim not in trace_stim_dict:
                continue

            r = k // inner_columns
            c = k % inner_columns
            ax = fig.add_subplot(inner[r, c])

            stim_traces = trace_stim_dict[stim]
            n_cells = stim_traces.shape[0]
            if first_n_cells is None:
                first_n_cells = n_cells

            local_ylim = ylim
            if local_ylim is None:
                local_ylim = group_ylim(
                    stim_traces,
                    pad=ylim_pad, co=ylim_co, q=ylim_percentiles,
                )

            ax.set_ylim(local_ylim)
            all_axes.append(ax)

            ax.axvspan(stim_start, stim_end, color='lightgrey', alpha=0.3)
            ax.hlines(y=0, color='grey', linestyle=':', xmin=0, xmax=stim_traces.shape[1])

            if show_individual_traces:
                for single_trace in stim_traces:
                    ax.plot(single_trace, color='grey', alpha=0.3, lw=0.8)

            quartiles = np.quantile(stim_traces, q=(0.25, 0.5, 0.75), axis=0)
            ax.fill_between(
                np.arange(quartiles.shape[1]),
                quartiles[0], quartiles[2],
                color=clr, alpha=0.5
            )
            ax.plot(quartiles[1], c=clr, lw=2)
            if show_panel_title:
                panel_title = (
                    f"  {cell_type}"
                    if index_level is None
                    else f"  {cell_type} - {index_level}"
                )
            else:
                panel_title = ''
            if display_number_of_cells:
                panel_title += f" ({first_n_cells} cells)"
            elif display_fraction_of_cells:
                if isinstance(total_number_of_cells, dict) and index_level_name is not None:
                    n = total_number_of_cells[index_level]
                else:
                    n = total_number_of_cells
                panel_title += f" ({np.round((first_n_cells / n) * 100, 1)}% of cells)"
            if not show_panel_title:
                panel_title=panel_title.replace('(', '').replace(')', '')

            stim_text = str(stim).replace("_", " ") if show_stim_names else None
            style_axis(
                ax,
                title=panel_title if k == 0 else None,
                stim_text=stim_text,
            )

            if show_scale_bar:
                if i == 0 and j == 0 and r == 0 and c == 0:
                    add_scale_bar_x(ax, x_scale_length, x_scale_label)

                if use_same_axes_for_all_plots and (i == 0 and j == 0 and r == 0 and c == 0):
                    add_scale_bar_y(ax, y_scale_length, y_scale_label)
                elif (use_same_axes_for_one_cell_type and
                      (r == 0 and c == 0 and (j == 0 or index_level_name is None))):
                    add_scale_bar_y(ax, y_scale_length, y_scale_label)
                elif not (use_same_axes_for_all_plots or use_same_axes_for_one_cell_type):
                    add_scale_bar_y(ax, y_scale_length, y_scale_label)

    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    plt.show()
    return fig


def plot_separated_histograms(
        corr_df,
        colors,
        n_bins=40,
        value_col_name='correlation_value',
        group_col_name='cell_type',
        index_name='age',
        xlab='Correlation to Regressor',
        custom_index_order=None):
    column_values = corr_df[group_col_name].unique()
    column_values = column_values[~pd.isnull(column_values)]
    fig = plt.figure()
    outer_gs = fig.add_gridspec(1, len(column_values), wspace=0.35)

    for col, (ct, ct_df) in enumerate(corr_df.groupby(group_col_name)):
        bins = np.linspace(
            ct_df[value_col_name].min(),
            ct_df[value_col_name].max(),
            n_bins
        )
        max_density = 0
        n_index = 0
        for age, age_df in ct_df.groupby(index_name):
            counts, _ = np.histogram(age_df['correlation_value'], bins=bins, density=True)
            max_density = max(max_density, counts.max())
            n_index += 1
        y_top = max_density * 1.1
        # negative hspace is what makes the ridges overlap
        inner_gs = outer_gs[0, col].subgridspec(n_index, 1, hspace=-0.5)
        groups = ct_df.groupby(index_name)
        if custom_index_order is not None:
            groups = sorted(groups, key=lambda x: custom_index_order.index(x[0]))

        for i, (age, age_df) in enumerate(groups):
            ax = fig.add_subplot(inner_gs[i, 0])
            ax.hist(
                age_df['correlation_value'], bins=bins, density=True,
                color=colors[age], edgecolor='none', alpha=0.9
            )
            ax.set_ylim(0, y_top)
            ax.set_xlim(bins[0], bins[-1])
            ax.patch.set_alpha(0)  # transparent so the ridge behind shows through

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)

            if i == len(custom_index_order) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('#444')  # darker at bottom
                if col == 0:
                    ax.set_xlabel(xlab, fontsize=10)
            else:
                ax.spines['bottom'].set_color('#AAA')  # lighter line on the others
                ax.set_xticks([])

            ax.set_yticks([])
            if col == 0:
                ax.text(-0.05, 0, age,
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
                if i == 1:
                    ax.set_ylabel('Density', fontsize=10, labelpad=40)

        fig.text(outer_gs[0, col].get_position(fig).x0
                 + 0.5 * outer_gs[0, col].get_position(fig).width,
                 0.90, ct, ha='center', va='bottom', fontsize=10)


    plt.show()


def plot_active_regions_on_brain(
        df,
        key,
        brain_region_masks,
        brain_volume,
        y_crop=(250, 1050),
        x_crop=None,
        color_map=None,
        custom_color_range=None,
        slices=[(100, 110), (90, 95), (82, 88), (70, 77), (58, 64)],
        show_region_labels=False,
        title=None):
    if title is None: title = key

    if x_crop is None: x_crop = (0, -1)

    sorted_activity = df.sort_values(by=key, ascending=True)

    change_values = sorted_activity[key].values
    if custom_color_range is None:
        min_change = np.round(np.min(change_values), 2)
        max_change = np.round(np.max(change_values), 2)
    else:
        min_change = custom_color_range[0]
        max_change = custom_color_range[1]

    if min_change >= 0:
        norm = Normalize(vmin=min_change, vmax=max_change)
        if color_map is None: color_map = 'plasma'
    else:
        norm = TwoSlopeNorm(vmin=min_change, vcenter=0, vmax=max_change)
        if color_map is None: color_map = 'Spectral_r'
    cmap = plt.get_cmap(color_map)
    cmap_colors = cmap(norm(change_values))
    color_dict = {region: cmap_colors[i] for i, region in enumerate(sorted_activity['region'])}

    region_patches = [[] for _ in range(len(slices))]
    for region in sorted_activity['region']:
        for i, (z0, z1) in enumerate(slices):
            mask_coords = brain_region_masks[region]
            z_filter = (mask_coords[0] >= z0) & (mask_coords[0] < z1)
            if any(z_filter):
                # Define grid size for the 2D mask on (y, x)
                y_sel = mask_coords[1][z_filter]
                x_sel = mask_coords[2][z_filter]

                y_min = int(np.floor(np.min(y_sel))) - 10  # add padding so find_contours works
                y_max = int(np.ceil(np.max(y_sel))) + 11
                ny = y_max - y_min

                x_min = int(np.floor(np.min(x_sel))) - 10
                x_max = int(np.ceil(np.max(x_sel))) + 11

                nx = x_max - x_min

                mask_2d = np.zeros((ny, nx), dtype=bool)
                y_idx = y_sel - y_min  # map y to [0, ny-1]
                x_idx = x_sel - x_min  # map x to [0, nx-1]
                mask_2d[y_idx, x_idx] = True

                contours = find_contours(mask_2d, level=0)
                for contour in contours:
                    region_patches[i].append(
                        mpatches.PathPatch(
                            mpath.Path((contour[:, [1, 0]] + [x_min - x_crop[0], y_min - y_crop[0]]).astype(int)),
                            # (x, y) in global coordinates
                            facecolor=color_dict[region],
                            edgecolor='black',
                            lw=1,
                            alpha=0.5,
                            label=region
                        )
                    )

    fig, axes = plt.subplots(1, len(slices))

    for i, (z0, z1) in enumerate(slices):
        try:
            ax = axes[i]
        except TypeError:
            #only one ax
            ax = axes

        ax.imshow(np.nanmean(brain_volume[z0:z1, y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]], axis=0), cmap='grey')
        for patch in region_patches[i]:
            p = deepcopy(patch)
            ax.add_patch(p)
            if show_region_labels:
                verts = patch.get_path().vertices
                centroid = verts.mean(axis=0)
                ax.text(centroid[0], centroid[1], patch.get_label().split(' - ')[1])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    # Add a colorbar at the bottom
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=axes,  # Pass all axes to span the entire figure
        orientation='horizontal',
        pad=0.,
        fraction=0.02,
        aspect=40)
    tick_values = np.array([min_change, 0., max_change])
    tick_labels = np.array([f'{min_change}', '0', f'{max_change}'])
    if min_change > 0:
        tick_values = tick_values[[0, 2]]
        tick_labels = tick_labels[[0, 2]]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Difference in Fraction of \nActive Cells per Region', fontsize=12)
    plt.subplots_adjust(wspace=.05)
    plt.subplots_adjust(left=.01, right=.99, bottom=.13, top=.99)
    fig.suptitle(title)
    plt.show()


def plot_boxplots_corr_values(
    df,
    index_level_name='age',
    group_col_name='cell_type',
    data_col_name='correlation_value',
    do_stat_test=False,
    print_pvalues=False,
    color_by_index=False,
    colors=None,
    figsize=None,
    n_rows_cols = None,
    tick_values=[.3,.6,.9],
    bp_width=.6,
    title=None,
    hide_labels=False,
    y_offset_value=.08,
    median_color=None,
):
    group_types = df[group_col_name].unique()
    n_groups = len(group_types)

    if n_rows_cols is None:
        n_rows_cols = best_rectangle(n_groups, max_cols=4)

    colors, colors_is_dict = prepare_colors(colors, n_groups)
    if colors_is_dict:
        color_lookup = colors
    else:
        color_by_index = False
        color_lookup = {ct: colors[i] for i, ct in enumerate(group_types)}

    if median_color is None: median_color = 'black'
    fig, axes = plt.subplots(nrows=n_rows_cols[0], ncols=n_rows_cols[1], figsize=figsize)
    axes = axes.flatten()
    for i, (ct, ct_df) in enumerate(df.groupby(group_col_name)):
        ax = axes[i]
        ages = []
        data = []
        for j,(age, age_df) in enumerate(ct_df.groupby(index_level_name)):
            ages.append(age)
            plot_data = age_df[data_col_name].values
            data.append(plot_data)
            bp = ax.boxplot(plot_data, positions=[j+1,], patch_artist=True, showfliers=False, widths=bp_width)
            box_color = color_lookup[age] if color_by_index else color_lookup[ct]
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_edgecolor('black')
            for median in bp['medians']:
                median.set_color(median_color)
                median.set_linewidth(2)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
        if hide_labels:
            ages = ['','','']
        ax.set_xticklabels(ages)

        ymax = ax.get_ylim()[1]

        if do_stat_test:
        #pairwise tests between all age groups in this subplot
            y_offset = ymax * y_offset_value
            p_values = []
            positions = []
            for a1, a2 in combinations(range(len(data)), 2):
                stat, p = mannwhitneyu(data[a1], data[a2], alternative='two-sided')
                p_values.append(p)
                positions.append((a1, a2))
            p_adj = false_discovery_control(p_values, method='bh')

            for j, (p, (a1, a2)) in enumerate(zip(p_adj, positions)):
                if print_pvalues:
                    print(f"{ct}: {ages[a1]} x {ages[a2]}: {p}")

                if p < 0.001:
                    symbol = '***'
                elif p < 0.01:
                    symbol = '**'
                elif p < 0.05:
                    symbol = '*'
                else:
                    continue
                y = ymax + (j + .5) * y_offset
                x1, x2 = a1 + 1, a2 + 1
                ax.plot([x1, x1, x2, x2], [y, y + y_offset * 0.2, y + y_offset * 0.2, y], color='k', lw=1)
                ax.text((x1 + x2) / 2, y + y_offset * 0.25, symbol, ha='center', va='bottom', fontsize=8)

        ax.set_ylim((0, ax.get_ylim()[1]))
        ticks = []
        for t in tick_values:
            if t < ymax:
                ticks.append(t)

        ax.set_yticks(ticks)
        if hide_labels:
            ax.set_yticklabels(['' for _ in ticks])
        if not hide_labels:
            ax.set_xlabel(ct, fontsize=12)
        if i == 0 and not hide_labels:
            ax.set_ylabel(r'$\rho$')

    for ax in axes:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        if len(ax.lines) == 0:
            ax.set_xticks([])
            ax.set_yticks([])
    if title is not None and not hide_labels:
        fig.suptitle(title)
    plt.show()
    return fig
