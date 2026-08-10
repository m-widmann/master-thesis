import numpy as np
import pathlib
import h5py
from cv2 import arcLength, contourArea
import pandas as pd
from itertools import product

def load_segmentation_mask(path, channel=0, repeat=0, tile=0, z_stack=0, segmentation='cellpose'):
    repeat = str(repeat).zfill(2)
    tile = str(tile).zfill(3)
    z_stack = str(z_stack).zfill(3)

    with h5py.File(path, 'r') as f:
        return np.array(f[f'repeat{repeat}_tile{tile}_z{z_stack}_950nm/preprocessed_data/fish00/{segmentation}_segmentation/masks'])

def load_contours_and_centroids(path, channel=0, repeat=0, tile=0, z_stack=0, wavelength=950, segmentation='cellpose', plane_key=None):
    repeat = str(repeat).zfill(2)
    tile = str(tile).zfill(3)
    z_stack = str(z_stack).zfill(3)

    if plane_key is None:
        plane_key = f'repeat{repeat}_tile{tile}_z{z_stack}_{wavelength}nm'

    with h5py.File(path, 'r') as f:
        contour_group = f[f'{plane_key}/preprocessed_data/fish00/{segmentation}_segmentation/unit_contours/']
        contours = [dset[()] for dset in contour_group.values()]
        centroids = np.array(f[f'{plane_key}/preprocessed_data/fish00/{segmentation}_segmentation/unit_centroids'])
    return contours, centroids

def clean_segmentation_shapes(contours, max_size_factor=2., min_size_factor=1.1, circularity_threshold=.6):
    areas = np.array([contourArea(cnt) for cnt in contours])
    max_size = np.median(areas) * max_size_factor
    min_size = np.quantile(areas, q=.01) * min_size_factor
    size_outliers = np.where(np.bitwise_or(areas > max_size, areas < min_size))[0]

    perimeters = np.array([arcLength(cnt, True) for cnt in contours])
    circularities = 4 * np.pi * areas / ((perimeters ** 2) + 1e-12)
    circ_outliers = np.where(circularities < circularity_threshold)[0]

    cells_to_remove = np.unique(np.concatenate([size_outliers, circ_outliers]))
    mask = np.ones(len(contours), dtype=bool)
    mask[cells_to_remove] = False
    good_cells = np.arange(len(contours))[mask]
    return good_cells

def normalize_traces(traces, method=None, window=None):
    if method is None:
        method = "df/f0"
    if window is None:
        window = (0,2)
    f0 = np.nanmean(traces[:, :,window[0]:window[1]], 2)
    if method == "df/f0":
        return (traces - f0[:, :, None]) / f0[:, :, None]
    elif method == "df":
        return traces - f0[:, :, None]
    else:
        raise ValueError('Normalization method must be either "df/f0" or "df"')


def estimate_midline_from_mask(mask, window=10):
    bin_mask = mask == 9999 #9999 equals 0
    image_midline = mask.shape[1] // 2
    best_midline = image_midline - window + bin_mask[:, image_midline - window:image_midline + window].sum(
        axis=0).argmax() #sum all black pixels in each column to estimate the midline
    return best_midline

def map_ipsi_contra(row):
    hemisphere = row['hemisphere']
    stim = row['stimulus']
    if "bright" in stim: #oopsie
        return stim
    if hemisphere == 'left':
        ipsi = 'left'
        contra = 'right'
    else:
        ipsi = 'right'
        contra = 'left'
    return stim.replace(ipsi,'ipsi').replace(contra,'contra')


def get_segmented_signals_multiindex_df(
        path,
        z_plane=None,
        repeat=None,
        tile=None,
        wavelengths=950,
        registration_type=None,
        brain_regions_mask_path=None,
        norm_mode=None,
        norm_window=None,
        segmentation_type='cellpose',
        do_flipping_based_on_hemisphere=False,
        do_segmentaion_shape_cleaning=False,
        return_cell_position_df=False,
        return_brain_region_assignments=False,
        index_columns=None
    ):

    """

    Parameters
    ----------
    path: str or pathlib path
        Path to the preprocessed_data.h5 file or the directory containing it.
    z_plane: int, iterable of ints, or 'all', optional
        Z plane(s) to load. If 'all', loads all z planes. Default is 'all'.
        'all' looks up number of z_planes from experiment_information.txt file, this fails defaults to plane 0
    repeat: int, iterable of ints, or 'all'. Default is 'all'.
    tile: int, iterable of ints, optional. Default is 0.
    wavelengths: int, optional. Default is 950.
    registration_type: str, optional. Default is None. (only works with 'ZBRAIN' for now)
    brain_regions_mask_path: str
        Path to the brain region masks hdf5 file, only needed if return_brain_region_assignments is True and registration_type includes 'ZBRAIN'.
         If None, defaults to Z:/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/all_masks_indexed.hdf5
    norm_mode: "df/f0" or "df".
        Uses normalize_traces function above.
    norm_window: tuple, optional. Default is (0,2).
        Time window in frames to use as baseline for normalization. Only used if norm_mode is set.
    segmentation_type: str, optional. Default is 'cellpose'.
        Which segmentation to load from the hdf.
    do_flipping_based_on_hemisphere: bool, optional. Default is True.
            If True, flips the stimulus labels for cells in the right hemisphere to be based on ipsi/contra rather than left/right.
            Only works if stimulus labels contain "left" and "right". Estimated the midline based on the segmentation.
            If registration_type=='ZBRAIN' uses default x value of 318 for midline.
    return_cell_position_df: bool, optional. Default is False.
    return_brain_region_assignments: bool, optional. Default is False.
    index_columns: list of str. All possible columns are:
        ['fish_id','hemisphere','repeat', 'z_plane', 'tile', 'wavelength', 'cell_number','cell_name','stimulus','trial']

    Returns
    -------
`   pandas Multiindex DataFrame
    Each row is one trial of one stimulus for one segmented cell.
    If norm_mode is set ("df/f0" or "df") traces are normalized accordingly, using the mean of the frames in norm_window as baseline.
    Index columns can be limited, all possible columns are:
    ['fish_id','hemisphere','repeat', 'z_plane', 'tile', 'wavelength', 'cell_number','cell_name','stimulus','trial']
    If return_cell_position_df is True, also returns a second dataframe with the position of each cell and its hemisphere.
    If return_brain_region_assignments is True, also returns the brain region assignments for each cell based on the ZBRAIN atlas in the second df.
    """


    path = pathlib.Path(path)
    if path.is_dir():
        try:
            path = next(path.glob('*preprocessed_data.h5'))
        except StopIteration:
            raise FileNotFoundError(f'No "preprocessed_data" file found in directory {path}')
    try:
        experiment_info = get_experiment_information(path.parent / 'experiment_information.txt')
    except Exception as e:
        print(f'Error loading experiment information: {e}')
        experiment_info = None

    all_columns = ['fish_id','hemisphere','repeat', 'z_plane', 'tile', 'wavelength', 'cell_number','cell_name','stimulus','trial']
    if index_columns is None:
        index_columns = all_columns
    if 'cell_name' not in index_columns:
        index_columns = ['cell_name'] + index_columns
    if 'trial' not in index_columns:
        index_columns.append('trial')
    columns_to_drop = [col for col in all_columns if col not in index_columns]

    # get the keys for the different scanning parameters as lists
    # tiles
    if tile is None:
        tile = [0]

    # repeats
    if repeat is None:
        repeat = 'all'
    if repeat == 'all':
        if experiment_info is None:
            print('No experiment information file - defaulting to repeat 0')
            repeat = 0
        else:
            repeat = range(int(experiment_info['number_of_repeats']))
    if hasattr(repeat, '__iter__'):
        repeats = repeat
    else:
        repeats = [repeat]

    # z planes
    if z_plane is None:
        z_plane = 'all'
    if z_plane == 'all':
        if experiment_info is None:
            print('No experiment information file - defaulting to z_plane 0')
            z_plane = 0
        else:
            z_plane = range(int(experiment_info['z_scan_number_of_z_positions']))
    if hasattr(z_plane, '__iter__'):
        z_stack = z_plane
    else:
        z_stack = [z_plane]

    if not hasattr(wavelengths, '__iter__'):
        wavelengths = [wavelengths]

    if registration_type is None:
        reg_key = ''
        ZBRAIN_registered = False
        return_brain_region_assignments = False
    else:
        reg_key = f'_ants_{registration_type}_registered'
        ZBRAIN_registered = 'ZBRAIN' in registration_type

        if return_brain_region_assignments:
            return_cell_position_df = True
            if brain_regions_mask_path is None:
                brain_regions_mask_path = r"Z:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\all_masks_indexed.hdf5"
            masks = h5py.File(brain_regions_mask_path)
            regions = list(masks.keys())




    path = str(path)
    # create unique identifiers from filename
    fish_id = path.split('fish')[-1][:3]
    date = path.split('\\')[-2].split('_')[0].split('-')

    traces_df = []
    cell_position_df = []

    print(f'Loading fish {pathlib.Path(path).stem}')
    for r,t,z,w in product(repeats, tile, z_stack, wavelengths):
        plane_key = f'repeat{str(r).zfill(2)}_tile{str(t).zfill(3)}_z{str(z).zfill(3)}_{w}nm'
        print(f'Plane: {plane_key}')

        # get midline for this plane
        # if ZBRAIN midline is approx 318 in x coordinates
        if ZBRAIN_registered:
            midline = 318
        else:
            midline = estimate_midline_from_mask(load_segmentation_mask(path, z_stack=z, tile=t, repeat=r))

        with h5py.File(path, "r") as preproc_hdf5:
            cell_number = preproc_hdf5[plane_key]['preprocessed_data']['fish00'][f'{segmentation_type}_segmentation'][
                'unit_names']
            coordinates = np.array(preproc_hdf5[plane_key]['preprocessed_data']['fish00'][f'{segmentation_type}_segmentation'][
                         f'unit_centroids{reg_key}'])
            if coordinates.shape[1] == 3:
                cell_x, cell_y, cell_z = coordinates.T
            else:
                cell_x, cell_y = coordinates.T
                cell_z = np.array([z] * len(cell_number))

            hemisphere = ['right' if x > midline else 'left' for x in cell_x]
            fish_idx = ('_'.join(date).replace('_','') + "-" + str(fish_id))
            cell_name = 'f' + fish_idx + 'r' + str(r).zfill(2) + 't' + str(t).zfill(3) + 'z' + str(z).zfill(3) + 'c' + (np.array(cell_number)).astype(str) #unique identifier for each cell
            cell_number = np.array(cell_number).astype(int) - 10000
            stim_df_list = []
            for stim_key in preproc_hdf5[plane_key]['preprocessed_data']['fish00'][f'{segmentation_type}_segmentation'][
                'stimulus_aligned_dynamics'].keys():
                if norm_mode is None:
                    traces = np.array(preproc_hdf5[plane_key]['preprocessed_data']['fish00'][f'{segmentation_type}_segmentation'][
                        'stimulus_aligned_dynamics'][stim_key]['F'])
                else:
                    traces = normalize_traces(
                        preproc_hdf5[plane_key]['preprocessed_data']['fish00'][f'{segmentation_type}_segmentation'][
                            'stimulus_aligned_dynamics'][stim_key]['F'],
                    method=norm_mode, window=norm_window)
                n_trials, n_cells, n_t = traces.shape
                # reshape array to turn into multiindex df of structure cell_id x trial_no, time
                index = pd.MultiIndex.from_product([cell_name, np.arange(n_trials)], names=['cell_name', 'trial'])
                df = pd.DataFrame(traces.transpose(1, 0, 2).reshape(n_cells * n_trials, n_t),index=index)
                # add constant level of stimulus to the index
                df['stimulus'] = stim_key
                df['fish_id'] = fish_idx
                df['hemisphere'] = np.repeat(np.array(hemisphere), n_trials)
                df['cell_number'] = np.repeat(cell_number, n_trials)
                df['z_plane'] = z
                df['repeat'] = r
                df['tile'] = t
                df['wavelength'] = w
                stim_df_list.append(df)
            plane_traces = pd.concat(stim_df_list)

            if do_segmentaion_shape_cleaning:
                contour_group = preproc_hdf5[f'{plane_key}/preprocessed_data/fish00/{segmentation_type}_segmentation/unit_contours/']
                contours = [dset[()] for dset in contour_group.values()]
                good_cells = clean_segmentation_shapes(contours=contours)
                plane_traces = plane_traces.loc[plane_traces['cell_number'].isin(good_cells),:]

            if return_cell_position_df:
                if stim_key == '':
                    cell_df = pd.DataFrame(np.array([cell_name, cell_z, cell_x, cell_y, hemisphere]).T)
                    cell_df.columns = ['cell_name','z','y','x','hemisphere']
                else:
                    cell_df = pd.DataFrame(np.array([cell_name, cell_z, cell_y, cell_x]).T)
                    cell_df.columns = ['cell_name', 'z', 'y', 'x']
                if return_brain_region_assignments:
                    regions_df = pd.DataFrame(
                        np.array(preproc_hdf5[plane_key][f'preprocessed_data/fish00/{segmentation_type}_segmentation/region_mask_units_matrix_ants_ZBRAIN_registered']).T,
                        columns=regions)
                    cell_df = pd.concat([cell_df,  regions_df], axis=1)

                if do_segmentaion_shape_cleaning:
                    cell_df = cell_df.iloc[good_cells]
                    if return_brain_region_assignments: # here this also removes all cells not assigned to a region
                        cell_df = cell_df.loc[cell_df.iloc[:,5:].sum(axis=1) != 0]
                        plane_traces = plane_traces.loc[plane_traces.index.get_level_values('cell_name').isin(list(cell_df['cell_name']))]

                cell_position_df.append(cell_df)

            traces_df.append(plane_traces)

    traces_df = pd.concat(traces_df)
    traces_df.reset_index(drop=False, inplace=True)
    traces_df.drop(columns=columns_to_drop, inplace=True)
    traces_df.set_index(index_columns, inplace=True)

    if do_flipping_based_on_hemisphere:
        idx_df = traces_df.index.to_frame(index=False)
        idx_df['stimulus'] = idx_df.apply(map_ipsi_contra, axis=1)
        traces_df.index = pd.MultiIndex.from_frame(idx_df)
    traces_df = traces_df.sort_index()

    if return_cell_position_df:
        cell_position_df = pd.concat(cell_position_df)
        cell_position_df.set_index('cell_name', inplace=True)
        cell_position_df[['x', 'y', 'z']] = cell_position_df[['x', 'y', 'z']].apply(pd.to_numeric)

        return traces_df, cell_position_df
    else:
        return traces_df


def get_experiment_information(path):
    # return the experiment info as a dict
    path = pathlib.Path(path)
    if path.stem == 'experiment_information':
        exp_p = path
    elif (path / 'experiment_information.txt').exists():
        exp_p = path / 'experiment_information.txt'
    else:
        print('No experiment information file')
        return None

    info = {}
    with open(exp_p, 'r') as f:
        for l in f:
            l = l.strip().replace('\t', '').replace('"', '')
            if ':' in l:
                k, v = l.split(':', 1)
                info[k] = v
            else:
                if not hasattr(info[k], 'append'):
                    info[k] = [info[k]]
                info[k].append(l)
    return info
