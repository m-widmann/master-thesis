import numpy as np
import pandas as pd
import pathlib
from scipy.spatial import cKDTree
from two_p_image_analysis_tools import \
    get_segmented_signals_multiindex_df, load_contours_and_centroids, get_experiment_information
from general_util_functions import pickle_save_object
from cv2 import arcLength, contourArea
from copy import deepcopy

def check_shape_similarity(contour1, contour2):
    areas = np.array([contourArea(c) for c in [contour1, contour2]])
    frac_difference = areas[0]/areas[1]
    if frac_difference < .8 or frac_difference > 1.2:
        return False
    perimeters = np.array([arcLength(c, True) for c in [contour1, contour2]])
    frac_difference = perimeters[0]/perimeters[1]
    if frac_difference < .8 or frac_difference > 1.2:
        return False
    return True

def main(paths, max_distance):
    region_df = []
    for p in paths:
        traces, regions = get_segmented_signals_multiindex_df(
            p,
            return_cell_position_df=True,
            do_segmentaion_shape_cleaning=True,
            registration_type='volume')
        stim = next(p.glob('*py')).name
        if stim == 'WARP_stimulus.py':
            stim = 'WARP'
        else:
            stim = 'DOTS'
        regions['stimulus_type'] = stim
        regions.set_index('stimulus_type', append=True, inplace=True)
        region_df.append(regions)

    region_df = pd.concat(region_df)
    contours = [load_contours_and_centroids(next(p.glob('*preprocessed_data.h5')))[0] for p in paths]

    # build distance tree
    tree = cKDTree(region_df.to_numpy()[:,1:]) # just x,y assuming they are in the same plane
    sparse_dist_matrix = tree.sparse_distance_matrix(tree, max_distance=max_distance)
    dist_matrix = sparse_dist_matrix.toarray()
    highest_cell_index = len(region_df.xs("DOTS", level='stimulus_type'))

    index_to_cell_name_LUT = {}
    for i,cell in enumerate(region_df.index.get_level_values('cell_name')):
        index_to_cell_name_LUT[i] = int(cell.split('c')[1]) - 10000

    matches = []
    for i in range(highest_cell_index):
        # get the indices potential matches
        neighbor_indices = np.where(dist_matrix[i] != 0)[0]
        neighbor_indices = neighbor_indices[neighbor_indices > highest_cell_index]
        if len(neighbor_indices) == 0: continue
        # sort them by distance
        distances = dist_matrix[i, neighbor_indices]
        neighbor_order = np.argsort(distances)
        closest_neighbor = None
        for j in neighbor_order:
            n = neighbor_indices[j]
            if check_shape_similarity(
                    contours[0][index_to_cell_name_LUT[i]],
                    contours[1][index_to_cell_name_LUT[n]]):
                closest_neighbor = n
                continue
        if closest_neighbor is None: continue
        dist_matrix[:, closest_neighbor] = 0. # remove cell from distance matrix so it cannot be matched again
        matches.append((index_to_cell_name_LUT[i], index_to_cell_name_LUT[closest_neighbor]))

    matches = np.array(matches)
    print(f'Found {len(matches)} matches between the two segmentations.')
    return matches, contours

#%%
if __name__ == '__main__':
    base_dir = pathlib.Path(r'Y:\M11 2P microscopes\Max W\master_thesis_imaging\WARP_stimulus\functional')
    path_df = []
    for p in base_dir.iterdir():
        info = get_experiment_information(p)
        path_df.append([
            info['date_and_time'].split('T')[0].replace('-','') + '-' + info['experiment_ID'].zfill(3),
            info['fish_comment'].split('#')[1][0],
            p
        ])
    path_df = pd.DataFrame(path_df, columns=['fish_id', 'z_plane', 'path'])

    segementation_matches = {
        fish:{} for fish in path_df['fish_id'].unique()
    }
    contours_dict = deepcopy(segementation_matches)

    for (fish,plane),grouped_df in path_df.groupby(['fish_id','z_plane']):
        paths = [pathlib.Path(p) for p in grouped_df['path']]
        print(f'\n{fish}, plane{plane}')
        print(f'{paths[0].name}, {paths[1].name}')
        matches, contours = main(paths, 2)
        segementation_matches[fish][plane] = matches
        contours_dict[fish][plane] = contours
    pickle_save_object(segementation_matches, r'.data\segmentation_matches_dict.pkl')
