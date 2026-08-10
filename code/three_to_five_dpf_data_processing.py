import pandas as pd
import numpy as np
from two_p_image_analysis_tools \
    import get_segmented_signals_multiindex_df, get_experiment_information
from general_util_functions import  pickle_save_object
import pathlib
from datetime import datetime
today = datetime.now().strftime('%Y-%m-%d')

df_dir = pathlib.Path(r".\data\3to5dpf")
full_traces_df = []
for p in df_dir.glob("*2026-06-23_trial-clustering*.tsv"):
    full_traces_df.append(pd.read_csv(p, sep="\t"))

full_traces_df.append(pd.read_csv(
    r".\data\WARP\2026-06-23_trial-clustering_dots-lumi-stimulus_k6-th03-frac03.tsv",
    sep="\t")
)
full_traces_df = pd.concat(full_traces_df)

id_to_age = {}
# for overnight imaging first 2/3 repeats get first age last 2 a day later
for k in ['20260305-000' , '20260305-001', '20260427-000', '20260427-001',
          ('20260427-002',(0,1)), ('20260427-003',(0,1)), ('20260504-004',(0,1)), ('20260504-005',(0,1))]:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '3dpf'
for k in ['20260306-000', ('20260427-002',(2,3,4)), ('20260427-003',(2,3)), ('20260504-004',(2,3)),
          ('20260504-005',(2,3)), ('20260505-006', (0,1,2)), ('20260505-007', (0,1,2)), ('20260511-008', (0,1,2))]:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '4dpf'
for k in ['20260307-001', '20260512-010', ('20260505-006', (3,4)),
          ('20260505-007', (3,4)), ('20260511-008', (3,4)), '20260423-002', '20260601-000', '20260601-001',
          '20260601-002', '20260601-003']:
    if isinstance(k, str):
        k = (k, (0,))
    id_to_age[k] = '5dpf'

age_array = np.zeros(len(full_traces_df)).astype(str)
for k,v in id_to_age.items():
    fish = k[0]
    reps = k[1]
    mask = np.bitwise_and(full_traces_df['fish_id'] == fish, full_traces_df['repeat'].isin(reps))
    age_array[mask] = v
full_traces_df['age'] = age_array

stim_dict = {x: x for x in full_traces_df['stimulus'].unique()}
for k in stim_dict.keys():
    if 'dots' in k:
        if 'lumi' in k:
            continue
        stim_dict[k] = 'lumi_off_' + k
    elif 'lumi' in k:
        stim_dict[k] = k + '_dots_off'
full_traces_df['stimulus'] = full_traces_df['stimulus'].map(stim_dict)
cell_to_age = {c:a for c,a in np.unique(full_traces_df.loc[:,['cell_name','age']].values.astype(str), axis=0)}



full_traces_df.set_index([
 'fish_id',
 'age',
 'region',
 'fine_region',
 'repeat',
 'z_plane',
 'cell_number',
 'cell_name',
 'stimulus',
], inplace=True)

cells_to_keep = [c for c, grouped_df in full_traces_df.groupby(level='cell_name')
                    if sum(grouped_df['reliability'] >= .3) >= len(grouped_df) / 2]


pickle_save_object((full_traces_df, cell_to_age),
                   df_dir/ f"{today}_3to5dpf_trace_data_full.pkl")

pickle_save_object((full_traces_df.loc[full_traces_df.index.get_level_values('cell_name').isin(cells_to_keep)]
                        ,cell_to_age), df_dir/ f"{today}_3to5dpf_trace_data_reliability_filtered.pkl")

# cell numbers for all cells
#%% get cell numbers
all_regions = []
functional_dir = pathlib.Path(r'Y:\M11 2P microscopes\Max W\master_thesis_imaging\3to5dfp\functional')
warp_dir = pathlib.Path(r'Y:\M11 2P microscopes\Max W\master_thesis_imaging\WARP_stimulus\functional')
warp_files = [p.parent for p in warp_dir.rglob('*2P_dots_lumi_*.py')]
files = warp_files + list(functional_dir.iterdir())
for file in files:
    _, regions = get_segmented_signals_multiindex_df(
        file,
        do_flipping_based_on_hemisphere=True,
        do_segmentaion_shape_cleaning=True,
        return_brain_region_assignments=True,
        registration_type="ZBRAIN"
    )
    if file in warp_files:
        info = get_experiment_information(file)
        plane = info['fish_comment'].split('#')[1][0]
        regions.reset_index(drop=False, inplace=True)
        regions['cell_name'] = [s.replace('z000', f'z00{plane}') for s in regions['cell_name']]
        regions.set_index('cell_name', inplace=True)
    all_regions.append(regions)
all_regions = pd.concat(all_regions)

coarse_region_column_lists = {r:[c for c in all_regions.columns if r in c] for r in ['Diencephalon', 'Mesencephalon', 'Rhombencephalon', 'Telencephalon']}
coarse_region_series = np.repeat(np.nan, len(all_regions)).astype(object)
for region,columns in coarse_region_column_lists.items():
    mask = all_regions.loc[:,columns].values.any(axis=1)
    coarse_region_series[mask] = region
all_regions = all_regions.drop(['Diencephalon -', 'Mesencephalon -', 'Rhombencephalon -', 'Telencephalon -'], axis=1)
fine_region_series = (all_regions.iloc[:, 3:].idxmax(axis=1).where(all_regions.iloc[:, 3:].any(axis=1), np.nan))
all_regions['fine_region'] = all_regions.index.get_level_values('cell_name').map(fine_region_series)
all_regions['region'] = coarse_region_series
all_regions = all_regions.loc[:,['region', 'fine_region', 'z', 'x', 'y']]
all_regions.reset_index(drop=False, inplace=True)
all_regions['fish_id'] = [s[1:13] for s in all_regions['cell_name']]
all_regions['repeat'] = [int(s.split('r')[1][:2]) for s in all_regions['cell_name']]
all_regions['z_plane'] = [int(s.split('z')[1][:3]) for s in all_regions['cell_name']]
age_column = np.zeros(len(all_regions)).astype(str)
for k,v in id_to_age.items():
    fish = k[0]
    reps = k[1]
    mask = np.bitwise_and(all_regions['fish_id'] == fish, all_regions['repeat'].isin(reps))
    age_column[mask] = v
all_regions['age'] = age_column
all_regions.set_index(['fish_id', 'age', 'repeat', 'z_plane', 'region', 'fine_region', 'cell_name'], inplace=True)

cell_numbers = all_regions.groupby(['age', 'fish_id', 'repeat', 'z_plane', 'region']).size()
cell_numbers_fine = all_regions.groupby(['age', 'fish_id', 'repeat', 'z_plane', 'fine_region']).size()
#
pickle_save_object(cell_numbers, rf'.\data\3to5dpf\{today}-cell_numbers_3to5dpf.pkl')
pickle_save_object(cell_numbers_fine, rf'.\data\3to5dpf\{today}-cell_numbers_3to5dpf_(fine_regions).pkl')

position_df = all_regions.loc[:,['z', 'x', 'y']]

pickle_save_object(position_df, rf".\data\3to5dpf\{today}_3to5dpf_cell_positions.pkl")
