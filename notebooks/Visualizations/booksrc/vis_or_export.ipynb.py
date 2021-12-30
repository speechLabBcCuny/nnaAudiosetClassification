# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
'''This was initially designed for visuallizations but then used for exporting data.

    export_cli moved under the script dir, this is only visualization
    


'''

# %%
import pandas as pd
import datetime
import matplotlib.pylab as pl
from nna import visutils
import numpy as np
from pathlib import Path
from pathlib import Path


# %%
# CONFIGS
class pathMap():

    def __init__(self) -> None:
        scratch = '/scratch/enis/data/nna/'
        home = '/home/enis/projects/nna/'
        self.data_folder = scratch + 'database/'

#         self.exp_dir = '/home/enis/projects/nna/src/nna/exp/megan/run-3/'

        self.clipping_results_path = Path(scratch +
                                          'clipping_info/all_data_2021-02-10/')

        self.output_dir = scratch + 'real/'

        self.file_properties_df_path = self.data_folder + '/allFields_dataV5.pkl'
        # weather_cols=[]

        self.results_folder = home + 'results/'
        self.vis_output_path = self.results_folder + 'vis/182tahb6-V1/'


def setup_configs():
    pathmap = pathMap()

    config = {}
#     id2name = {}
#     id2name['V3-1-1-10'] = 'duck-goose-swan'
#     id2name['V3-1-1-7'] = 'songbirds'

    id2name={'multi9-V1-1-0-0': 'biophony',
     'multi9-V1-1-1-0': 'bird',
     'multi9-V1-1-1-10': 'songbirds',
     'multi9-V1-1-1-7': 'duck-goose-swan',
     'multi9-V1-0-0-0': 'anthrophony',
     'multi9-V1-1-3-0': 'insect',
     'multi9-V1-1-1-8': 'grouse-ptarmigan',
     'multi9-V1-0-2-0': 'aircraft',
     'multi9-V1-3-0-0': 'silence'}

    config['id2name'] = id2name
    config['input_data_freq'] = '10S'
    # FREQS to reduce results
    config['output_data_freq'] = '270min'

    return pathmap, config



def setup(args, pathmap, region_location):

    file_properties_df = pd.read_pickle(pathmap.file_properties_df_path)

    #important to keep them in order
    file_properties_df.sort_values(by=['timestamp'], inplace=True)

    # delete older than 2016
    fromtime = datetime.datetime(2016, 1, 1, 0)
    file_properties_df = file_properties_df[
        file_properties_df.timestamp >= fromtime]

    if not region_location:
        # region_location = [('anwr','49'),('prudhoe','11'),('prudhoe','26')]
        region_location = tuple(
            sorted(
                set(
                    zip(file_properties_df.region.values,
                        file_properties_df.locationId.values))))

    return region_location,file_properties_df
# %%
def sigmoid(data):
    return 1 / (1 + np.exp(-data))


# %%
def vis_preds_with_clipping(region_location, config, file_properties_df,
                            pathmap):

    cmap = pl.cm.tab10
    aCmap = cmap
    my_cmaps = visutils.add_normal_dist_alpha(aCmap)

    no_result = {}

    for region, location_id in region_location:
        # print(region, all_regions.index(region),'location_id',location_id)
        filtered_files = file_properties_df[file_properties_df.region == region]
        filtered_files = filtered_files[filtered_files.locationId ==
                                        location_id]
        filtered_files = filtered_files[filtered_files.durationSec > 0]

        no_result_paths = visutils.vis_preds_with_clipping(
            region,
            location_id,
            filtered_files,
            config['input_data_freq'],
            config['output_data_freq'],
            config['id2name'].keys(),
            my_cmaps,
            pathmap.output_dir,
            pathmap.clipping_results_path,
            pathmap.vis_output_path,
            config['id2name'],
            clipping_threshold=1.0,
            pre_process_func=sigmoid)
        if no_result_paths:
            no_result[(region, location_id)] = no_result_paths

    return no_result


def main(args):

    pathmap, config = setup_configs()
    location = args.location
    region = args.region
    if region!='' and location !='':
        region_location = [(region,location)]
    else:
        print('Region and location are not given, we will do all of them.')
        region_location = None
    region_location,file_properties_df = setup(args, pathmap,region_location)

    vis_preds_with_clipping(region_location, config, file_properties_df,
                                pathmap)

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--region',
        help='region such as anwr or stinchcomb etc',
        required=False, default='')
    parser.add_argument(
        '--location',
        help='location_id such as 11 or 14-Rocky etc',
        required=False, default='')
    args = parser.parse_args()

    main(args)

# %%

# # %%
# output_folder_path = '/home/enis/projects/nna/results/ExternalProject/megan/export_raw_v7'
# merge_folder = '/home/enis/projects/nna/results/ExternalProject/megan/merge_folder_v7'


# # %%
# def export_raw_results_2_csv(region_location, config, file_properties_df,
#                              pathmap):

#     for region, location_id in region_location:
#         print(region, location_id)
#         filtered_files = file_properties_df[file_properties_df.region == region]
#         filtered_files = filtered_files[filtered_files.locationId ==
#                                         location_id]
#         filtered_files = filtered_files[filtered_files.durationSec > 0]

#         csv_files_written, no_result_paths = visutils.export_raw_results_2_csv(
#             output_folder_path,
#             config['id2name'].keys(),
#             filtered_files,
#             input_data_freq='10S',
#             output_data_freq='10S',
#             channel=1,
#             result_files_folder=result_path,
#             prob2binary_flag=False)
#         print(len(no_result_paths))


# %%

# # %%
# # FAulty files
# # /tank/data/nna/real/anwr/41/2019/S4A10273_20190705_105723.flac

# # %%
# # !zip -r export_raw.zip export_raw/

# # %%
# get_ipython().system(
#     'cd /home/enis/projects/nna/results/ExternalProject/megan/')

# # %%
# merge_folder = '/home/enis/projects/nna/results/ExternalProject/megan/merge_folder_v7'

# # %%
# Path(merge_folder).mkdir(exist_ok=True)

# # %%
# import csv
# import glob

# for i in region_location:
#     # location_csv_files=glob.glob(f'export_raw_v6/{i[1]}*')
#     location_csv_files = glob.glob(f'{output_folder_path}/{i[1]}*')
#     locationCsv_dict = {}
#     timeLines = set()
#     Lines = {}
#     lineDicts = {}
#     for csvf in location_csv_files:
#         with open(csvf, 'r') as csvf_handle:
#             rr = csv.reader(csvf_handle)
#             lines = list(rr)
#             headers, lines = lines[0], lines[1:]
#             # locationCsv_dict[csvf] = (headers[1],lines,)
#             # timeLine = [i[0] for i in lines]
#             # valueLine = [i[1] for i in lines]
#             lineDict = dict(lines)
#             lineDicts[headers[1]] = lineDict

#     # for i,n in sorted(list(zip(timeLines,location_csv_files))):
#     #     print(len(i),n)
#     # print('***END**')
#     print(i)
#     #     'CABLE', 'RUNNINGWATER', 'INSECT', 'RAIN', 'WATERBIRD', 'WIND', 'SONGBIRD',
#     # 'AIRCRAFT'
#     aa = pd.DataFrame(data=lineDicts, columns=lineDicts.keys())
#     aa.index.name = 'TIMESTAMP'
#     aa.to_csv(f'{merge_folder}/{'_'.join(i)}.csv')
#     del aa

# # %%
# get_ipython().system('pwd')

# # %%
# get_ipython().system(
#     'zip -r /home/enis/projects/nna/results/ExternalProject/megan/csv_version_merged.zip merge_folder_v7/'
# )

# # %%
# get_ipython().system(
#     'rclone copy csv_version_merged.zip 1Drive:/Data_sharing/NNA/ExternalProject/megan/'
# )

# # %%
# get_ipython().system('pwd')

# # %%

# # %%

# # %%
# # !ls -alh /home/enis/projects/nna/src/scripts/clipping_output/

# # %%
# # no_result_paths[0]

# # %%
# get_ipython().system('ls /home/enis/projects/nna/results/vis/testtestV3')


# # %%
# def load_clipping_2dict(clippingResultsPath,
#                         selected_areas,
#                         selected_tag_name,
#                         threshold: float = 1.0):
#     gathered_results_perTag = {selected_tag_name: {}}
#     gathered_results = {}
#     selected_areas_files = {}

#     gathered_results_perTag[selected_tag_name].update(resultsDict)
#     return gathered_results_perTag


# # %%
# from pathlib import Path
# file_properties_df = pd.read_pickle(
#     '../../../data/prudhoeAndAnwr4photoExp_dataV1.pkl')

# # %%
# import numpy as np
# clippingResultsPath = '/home/enis/projects/nna/src/scripts/clipping_output/'
# selected_areas = list(range(11, 51))
# clipping_threshold_str = '1,0'
# results = {}
# for i, area in enumerate(selected_areas):
#     print(i)
#     fileName = (clippingResultsPath + str(area) +
#                 f'_{clipping_threshold_str}.pkl')
#     resultsDict = np.load(fileName, allow_pickle=True)
#     resultsDict = resultsDict[()]
#     results[str(area)] = resultsDict.copy()

# # %%
# import datetime
# from nna import fileUtils

# # %%
# for location_id in results.keys():
#     csv_output = [['file_name', 'timestamp', 'channel-1', 'channel-2']]

#     file_names = list(results[location_id].keys())
#     timestamps = []
#     for file_name in file_names:
#         row = file_properties_df.loc[Path(file_name)]
#         timestamps.append(row.timestamp)
#     name_timestamp = list(zip(timestamps, file_names))
#     name_timestamp.sort()

#     for timestamp, file_name in name_timestamp:
#         res_array = results[location_id][file_name]
#         for i, (c1, c2) in enumerate(res_array):
#             time_str = ((timestamp + datetime.timedelta(seconds=10 * i)
#                         ).strftime('%Y-%m-%d_%H:%M:%S'))
#             line = [file_name, time_str, f'{c1:.4f}', f'{c2:.4f}']
#             csv_output.append(line)

#     fileUtils.save_to_csv(
#         './csv_version/' + location_id + '_clipping_frequency.csv', csv_output)

# # %%
# get_ipython().system(
#     'rm -r /home/enis/projects/nna/notebooks/Visualizations/booksrc/csv_version/.ipynb_checkpoints/ '
# )

# # %%
# get_ipython().system('mkdir csv_version')

# # %%
# get_ipython().system('zip -r csv_version3.zip csv_version/')

# # %%
# # !rm /home/enis/projects/nna/notebooks/Visualizations/booksrc/csv_version/*

# # %%
# # !head /home/enis/projects/nna/notebooks/Visualizations/booksrc/csv_version/11_clipping_frequency.csv

# # %%
# get_ipython().system(
#     'du -h /home/enis/projects/nna/notebooks/Visualizations/booksrc/')

# # %%
