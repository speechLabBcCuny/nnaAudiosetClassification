'''
* applies sigmoid by default
* input directory is hard coded TODO
* raw_folder4_merge and merged_folder_path are hard coded TODO
'''
# %%
import pandas as pd
from nna import visutils
import datetime
from pathlib import Path
import numpy as np

import glob
import csv

# %%


# %%
def setup_configs(args):
    pathmap = pathMap(args)

    config = {}
    id2name = {}
    versiontag = args.versiontag

    id2name = {
        '1-0-0': 'biophony',
        '1-1-0': 'bird',
        '1-1-10': 'songbirds',
        '1-1-7': 'duck-goose-swan',
        '0-0-0': 'anthrophony',
        '1-3-0': 'insect',
        '1-1-8': 'grouse-ptarmigan',
        '0-2-0': 'aircraft',
        '3-0-0': 'silence'
    }
    # # model # 3rk9ayjc, validation
    # # label               F-1 score    threshold
    # # ----------------  -----------  -----------
    # # biophony                0.942        0.43
    # # insect                  0.531        0.402
    # # bird                    0.917        0.425
    # # songbirds               0.775        0.376
    # # duck-goose-swan         0.366        0.548
    # # anthrophony             0.514        0.548
    # # grouse-ptarmigan        0.571        0.899
    # # aircraft                0.571        0.764
    # # silence                 0.358        0.561
    #     prob2binary_thresholds_dict = {
    #         '1-0-0': 0.43,
    #         '1-1-0': 0.425,
    #         '1-1-10': 0.376,
    #         '1-1-7': 0.548,
    #         '0-0-0': 0.548,
    #         '1-3-0': 0.402,
    #         '1-1-8': 0.899,
    #         '0-2-0': 0.764,
    #         '3-0-0': 0.561,
    #     }

    # model 1jvt1zva, validation
    # label               F-1 score    threshold
    # ----------------  -----------  -----------
    # biophony                0.846    0.301301
    # insect                  0.897    0.116116
    # bird                    0.907    0.267267
    # songbirds               0.879    0.214214
    # duck-goose-swan         0.929    0.0970971
    # anthrophony             0.892    0.154154
    # grouse-ptarmigan        0.933    0.414414
    # aircraft                0.939    0.500501
    # silence                 0.913    1

    # prob2binary_thresholds_dict = {
    #     '1-0-0': 0.301301,
    #     '1-1-0': 0.267267,
    #     '1-1-10': 0.214214,
    #     '1-1-7': 0.0970971,
    #     '0-0-0': 0.154154,
    #     '1-3-0': 0.116116,
    #     '1-1-8': 0.414414,
    #     '0-2-0': 0.500501,
    #     '3-0-0': 1,
    # }

    # model yfitloiq, test
    # label               F-1 score    threshold
    # ----------------  -----------  -----------
    # biophony                0.909    0.796797
    # insect                  0.931    0.394394
    # bird                    0.957    0.714715
    # songbirds               0.951    0.471471
    # duck-goose-swan         0.952    0.122122
    # anthrophony             0.976    0.702703
    # grouse-ptarmigan        0.945    0.60961
    # aircraft                0.976    0.211211
    # silence                 0.883    0.0620621

    prob2binary_thresholds_dict = {
        '1-0-0': 0.7967967967967968,
        '1-1-0': 0.7147147147147147,
        '1-1-10': 0.47147147147147145,
        '1-1-7': 0.12212212212212212,
        '0-0-0': 0.7027027027027027,
        '1-3-0': 0.3943943943943944,
        '1-1-8': 0.6096096096096096,
        '0-2-0': 0.2112112112112112,
        '3-0-0': 0.062062062062062065
    }

    generic_id2name = list(id2name.items())
    id2name = {}
    for k, v in generic_id2name:
        id2name[f'{versiontag}-{k}'] = v
        prob2binary_thresholds_dict[
            f'{versiontag}-{k}'] = prob2binary_thresholds_dict[k]
        del prob2binary_thresholds_dict[k]

    # print(prob2binary_thresholds_dict)
    config['prob2binary_threshold'] = prob2binary_thresholds_dict

    config['id2name'] = id2name
    config['input_data_freq'] = args.input_data_freq
    # FREQS to reduce results
    config['output_data_freq'] = args.output_data_freq
    config['prob2binary'] = args.prob2binary
    config['versiontag'] = args.versiontag
    return pathmap, config


class pathMap():

    def __init__(self, args) -> None:
        scratch = '/scratch/enis/data/nna/'
        home = '/home/enis/projects/nna/'

        self.data_folder = home + 'data/'

        self.clipping_results_path = Path(
            scratch + 'clipping_info/all-merged_2021-12-24/')

        # source of the input files
        self.output_dir = scratch + 'real/'

        self.input_dir = scratch + 'real/'
        # self.input_dir = scratch + 'audio_collars/'

        if args.output_folder == '':
            out_dir = Path(self.output_dir)
        else:
            out_dir = Path(args.output_folder)
        # self.merge_folder = (out_dir.parent / (out_dir.stem + '-merged'))
        self.merge_folder = (out_dir / args.region / args.location)
        self.export_output_path = (out_dir / args.region / args.location)

        self.file_properties_df_path = args.file_database

        self.export_output_path = Path(self.export_output_path)
        self.export_output_path.mkdir(parents=True, exist_ok=True)

        self.raw_folder4_merge = '/scratch/enis/data/nna/results/csv_export_raw'


def setup(args, pathmap, region_location):
    '''
    setup the pathmap and config

    Parameters
    ----------
    args : dict
        arguments from command line.
    pathmap : pathMap class
        paths required for the script.
    region_location : tuple
        list of region and location ids.
    '''
    file_properties_df = pd.read_pickle(pathmap.file_properties_df_path)

    #important to keep them in order
    file_properties_df.sort_values(by=['timestamp'], inplace=True)

    if not region_location:
        region_location = tuple(
            sorted(
                set(
                    zip(file_properties_df.region.values,
                        file_properties_df.locationId.values))))

    return region_location, file_properties_df


# %%
def sigmoid(data):
    return 1 / (1 + np.exp(-data))


# %%


def filter_files(file_properties_df, region, location_id):
    '''
    filter the files based on region and location id

    Also make sure all files are longer than 0 seconds.

    Parameters
    ----------
    file_properties_df : pandas.DataFrame
        dataframe containing the file properties.
    region : str
        region id
    location_id : str
        location id
    
    Returns
    -------
    filtered_files : pandas.DataFrame
        filtered files based on region and location id.
    years : numpy.ndarray
        years of the filtered files.
    years_set : set
        set of years of the filtered files.
    '''
    # print(region, all_regions.index(region),'location_id',location_id)

    filtered_files = file_properties_df[file_properties_df.region == region]
    filtered_files = filtered_files[filtered_files.locationId == location_id]
    filtered_files = filtered_files[filtered_files.durationSec > 0]

    # year is an integer here
    years = np.array([i.year for i in filtered_files.timestamp])

    return filtered_files, years


def export_files2table_dict(region_location, config, file_properties_df,
                            pathmap):

    no_result = {}
    results = {}
    for region, location_id in region_location:
        filtered_files, years = filter_files(file_properties_df, region,
                                             location_id)
        for year in set(years):
            filtered_by_year = filtered_files[years == year]

            df_dict, no_result_paths = visutils.file2TableDict(
                config['id2name'].keys(),
                filtered_by_year,
                config['input_data_freq'],
                config['output_data_freq'],
                prob2binary_threshold=config['prob2binary_threshold'],
                result_path=pathmap.input_dir,
                prob2binary_flag=config['prob2binary'],
                pre_process_func=sigmoid,
            )

            no_result[(region, location_id, year)] = no_result_paths
            results[(region, location_id, year)] = df_dict.copy()

    return results, no_result


def export_raw_results_2_csv(region_location, config, file_properties_df,
                             pathmap):

    no_result = {}
    results = {}
    for region, location_id in region_location:
        filtered_files, years = filter_files(file_properties_df, region,
                                             location_id)
        for year in set(years):
            filtered_by_year = filtered_files[years == year]

            output_path = (pathmap.export_output_path / str(year))
            output_path.mkdir(exist_ok=True, parents=True)

            csv_files_written, no_result_paths = visutils.export_raw_results_2_csv(
                output_path,
                config['id2name'].keys(),
                filtered_by_year,
                config['input_data_freq'],
                config['output_data_freq'],
                channel=1,
                result_files_folder=pathmap.input_dir,
                prob2binary_flag=config['prob2binary'],
                pre_process_func=sigmoid)

            no_result[(region, location_id, year)] = no_result_paths
            results[(region, location_id, year)] = csv_files_written

    return results, no_result


def merge_raw_csv_files(versiontag, region_location, raw_folder4_merge,
                        merge_folder, id2name):
    # assert str(raw_folder4_merge[-1]) == '/'
    assert len(region_location) == 1
    region, location = region_location[0]
    # for region, location in region_location:
    #     # location_csv_files=glob.glob(f'export_raw_v6/{i[1]}*')
    search_folder = f'{str(raw_folder4_merge)}/{versiontag}/{region}/{location}/*'
    year_folders = glob.glob(search_folder)
    # print(search_folder)
    for year in year_folders:
        line_dicts = {}

        csv_files = glob.glob(year + '/*')
        # print(csv_files)
        for csv_file in csv_files:
            with open(csv_file, 'r') as csvf_handle:
                rr = csv.reader(csvf_handle)
                lines = list(rr)
                headers, lines = lines[0], lines[1:]
                line_dict = dict(lines)  # type: ignore
                line_dicts[headers[1]] = line_dict

        merged_into_dataframe = pd.DataFrame(data=line_dicts,
                                             columns=sorted(
                                                 list(line_dicts.keys())))
        merged_into_dataframe = merged_into_dataframe.rename(columns=id2name,)
        merged_into_dataframe.index.name = 'TIMESTAMP'
        year_str = Path(year).stem
        output_csv_path = (merge_folder / year_str / (f'{versiontag}.csv'))
        Path(output_csv_path).parent.mkdir(exist_ok=True, parents=True)
        merged_into_dataframe.to_csv(output_csv_path)
        del merged_into_dataframe


def get_cached_file_name(config, df_type='prob'):

    csv_file_name_parts = (config['versiontag'],
                           f'prob2binary={config["prob2binary"]}',
                           f'output-data-freq={config["output_data_freq"]}',
                           df_type)
    csv_file_name = '_'.join(csv_file_name_parts)

    csv_file_name = csv_file_name + '.csv'

    return csv_file_name


def df_export_csv(results, pathmap, config):
    files = []
    for region, location, year in results.keys():
        df_count, df_sums = results[(region, location, year)][location]
        df_prob = df_sums / df_count

        output_path = (pathmap.export_output_path / str(year))
        output_path.mkdir(exist_ok=True, parents=True)

        col_names = [config['id2name'][col] for col in list(df_prob.columns)]
        for df, name_str in ((df_sums, 'sums'), (df_count, 'counts'), (df_prob,
                                                                       'prob')):
            csv_file_name = get_cached_file_name(config, df_type=name_str)
            filename = str(output_path / csv_file_name)
            df.to_csv(filename,
                      index_label='TimeStamp',
                      header=col_names,
                      float_format='%.3f',
                      date_format='%Y-%m-%d_%H:%M:%S')
            print(filename)
            files.append(csv_file_name)

    return files


def main(args):
    '''
    Main function to run the export

    There are two modes of operation:
    1. Export raw results to csv: 
    2. Merge raw results to csv
    3. Export results to csv
    4. Merge results to csv
    
    '''

    pathmap, config = setup_configs(args)
    location = args.location
    region = args.region
    raw_export = args.raw_export
    raw_csv_merge = args.raw_csv_merge
    # versiontag = args.versiontag
    # prob2binary = args.prob2binary
    # output_data_freq = args.output_data_freq

    if region != '' and location != '':
        region_location = [(region, location)]
    else:
        print('Region and location are not given, exporting for\
                 all regions and locations')
        region_location = None

    region_location, file_properties_df = setup(args, pathmap, region_location)
    if raw_export:
        results, no_result = export_raw_results_2_csv(region_location, config,
                                                      file_properties_df,
                                                      pathmap)
        return results, no_result, list(results.values())
    elif raw_csv_merge:

        merge_raw_csv_files(config['versiontag'], region_location,
                            pathmap.raw_folder4_merge, pathmap.merge_folder,
                            config['id2name'])
    else:
        results, no_result = export_files2table_dict(region_location, config,
                                                     file_properties_df,
                                                     pathmap)

        files = df_export_csv(results, pathmap, config)
        return results, no_result, files


# %%

# %%

# %%
# class Args():
#     def __init__(self,region,location,output_folder,file_database):
#         self.region=region
#         self.location=location
#         self.file_database = file_database
#         self.output_folder = output_folder

# args= Args('anwr','49','./','/scratch/enis/data/nna/database/allFields_dataV4.pkl',)

# results,no_result,files = main(args)

# %%
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--region',
        help='region-location_id such as anwr or stinchcomb etc',
        required=True)
    parser.add_argument('--location',
                        help='location_id such as 11 or 14-Rocky etc',
                        required=True)
    parser.add_argument(
        '--versiontag',
        help='tag of the model created predictions, like multi9-V1 ',
        required=True)

    parser.add_argument(
        "--raw_export",
        type=str2bool,
        required=False,
        help=
        'if set True, outputs are not aggregated, they are exported as it is,\
         False by default, reuqires that output and input data frequencies are same',
        default=False)

    parser.add_argument(
        "--prob2binary",
        type=str2bool,
        required=True,
        help=
        'to average probabilities or binary values per segment, True or False')

    parser.add_argument(
        '--output_data_freq',
        help='which time range to aggreate results, ex: 270min,10S,1min',
        required=True)
    parser.add_argument(
        '--input_data_freq',
        help='how long audio corresponding to single prediction',
        default='10S')

    parser.add_argument('-O',
                        '--output_folder',
                        help='where to save outputs, default is scratch folder',
                        default='')
    parser.add_argument(
        '-d',
        '--file_database',
        help='path to file_properties_df_path',
        required=True,
        #default='/scratch/enis/data/nna/database/allFields_dataV6.pkl'
    )
    parser.add_argument("--raw_csv_merge",
                        type=str2bool,
                        required=False,
                        help='if set True, raw csv exports are merged, check \
         False by default, reuqires that output and input data frequencies are same',
                        default=False)

    args = parser.parse_args()

    main(args)
