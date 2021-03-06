'''
* applies sigmoid by default
'''
# %%
import pandas as pd
from nna import visutils
import datetime
from pathlib import Path
import numpy as np

# %%


# %%
def setup_configs(args):
    pathmap = pathMap(args)

    config = {}
    id2name = {}
    versiontag = args.versiontag

    id2name={'1-0-0': 'biophony',
     '1-1-0': 'bird',
     '1-1-10': 'songbirds',
     '1-1-7': 'duck-goose-swan',
     '0-0-0': 'anthrophony',
     '1-3-0': 'insect',
     '1-1-8': 'grouse-ptarmigan',
     '0-2-0': 'aircraft',
     '3-0-0': 'silence'}

    generic_id2name = list(id2name.items())
    id2name = {}
    for k, v in generic_id2name:
        id2name[f'{versiontag}-{k}'] = v

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
            scratch + 'clipping_info/all-merged_2021-02-10/')

        # source of the input files
        self.output_dir = scratch + 'real/'
        self.input_dir = scratch + 'real/'

        if args.output_folder != '':
            out_dir = Path(args.output_folder)
            out_dir = (out_dir / args.region / args.location / 'aggregated')
            self.export_output_path = out_dir
        else:
            out_dir = Path(args.output_folder)
            out_dir = (out_dir / args.region / args.location /
                       'aggregated')
            self.export_output_path = out_dir

        self.file_properties_df_path = args.file_database

        self.export_output_path = Path(self.export_output_path)


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

    return region_location, file_properties_df


# %%
def sigmoid(data):
    return 1 / (1 + np.exp(-data))


# %%
def export_files2table_dict(region_location, config, file_properties_df,
                            pathmap):

    no_result = {}
    results = {}
    for region, location_id in region_location:
        # print(region, all_regions.index(region),'location_id',location_id)
        filtered_files = file_properties_df[file_properties_df.region == region]
        filtered_files = filtered_files[filtered_files.locationId ==
                                        location_id]
        filtered_files = filtered_files[filtered_files.durationSec > 0]

        df_dict, no_result_paths = visutils.file2TableDict(
            config['id2name'].keys(),
            filtered_files,
            config['input_data_freq'],
            config['output_data_freq'],
            result_path=pathmap.input_dir,
            prob2binary_flag=config['prob2binary'],
            pre_process_func=sigmoid,
        )

        no_result[(region, location_id)] = no_result_paths

        results[(region, location_id)] = df_dict.copy()

    return results, no_result


def df_export_csv(results, pathmap, config):
    files = []
    for region, location in results.keys():
        df_count, df_sums = results[(region, location)][location]
        df_prob = df_sums / df_count

        csv_file_name = "_".join(
            (config['versiontag'], f'prob2binary={config["prob2binary"]}',
             f'output-data-freq={config["output_data_freq"]}'))

        pathmap.export_output_path.mkdir(exist_ok=True, parents=True)

        csv_file_name = pathmap.export_output_path / csv_file_name

        col_names = [config['id2name'][col] for col in list(df_prob.columns)]
        for df, name_str in ((df_sums, 'sums'), (df_count, 'counts'), (df_prob,
                                                                       'prob')):
            filename = str(csv_file_name) + f'_{name_str}.csv'
            df.to_csv(filename,
                      index_label="TimeStamp",
                      header=col_names,
                      float_format='%.3f',
                      date_format='%Y-%m-%d_%H:%M:%S')

        files.append(csv_file_name)

    return files


def main(args):

    pathmap, config = setup_configs(args)
    location = args.location
    region = args.region
    # versiontag = args.versiontag
    # prob2binary = args.prob2binary
    # output_data_freq = args.output_data_freq

    if region != '' and location != '':
        region_location = [(region, location)]
    else:
        print('Region and location are not given, we will do all of them.')
        region_location = None

    region_location, file_properties_df = setup(args, pathmap, region_location)

    results, no_result = export_files2table_dict(region_location, config,
                                                 file_properties_df, pathmap)

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

# args= Args('anwr','49','./','/home/enis/projects/nna/data/allFields_dataV4.pkl',)

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
        required=False,
        default='/home/enis/projects/nna/data/allFields_dataV4.pkl')

    args = parser.parse_args()

    main(args)
