''' Labeling data using a model for training another one.'''

import glob
from pathlib import Path
import csv
from datetime import datetime
from datetime import timedelta
from copy import deepcopy

import pandas as pd

from nna.fileUtils import find_filesv2
from nna.labeling_utils import splitmp3


def get_csv_into_pd(merged_out_dir, versiontag, region, location, year):

    csv_path = (f'{merged_out_dir}{versiontag}/{region}/' +
                f'{location}/{year}/{versiontag}.csv')
    # print(csv_path)
    pred_for_year = pd.read_csv(csv_path)
    pred_for_year['sum'] = pred_for_year.sum(axis=1)
    pred_for_year['region'] = region
    pred_for_year['location'] = location

    return pred_for_year

def load_all_rawcsv_as_df(reg_loc,merged_out_dir,versiontag,):
    '''
    Iterate all files, load with get_csv_into_pd
    '''
    preds_list= []
    for region, location in reg_loc:
        year_folders = glob.glob(
            f'{merged_out_dir}{versiontag}/{region}/{location}/*')
        for year in year_folders:
            year = (Path(year).stem)
            pred_for_year = get_csv_into_pd(merged_out_dir, versiontag, region,
                                            location, year)
            preds_list.append(pred_for_year.copy())

    preds_df=pd.concat(preds_list)

    return preds_df

def pick_confidents(pred_for_year, label_to_sample, confidence_threshold,
                    related_cols, count_from_each_year,
                    unrelated_label_thresholds):
    '''
    Filter by label's confidence, sum of other's confidence, and per year

    '''
    pred_confident = filterby_label_confidence(pred_for_year, label_to_sample,
                                               confidence_threshold)

    pred_confident = filter_other_high_preds(pred_confident, label_to_sample,
                                             related_cols,
                                             unrelated_label_thresholds)

    pred_confident = pred_confident.sort_values(by='sum', ascending=True)
    pred_confident = pred_confident.iloc[:count_from_each_year]
    return pred_confident


def filterby_label_confidence(pred_for_year, label_to_sample,
                              confidence_threshold):

    pred_confident = pred_for_year[
        pred_for_year[label_to_sample] > confidence_threshold]

    return pred_confident


def filter_other_high_preds(
    pred_confident,
    targe_label,
    related_cols,
    labels_threholds=None,
):
    '''
        Filter sample out if other labels has high confidence.
    '''
    if labels_threholds is None:
        labels_threholds = {'default':'0.5'}

    unrelated_ones = pred_confident.drop(columns=related_cols[targe_label])
    unrelated_ones = unrelated_ones.drop(columns=[
        'sum',
        'region',
        'location',
        'TIMESTAMP',
    ])
    if 'silence' in unrelated_ones.columns:
        unrelated_ones = unrelated_ones.drop(columns=['silence'])

    # remove samples that has other unrelated sounds
    for col in unrelated_ones.columns:
        threshold = labels_threholds.get(col,labels_threholds['default'])
        pred_confident = pred_confident[unrelated_ones[col] < threshold]
    # pred_confident = pred_confident[unrelated_ones.max(
    #     axis=1) < unrelated_label_threshold]

    return pred_confident


def load_filter_csv(new_data_classes_counts, reg_loc_per_set, merged_out_dir,
                    versiontag, confidence_threshold, related_cols,
                    count_from_each_year, unrelated_label_thresholds):
    '''
    Iterate all files, load with get_csv_into_pd and filter with pick_confidents

        This one creates a dict of df for each label, filters each one by
        confidence

    '''

    confident_preds_dict = {i: [] for i in new_data_classes_counts.keys()}

    for region, location in reg_loc_per_set[0]:
        print(region, location)
        # location_csv_files=glob.glob(f'export_raw_v6/{i[1]}*')
        year_folders = glob.glob(
            f'{merged_out_dir}{versiontag}/{region}/{location}/*')
        #     print(f'{merged_out_dir}{versiontag}/{region}/{location}/*')
        for year in year_folders:
            year = (Path(year).stem)

            pred_for_year = get_csv_into_pd(merged_out_dir, versiontag, region,
                                            location, year)

            for label_to_sample in new_data_classes_counts.keys():

                pred_confident = pick_confidents(pred_for_year, label_to_sample,
                                                 confidence_threshold,
                                                 related_cols,
                                                 count_from_each_year,
                                                 unrelated_label_thresholds)
                confident_preds_dict[label_to_sample].append(
                    pred_confident.copy())
                print(
                    len(pred_confident),
                    label_to_sample,
                )
    return confident_preds_dict


def pick_clips_equally_from_locations(new_data_classes_counts,
                                      confident_preds_dict):
    '''
        Get equal # of samples from each location.

            If a location have less than we need, we just get less samples
    '''

    confident_preds_dict_sampled = {
        i: [] for i in new_data_classes_counts.keys()
    }
    for label_to_sample, count in new_data_classes_counts.items():
        site_count = len(confident_preds_dict[label_to_sample])
        count_per_site = (count // site_count) + 1
        for site_data in confident_preds_dict[label_to_sample]:
            site_data_short = site_data.iloc[:count_per_site]
            confident_preds_dict_sampled[label_to_sample].append(
                site_data_short)
    return confident_preds_dict_sampled


def merge_picked_sampels(new_data_classes_counts, confident_preds_dict_sampled):
    '''
        Merge samples for a label into a single dataframe.

    '''
    confident_preds_dict_sampled_merged = {}
    for label_to_sample in new_data_classes_counts.keys():
        confident_preds_dict_sampled_merged[label_to_sample] = pd.concat(
            confident_preds_dict_sampled[label_to_sample])

    return confident_preds_dict_sampled_merged


def days_minutes_seconds(td):
    #     hours = td.seconds//3600
    return td.days, (td.seconds // 60), td.seconds % 60


def relative_time(file_start_time, clip_start_time, length):
    '''given timestamp objects return mm.ss string
    '''
    relative_start = clip_start_time - file_start_time
    relative_end = relative_start + timedelta(seconds=length)

    s_days, s_minutes, s_seconds = days_minutes_seconds(relative_start)
    e_days, e_minutes, e_seconds = days_minutes_seconds(relative_end)

    if e_days > 0 or s_days > 0:
        print('##### Error Error Error relative_time #########')
        # print(row)

    s_str = f'{s_minutes}.{s_seconds}'

    e_str = f'{e_minutes}.{e_seconds}'

    return s_str, e_str


def cut_corresponding_clip(clip_start_time, file_start_time, length,
                           output_folder, orig_row):

    s_str, e_str = relative_time(file_start_time, clip_start_time, length)

    Path(output_folder).mkdir(exist_ok=True, parents=True)

    out_file = splitmp3(str(orig_row.name),
                        output_folder,
                        s_str,
                        e_str,
                        backend_path='/home/enis/sbin/ffmpeg',
                        outputSuffix='.wav')

    return out_file


def generate_new_dataset(new_data_classes_counts,
                         confident_preds_dict_sampled_merged, versiontag,
                         split_out_path, file_properties_df, upper_taxo_links):
    # create dataset csv from picked rows
    # get related info and clip wav files

    length = 10
    buffer = 0

    new_dataset_csv = []
    not_found_rows = []

    for label_to_sample in new_data_classes_counts.keys():
        for index, row_old in confident_preds_dict_sampled_merged[
                label_to_sample].iterrows():
            row = {}
            start_time = row_old['TIMESTAMP']
            start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')

            output = find_filesv2(row_old['location'],
                                  row_old['region'],
                                  start_time,
                                  None,
                                  length,
                                  buffer,
                                  file_properties_df,
                                  only_continuous=True)
            (sorted_filtered, start_time, _, _, _) = output

            if len(sorted_filtered.index) == 0:
                print('Not Found')
                not_found_rows.append((label_to_sample, index, row_old))
                print(label_to_sample, index)
                continue

            if len(sorted_filtered.index) > 1:
                print('Too many')

            location = row_old['location']
            region = row_old['region']

            orig_row = sorted_filtered.iloc[0]
            row['data_version'] = 'V2.1.1'
            row['Annotator'] = versiontag
            row['Site ID'] = location
            row['File Name'] = str(orig_row.name)
            row['Start Time'] = datetime.strftime(
                start_time,  # type: ignore
                '%H:%M:%S.%f')
            row['End Time'] = datetime.strftime(
                start_time +  # type: ignore
                timedelta(seconds=length),
                '%H:%M:%S.%f')
            row['Date'] = datetime.strftime(
                start_time,  # type: ignore
                '%m/%d/%Y')
            row['Length'] = '00:00:10.000000'
            for col_name in upper_taxo_links[label_to_sample]:
                row[col_name] = '1'
                print('WARNING WE SHOULD LABEL BY THRESHOLD NOT otherwise')

            # clip and save audio file
            output_folder = f'{split_out_path}{versiontag}/{region}/{location}/'
            Path(output_folder).mkdir(exist_ok=True, parents=True)

            clip_start_time = start_time
            file_start_time = orig_row['timestamp']

            out_file = cut_corresponding_clip(
                clip_start_time,
                file_start_time,
                length,
                output_folder,
                orig_row,
            )
            print("WARNING WE NEED TO MAKE THIS SINGLE CHANNEL BY CLIPPING OR DIRECTLY")

            row['Clip Path'] = out_file
            row['Comments'] = ''

            new_dataset_csv.append(row.copy())
    return new_dataset_csv, not_found_rows


def write_csv(new_csv_file, rows_new, fieldnames=None):
    with open(new_csv_file, 'w', newline='') as csvfile:
        if fieldnames is None:
            fieldnames = rows_new[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows_new)


def setup():
    config = {}
    config['new_dataset_path'] = './datasetV2.1.2.csv'
    # try! to collect same amount of samples from each location
    config['balance_by_location'] = False
    config['confidence_threshold'] = 0.95  # only applies to main label

    # highest value other label predictions can take
    # anthrophony, bird, songbird have their own thresholds
    config['unrelated_label_threshold'] = 0.5

    config['count_from_each_year'] = 125
    config['new_data_classes_counts'] = {
        'aircraft': 1173,
        'insect': 988,
        'duck-goose-swan': 1026,
        'grouse-ptarmigan': 1163,
        'silence': 800,
        'anthrophony': 1000,
    }

    config['versiontag'] = '3rk9ayjc-V1'
    data_folder = '/scratch/enis/data/nna/'
    config['merged_out_dir'] = (data_folder + 'results/csv_export_raw_merged/')
    config['split_out_path'] = data_folder + 'labeling/megan/'

    # new_data_classes_counts_all = { 'biophony':1000,
    #                                'insect':2000,
    #                                'bird':500,
    #                                'songbirds':500,
    #                                'duck-goose-swan':1026,
    #                                'grouse-ptarmigan':1163,
    #                                'anthrophony':2000,
    #                                 'aircraft':2000,
    #                                'silence':1500,
    #                               }

    config['related_cols'] = {
        'biophony': ['biophony', 'grouse-ptarmigan', 'duck-goose-swan', 'bird'],
        'insect': ['insect', 'biophony'],
        'bird': ['bird', 'biophony', 'grouse-ptarmigan', 'duck-goose-swan'],
        'songbirds': [
            'songbirds',
            'bird'
            'biophony',
        ],
        'duck-goose-swan': [
            'duck-goose-swan',
            'biophony',
            'bird',
        ],
        'grouse-ptarmigan': ['grouse-ptarmigan', 'biophony', 'bird'],
        'anthrophony': [
            'anthrophony',
            'aircraft',
        ],
        'aircraft': [
            'aircraft',
            'anthrophony',
        ],
        'silence': ['silence',],
    }

    config['upper_taxo_links'] = {
        'biophony': ['biophony',],
        'insect': ['insect', 'biophony'],
        'bird': [
            'bird',
            'biophony',
        ],
        'songbirds': [
            'songbirds',
            'biophony',
        ],
        'duck-goose-swan': [
            'duck-goose-swan',
            'biophony',
            'bird',
        ],
        'grouse-ptarmigan': ['grouse-ptarmigan', 'biophony', 'bird'],
        'anthrophony': ['anthrophony',],
        'aircraft': [
            'aircraft',
            'anthrophony',
        ],
        'silence': ['silence',],
    }

    config['reg_loc_per_set'] = [[
        ('anwr', '33'), ('anwr', '35'), ('anwr', '37'), ('anwr', '38'),
        ('anwr', '39'), ('anwr', '40'), ('anwr', '41'), ('anwr', '42'),
        ('anwr', '43'), ('anwr', '44'), ('anwr', '46'), ('anwr', '48'),
        ('anwr', '49'), ('dalton', '01'), ('dalton', '02'), ('dalton', '03'),
        ('dalton', '04'), ('dalton', '05'), ('dalton', '06'), ('dalton', '07'),
        ('dalton', '08'), ('dalton', '09'), ('dalton', '10'),
        ('dempster', '11'), ('dempster', '12'), ('dempster', '13'),
        ('dempster', '14'), ('dempster', '15'), ('dempster', '16'),
        ('dempster', '17'), ('dempster', '19'), ('dempster', '20'),
        ('dempster', '21'), ('dempster', '22'), ('dempster', '23'),
        ('dempster', '24'), ('dempster', '25'), ('ivvavik', 'AR01'),
        ('ivvavik', 'AR02'), ('ivvavik', 'AR03'), ('ivvavik', 'AR04'),
        ('ivvavik', 'AR05'), ('ivvavik', 'AR06'), ('ivvavik', 'AR07'),
        ('ivvavik', 'AR08'), ('ivvavik', 'AR09'), ('ivvavik', 'AR10'),
        ('ivvavik', 'SINP01'), ('ivvavik', 'SINP02'), ('ivvavik', 'SINP03'),
        ('ivvavik', 'SINP04'), ('ivvavik', 'SINP05'), ('ivvavik', 'SINP06'),
        ('ivvavik', 'SINP07'), ('ivvavik', 'SINP08'), ('ivvavik', 'SINP09'),
        ('ivvavik', 'SINP10'), ('prudhoe', '11'), ('prudhoe', '13'),
        ('prudhoe', '14'), ('prudhoe', '16'), ('prudhoe', '17'),
        ('prudhoe', '19'), ('prudhoe', '21'), ('prudhoe', '22'),
        ('prudhoe', '23'), ('prudhoe', '24'), ('prudhoe', '25'),
        ('prudhoe', '28'), ('prudhoe', '29'), ('prudhoe', '30')
    ],
                                 [('prudhoe', '15'), ('prudhoe', '20'),
                                  ('anwr', '31'), ('anwr', '34'),
                                  ('anwr', '47')],
                                 [('prudhoe', '12'), ('prudhoe', '18'),
                                  ('prudhoe', '26'), ('prudhoe', '27'),
                                  ('anwr', '32'), ('anwr', '36'),
                                  ('anwr', '45'), ('anwr', '50')]]

    config['excell_label_headers'] = [
        'Anth', 'Bio', 'Geo', 'Sil', 'Auto', 'Airc', 'Mach', 'Flare', 'Bird',
        'Mam', 'Bug', 'Wind', 'Rain', 'Water', 'Truck', 'Car', 'Prop', 'Helo',
        'Jet', 'Corv', 'SongB', 'DGS', 'Grous', 'Crane', 'Loon', 'SeaB', 'Owl',
        'Hum', 'Rapt', 'Woop', 'ShorB', 'Woof', 'Bear', 'Mous', 'Deer', 'Weas',
        'Meow', 'Hare', 'Shrew', 'Mosq', 'Fly'
    ]
    config['excell_all_headers'] = [
        'data_version', 'Annotator', 'Site ID', 'Comments', 'File Name', 'Date',
        'Start Time', 'End Time', 'Length', 'Anth', 'Bio', 'Geo', 'Sil', 'Auto',
        'Airc', 'Mach', 'Flare', 'Bird', 'Mam', 'Bug', 'Wind', 'Rain', 'Water',
        'Truck', 'Car', 'Prop', 'Helo', 'Jet', 'Corv', 'SongB', 'DGS', 'Grous',
        'Crane', 'Loon', 'SeaB', 'Owl', 'Hum', 'Rapt', 'Woop', 'ShorB', 'Woof',
        'Bear', 'Mous', 'Deer', 'Weas', 'Meow', 'Hare', 'Shrew', 'Mosq', 'Fly',
        'Clip Path'
    ]

    return config


# print('test')


def main():

    config = setup()

    confident_preds_dict = load_filter_csv(
        config['new_data_classes_counts'], config['reg_loc_per_set'],
        config['merged_out_dir'], config['versiontag'],
        config['confidence_threshold'], config['related_cols'],
        config['count_from_each_year'], config['unrelated_label_threshold'])
    if config['balance_by_location']:
        confident_preds_dict = pick_clips_equally_from_locations(
            config['new_data_classes_counts'], confident_preds_dict)

    confident_preds_dict_sampled_merged = merge_picked_sampels(
        config['new_data_classes_counts'], confident_preds_dict)

    file_properties_df = pd.read_pickle(
        '/scratch/enis/data/nna/database/allFields_dataV5.pkl')

    new_dataset_csv, not_found_rows = generate_new_dataset(
        config['new_data_classes_counts'], confident_preds_dict_sampled_merged,
        config['versiontag'], config['split_out_path'], file_properties_df,
        config['upper_taxo_links'])

    new_dataset_csv_backup = deepcopy(new_dataset_csv)
    not_found_rows_backup = deepcopy(not_found_rows)

    del not_found_rows_backup, new_dataset_csv_backup

    # add missing labels
    for row in new_dataset_csv:
        for m in config['excell_label_headers']:
            k = row.get(m, None)
            if k is None:
                row[m] = '0'

    write_csv(config['new_dataset_path'],
              new_dataset_csv,
              fieldnames=config['excell_all_headers'])
