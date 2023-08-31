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


def load_all_rawcsv_as_df(
    reg_loc,
    merged_out_dir,
    versiontag,
    years=None,
):
    '''
    Iterate all files, load with get_csv_into_pd
    '''
    if years is None:
        years = []
    preds_list = []
    for region, location in reg_loc:
        year_folders = glob.glob(
            f'{merged_out_dir}{versiontag}/{region}/{location}/*')
        print(f'{merged_out_dir}{versiontag}/{region}/{location}/*')
        for year in year_folders:
            year = (Path(year).stem)
            if not years or year in years:
                pred_for_year = get_csv_into_pd(merged_out_dir, versiontag,
                                                region, location, year)
                preds_list.append(pred_for_year.copy())

    preds_df = pd.concat(preds_list)

    return preds_df


def pick_confidents(pred_for_year, label_to_sample, confidence_threshold,
                    related_cols, unrelated_label_thresholds,
                    columns_to_exclude):
    '''
    Filter by label's confidence and other high confidence labels.

    '''
    pred_confident = filterby_label_confidence(pred_for_year, label_to_sample,
                                               confidence_threshold)

    pred_confident = filter_other_high_preds(
        pred_confident,
        label_to_sample,
        related_cols,
        unrelated_label_thresholds,
        columns_to_exclude=columns_to_exclude)

    return pred_confident


def filterby_label_confidence(pred_for_year, label_to_sample,
                              confidence_threshold):

    pred_confident = pred_for_year[
        pred_for_year[label_to_sample] > confidence_threshold]

    return pred_confident


def filter_other_high_preds(
    pred_df,
    targe_label,
    related_cols,
    labels_thresholds=None,
    columns_to_exclude=None,
):
    '''
        Filter sample out if other labels has high confidence.
    '''
    if labels_thresholds is None:
        labels_thresholds = {'default': '0.5'}

    filtered_pred_confident = pred_df.copy()
    # remove samples that has other unrelated sounds
    col_2_ignore = (columns_to_exclude + related_cols[targe_label])
    col_2_ignore.append(targe_label)

    for col in filtered_pred_confident.columns:
        if col not in col_2_ignore:
            print(col)
            threshold = labels_thresholds.get(col, labels_thresholds['default'])
            filtered_pred_confident = filtered_pred_confident[
                filtered_pred_confident[col] < threshold]
            print(len(filtered_pred_confident))
    return filtered_pred_confident


# def load_filter_csv(new_data_classes_counts, reg_loc_per_set, merged_out_dir,
#                     versiontag, confidence_threshold, related_cols,
#                     count_from_each_year, unrelated_label_thresholds):
#     '''
#     Iterate all files, load with get_csv_into_pd and filter with pick_confidents

#         This one creates a dict of df for each label, filters each one by
#         confidence

#     '''

#     confident_preds_dict = {i: [] for i in new_data_classes_counts.keys()}

#     for region, location in reg_loc_per_set[0]:
#         print(region, location)
#         # location_csv_files=glob.glob(f'export_raw_v6/{i[1]}*')
#         year_folders = glob.glob(
#             f'{merged_out_dir}{versiontag}/{region}/{location}/*')
#         #     print(f'{merged_out_dir}{versiontag}/{region}/{location}/*')
#         for year in year_folders:
#             year = (Path(year).stem)

#             pred_for_year = get_csv_into_pd(merged_out_dir, versiontag, region,
#                                             location, year)

#             for label_to_sample in new_data_classes_counts.keys():

#                 pred_confident = pick_confidents(pred_for_year, label_to_sample,
#                                                  confidence_threshold,
#                                                  related_cols,
#                                                  unrelated_label_thresholds)
#                 confident_preds_dict[label_to_sample].append(
#                     pred_confident.copy())
#                 print(
#                     len(pred_confident),
#                     label_to_sample,
#                 )
#     return confident_preds_dict


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


def cut_corresponding_clip(
    clip_start_time,
    file_start_time,
    length,
    output_folder,
    orig_row,
    outputSuffix=None,
    dry_run=False,
    stereo2mono=False,
    overwrite=True,
    sampling_rate=48000,
):

    s_str, e_str = relative_time(file_start_time, clip_start_time, length)

    Path(output_folder).mkdir(exist_ok=True, parents=True)

    out_file = splitmp3(str(orig_row.name),
                        output_folder,
                        s_str,
                        e_str,
                        backend_path='/home/enis/sbin/ffmpeg',
                        outputSuffix=outputSuffix,
                        dry_run=dry_run,
                        stereo2mono=stereo2mono,
                        overwrite=overwrite,
                        sampling_rate=sampling_rate)

    return out_file


def generate_new_row(dataset_version, model_version_tag, region, location,
                     orig_filename, length, start_time):
    '''
        Generate single row for dataset with partial info.
    '''
    row = {}
    row['data_version'] = dataset_version
    row['Annotator'] = model_version_tag
    # row['Site ID'] = location
    row['location'] = location
    row['region'] = region
    row['Comments'] = ''
    row['File Name'] = str(orig_filename)
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
    row['Reviewed'] = 'false'
    row['extra_tags'] = ''
    row['start_date_time'] = datetime.strftime(start_time,
                                               '%Y-%m-%dT%H:%M:%S.%f')
    row['end_date_time'] = datetime.strftime(
        start_time +  # type: ignore
        timedelta(seconds=length),
        '%Y-%m-%dT%H:%M:%S.%f')
    return row


def label_row_by_thresholds(pred_row,
                            row,
                            upper_taxo_links,
                            excell_label_headers,
                            labels_thresholds=None,
                            excell_labels_2_pdnames=None):
    '''
        Given confidence values and thresholds, set 1,0 for labels.

        upper_taxo_links: dict of {label: [list of labels]} for taxonomy
                            labels that are parents to the label.
        excell_label_headers: list of labels that are in the excell file.
        labels_thresholds: dict of {label: threshold} for labels.
        excell_labels_2_pdnames: dict of {label: pdname} for labels.
    '''
    if labels_thresholds is None:
        labels_thresholds = {'default': '0.5'}
    if excell_labels_2_pdnames is None:
        excell_labels_2_pdnames = {}

    default_val = labels_thresholds['default']
    silence_flag = True
    for label in excell_label_headers:
        header_label = label
        label = excell_labels_2_pdnames.get(label, None)
        if label is None:
            row[header_label] = '0'
            continue
        # print(label,labels_thresholds)
        threshold = labels_thresholds.get(label, None)
        if threshold is None:
            print('threshold is None')
            threshold = default_val

        if pred_row[label] >= threshold:
            silence_flag = False
            row[header_label] = '1'
            for rel_label in upper_taxo_links[header_label]:
                row[rel_label] = '1'
        else:
            row[header_label] = '0'

    if silence_flag:
        row['Sil'] = '1'

    return row


def generate_new_dataset(
    pred_df,
    versiontag,
    split_out_path,
    file_properties_df,
    upper_taxo_links,
    dataset_version,
    length=10,
    buffer=0,
    excell_label_headers=None,
    labels_thresholds=None,
    outputSuffix=None,
    dry_run=False,
    excell_labels_2_names=None,
    stereo2mono=False,
    overwrite=True,
    sampling_rate=48000,
    label_row_by_threshold=True,
    print_logs=True,
):
    # create dataset csv from picked rows
    # get related info and clip wav files
    # needs region,location,timestamp in pred_df
    # needs file_properties_df

    new_dataset_csv = []
    not_found_rows = []

    for index, pred_row in pred_df.iterrows():
        row = {}
        start_time = pred_row['timestamp']
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')

        output = find_filesv2(pred_row['location'],
                              pred_row['region'],
                              start_time,
                              None,
                              length,
                              buffer,
                              file_properties_df,
                              only_continuous=True)
        (sorted_filtered, start_time, _, _, _) = output

        if len(sorted_filtered.index) == 0:
            not_found_rows.append((index, pred_row))
            if print_logs:
                print('--------------------------------------')
                print('Not Found')
                print(index, pred_row)
            continue

        if len(sorted_filtered.index) > 1:
            print('Too many')

        location = pred_row['location']
        region = pred_row['region']
        year = pred_row['timestamp'].year

        orig_row = sorted_filtered.iloc[0]
        row = generate_new_row(dataset_version, versiontag, region, location,
                               orig_row.name, length, start_time)
        if label_row_by_threshold:
            row = label_row_by_thresholds(
                pred_row,
                row,
                upper_taxo_links,
                excell_label_headers,
                labels_thresholds,
                excell_labels_2_pdnames=excell_labels_2_names)
        # clip and save audio file
        output_folder = f'{split_out_path}{versiontag}/audio_{dataset_version}/{region}/{location}/{year}/'
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        clip_start_time = start_time
        file_start_time = orig_row['timestamp']
        out_file = cut_corresponding_clip(clip_start_time,
                                          file_start_time,
                                          length,
                                          output_folder,
                                          orig_row,
                                          outputSuffix=outputSuffix,
                                          dry_run=dry_run,
                                          stereo2mono=stereo2mono,
                                          overwrite=overwrite,
                                          sampling_rate=sampling_rate)

        row['Clip Path'] = str(out_file)
        row['Comments'] = ''
        if dry_run and print_logs:
            print('--------------------------------------')
            # print(index, pred_row)
            print(out_file)
            print(row)
        new_dataset_csv.append(row.copy())

    return new_dataset_csv, not_found_rows


def write_csv(new_csv_file, rows_new, fieldnames=None):
    with open(new_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        if fieldnames is None:
            fieldnames = rows_new[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows_new)


def setup(versiontag=None,):
    config = {}

    # highest value other label predictions can take
    # anthrophony, bird, songbird have their own thresholds
    if versiontag is None:
        config['versiontag'] = '3rk9ayjc-V1'
    else:
        config['versiontag'] = versiontag
    data_folder = '/scratch/enis/data/nna/'
    config['merged_out_dir'] = (data_folder + f'results/csv_export_raw_merged/')
    config['split_out_path'] = data_folder + 'labeling/megan/'

    config['new_dataset_path'] = './datasetV2.1.XXX.csv'
    config['dataset_version'] = 'V2.1.XXX'
    # try! to collect same amount of samples from each location
    config['balance_by_location'] = False

    config['count_from_each_year'] = 125
    config['new_data_classes_counts'] = {
        'biophony': 10 * 3,
        'insect': 10 * 4,
        'bird': 10**3,
        'songbirds': 10 * 4,
        'duck-goose-swan': 10 * 4,
        'grouse-ptarmigan': 10 * 4,
        'anthrophony': 10 * 4,
        'aircraft': 10**4,
        'silence': 10 * 3,
    }
    config['confidence_threshold'] = 0.95  # only applies to main label

    config['confidence_thresholds'] = {
        'biophony': 0.99,
        'insect': 0.99,
        'bird': 0.99,
        'songbirds': 0.95,
        'duck-goose-swan': 0.9,
        'grouse-ptarmigan': 0.9,
        'anthrophony': 0.99,
        'aircraft': 0.99,
        'silence': 0.99,
    }
    config['labels_thresholds'] = {
        'default': 0.5,
        'biophony': 0.5,
        'insect': 0.5,
        'bird': 0.37,
        'songbirds': 0.12,
        'duck-goose-swan': 0.5,
        'grouse-ptarmigan': 0.5,
        'anthrophony': 0.08,
        'aircraft': 0.5,
        'silence': 0.5,
    }

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
        'biophony': [
            'biophony', 'grouse-ptarmigan', 'duck-goose-swan', 'bird',
            'songbirds'
        ],
        'insect': ['insect', 'biophony'],
        'bird': [
            'bird', 'biophony', 'grouse-ptarmigan', 'duck-goose-swan',
            'songbirds'
        ],
        'songbirds': [
            'songbirds',
            'bird',
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
        'Bio': ['Bio'],
        'Bug': ['Bug', 'Bio'],
        'Bird': ['Bird', 'Bio'],
        'Song': ['Song', 'Bio'],
        'DGS': ['DGS', 'Bio', 'Bird'],
        'Grous': ['Grous', 'Bio', 'Bird'],
        'Anth': ['Anth'],
        'Airc': ['Airc', 'Anth'],
        'Sil': ['Sil']
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

    config['excell_labels_2_names'] = {
        'Bio': 'biophony',
        'Bug': 'insect',
        'Bird': 'bird',
        'Song': 'songbirds',
        'DGS': 'duck-goose-swan',
        'Grous': 'grouse-ptarmigan',
        'Anth': 'anthrophony',
        'Airc': 'aircraft',
        'Sil': 'silence'
    }

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
        config['upper_taxo_links'], config['excell_label_headers'],
        config['labels_thresholds'], config['excell_labels_2_names'])

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
