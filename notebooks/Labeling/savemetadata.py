""" This script is used to save metadata of audio files in a given folder.
"""

from pathlib import Path
from datetime import datetime, timedelta
import os
import subprocess
import json
import shlex
import concurrent.futures
import glob
import argparse

from collections import Counter

from typing import Union, List

from tqdm import tqdm
import pandas as pd

try:
    from megantaxo import io as m_io
    TIMESTAMP_DATASET_FORMAT = m_io.TIMESTAMP_DATASET_FORMAT
except:
    TIMESTAMP_DATASET_FORMAT = '%Y-%m-%d %H:%M:%S'

FILE_PATH_COL = 'file_path'
REGION_COL = 'region'
LOC_COL = 'location'


### Functions
def count_file_extensions(path_list):
    file_extensions = [Path(i).suffix.lower() for i in path_list]
    file_extensions_counter = Counter(file_extensions)
    return file_extensions_counter


# Load previous data
def load_previous_data(metadata_file):

    if not metadata_file or metadata_file == '.':
        return pd.DataFrame()
    metadata_file = Path(metadata_file)
    if metadata_file.exists():
        if metadata_file.suffix == '.pkl':
            current_file_properties_df = pd.read_pickle(str(metadata_file))
            current_file_properties_df.set_index(FILE_PATH_COL,
                                                 inplace=True,
                                                 drop=False)
        elif metadata_file.suffix == '.csv':
            current_file_properties_df = pd.read_csv(str(metadata_file),
                                                     index_col=FILE_PATH_COL)
        else:
            raise ValueError('unknown file type for current_metadata_file')
        return current_file_properties_df
    else:
        print(f'{metadata_file} does NOT exists')
        return pd.DataFrame()


def list_files(search_path: str = '/search_path/',
               ignore_folders: Union[List, None] = None,
               file_name: str = '*.*'):
    if ignore_folders is None:
        ignore_folders = []
    if search_path[-1] != '/':
        search_path += '/'
    all_path_set = set(
        glob.glob(search_path + '**/' + file_name, recursive=True))

    for folder in ignore_folders:
        ignore_paths = set(glob.glob(folder + '**/*.*', recursive=True))
        all_path_set = all_path_set.difference(ignore_paths)
    all_path = sorted(list(all_path_set))
    return all_path


def get_media_info(
    file_path,
    ffprobe_path='/scratch/enis/conda/envs/speechEnv/bin/ffprobe',
    verbose=False,
):
    cmd = f'{ffprobe_path} -v quiet -print_format json -show_format'
    args = shlex.split(cmd)
    args.append(file_path)

    try:
        # run the ffprobe process, decode stdout into utf-8 & convert to JSON
        ffprobe_output = subprocess.check_output(args).decode('utf-8')
        ffprobe_output = json.loads(ffprobe_output)

        # just grab the duration
        info = (ffprobe_output['format'])

        return info, None

    except subprocess.CalledProcessError as e:
        if verbose:
            print('ffprobe encountered an error with this file:', e)
        return {}, str(e)

    except Exception as e:
        if verbose:
            print('An error occurred:', e)
        return {}, str(e)


def get_media_duration(
        file_path,
        ffprobe_path='/scratch/enis/conda/envs/speechEnv/bin/ffprobe',
        verbose=False):
    info, error = get_media_info(file_path, ffprobe_path, verbose)
    if error:
        return -1, error
    else:
        duration = float(info['duration'])
        return duration, error


def concurrent_get_media_duration(
        files,
        ffprobe_path='/scratch/enis/conda/envs/speechEnv/bin/ffprobe',
        verbose=False):
    files_werror = []
    length_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(get_media_duration, file, ffprobe_path, verbose):
            file for file in files
        }

        # Use the progress_wrapper for progress tracking
        wrapped_futures = tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="Processing")

        for i, future in enumerate(wrapped_futures):
            file = futures[future]
            try:
                duration, error = future.result()
                if error:
                    files_werror.append((file, error))
                else:
                    length_dict[file] = duration
            except Exception as exc:
                print(f'\nAn exception occurred with file {file}: {exc}'
                     )  # Added newline for cleaner print if exception occurs
    return length_dict, files_werror


def get_new_files(new_file_list, current_file_properties_df=None):
    # remove files we already know about
    if current_file_properties_df is None:
        current_file_properties_df = pd.DataFrame()
    prev_file_set = set(current_file_properties_df.index)
    prev_file_set = set([str(m) for m in prev_file_set])
    found_file_set = set(new_file_list)
    found_file_set = found_file_set.difference(prev_file_set)
    total_file_set = len(found_file_set) + len(prev_file_set)
    new_file_path_list = list(found_file_set)
    print(f'{len(new_file_path_list)} new files, previously: ' +
          f' {len(prev_file_set)} and total is {total_file_set}')
    return new_file_path_list


def parse_file_paths(file_paths, parent_path=''):
    parsed_data = {}
    exceptions = []

    for file_path in file_paths:
        # Ensure the path starts with the parent_path
        if not file_path.startswith(parent_path):
            exceptions.append(file_path)
            continue

        # Removing the parent path and
        #  splitting the remaining path into segments
        segments = os.path.relpath(file_path, parent_path).split(os.sep)

        # If there are not enough segments, add to exceptions
        if len(segments) < 4:
            exceptions.append(file_path)
            continue

        # Extracting the values based on segments
        region = segments[0]
        location = segments[1]
        year = segments[2]

        # Splitting the filename based on underscores
        filename_parts = segments[-1].split('_')

        # Depending on the number of segments,
        #  extracting recorder ID, date, and time
        if len(filename_parts) == 3:  # has recorder ID
            recorder_id, date, time_with_ext = filename_parts
            time = time_with_ext.split('.')[0]
        elif len(filename_parts) == 2:  # no recorder ID
            recorder_id = None
            date, time_with_ext = filename_parts
            time = time_with_ext.split('.')[0]
        else:
            exceptions.append(file_path)
            continue

        # Convert the combined date and time into a datetime object
        timestamp = datetime.strptime(f'{date} {time}', '%Y%m%d %H%M%S')

        parsed_data[file_path] = {
            REGION_COL: region,
            LOC_COL: location,
            'year': year,
            'recorder_id': recorder_id,
            'timestamp': timestamp
        }

    return parsed_data, exceptions


def check_year_consistency(parsed_data,):
    anomalies = []

    for file_path, data in parsed_data.items():
        year_from_folder = data['year']
        year_from_timestamp = data['timestamp'].strftime('%Y')

        if year_from_folder != year_from_timestamp:
            anomalies.append(file_path)

    # Printing number of files that are wrong
    if len(anomalies) > 0:
        print(
            f'WARNING: Number of files with inconsistent years: {len(anomalies)}'
        )
        print('\tYear from folder name is different than year from timestamp')
        print('\texample:', anomalies[0])

    return anomalies


def add_time2fileproperties(file_properties, length_dict):
    for f in file_properties:
        length_seconds = length_dict[f]
        length_time_delta = timedelta(seconds=length_seconds)
        start_ts = file_properties[f]['timestamp']
        end_ts = start_ts + length_time_delta

        file_properties[f]['duration_sec'] = length_dict[f]
        file_properties[f]['start_date_time'] = datetime.strftime(
            start_ts, TIMESTAMP_DATASET_FORMAT)
        file_properties[f]['end_date_time'] = datetime.strftime(
            end_ts, TIMESTAMP_DATASET_FORMAT)
        del file_properties[f]['timestamp']
        del file_properties[f]['year']
    return file_properties


def file_properties2df(file_properties):
    for f in file_properties:
        file_properties[f][FILE_PATH_COL] = f
    file_properties = list(file_properties.values())
    file_properties = pd.DataFrame(file_properties)
    file_properties.set_index(FILE_PATH_COL, inplace=True, drop=False)
    return pd.DataFrame(file_properties)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Save metadata of audio files in a given folder.')
    parser.add_argument('--search_path',
                        default='./',
                        help='Path to search for audio files,' +
                        ' suggested to use absolute path')
    parser.add_argument(
        '--search_parent_path',
        default='',
        help='Part of the search path until the region folder name:' +
        ' e.g. if search path is /data/mydata/region/location/year/ then ' +
        ' search_parent_path is /data/mydata/')
    parser.add_argument('--search_ignore_folders',
                        nargs='*',
                        default=[],
                        help='List of folders to ignore.')
    parser.add_argument('--previous_metadata_path',
                        default='',
                        help='Path to previous metadata.')
    parser.add_argument('--new_metadata_path',
                        default='./recordings_metadata.csv',
                        help='Where to save new metadata.')
    parser.add_argument('--ffprobe_path',
                        default='ffprobe',
                        help='Path to ffprobe.')
    parser.add_argument('--log_folder',
                        default='./metadata_logs/',
                        help='Folder to save errors and logs.')

    return parser.parse_args(args)


def main_logic(args):
    # External Programs
    # ffprobe version >= 4.3.1
    ffprobe_path = args.ffprobe_path

    # PARAMETERS for new metadata

    # increase version number accordinly
    previous_metadata_path = Path(
        args.previous_metadata_path) if args.previous_metadata_path else None
    new_metadata_path = Path(args.new_metadata_path)
    log_folder = Path(args.log_folder)
    search_path = args.search_path
    search_parent_path = args.search_parent_path
    ignore_folders = args.search_ignore_folders

    if search_parent_path == '':
        print('search_parent_path is not given, ' +
              'make sure search_path starts with region folder' +
              'however it is suggested to use search_parent_path')
    # we can load them
    new_metadata_path_name = new_metadata_path.name
    files_audio_error_out_file = (
        log_folder / f'files-audio-error_{new_metadata_path_name}.csv')
    files_w_wrong_name = (log_folder /
                          f'files-wrong-name_{new_metadata_path_name}.txt')

    # Find files
    # in given search path ignoring given directories
    print(f'Searching for files in {search_path}')
    files_path_list = list_files(search_path, ignore_folders)
    print(f'Found {len(files_path_list)} files')
    if len(files_path_list) == 0:
        raise Exception('No files found, exiting')
    if Path(files_path_list[0]).is_dir():
        raise Exception(
            'first file found is a directory, exiting, there is a bug')

    files_suffix_counts = count_file_extensions(files_path_list)
    print('File extensions counts:')
    print(files_suffix_counts)
    print('\n')

    previous_metadata = load_previous_data(previous_metadata_path)
    new_files_path_list = get_new_files(files_path_list, previous_metadata)
    if len(new_files_path_list) == 0:
        raise Exception('No new files found, exiting')
    print(f'Processing new files:\n')
    length_dict, files_werror = concurrent_get_media_duration(
        new_files_path_list, ffprobe_path, verbose=False)

    if len(files_werror) > 0:
        print(f'there are {len(files_werror)}' +
              ' number of files with errors reading length via ffprobe')
        with open(files_audio_error_out_file, 'w', encoding='utf-8') as f:
            lines = [line.join(',') for line in files_werror]
            f.write('\n'.join(lines) + '\n')
        print(f'files with errors are saved in {files_audio_error_out_file}')

    file_properties, exceptions = parse_file_paths(list(length_dict.keys()),
                                                   search_parent_path)

    check_year_consistency(file_properties)

    if len(exceptions) > 0:
        print(f'there are {len(exceptions)} number of ' +
              'files with errors reading metadata from filename')
        print('Possible reasons: - file name is not in the format of ' +
              'recorderid_YYYYMMDD_HHMMSS.extension' + '\n or ' +
              ' - file path is not /region/location/year/filename')
        with open(files_w_wrong_name, 'w', encoding='utf-8') as f:
            lines = [line[0] for line in exceptions]
            f.write('\n'.join(lines) + '\n')
        print(f'files with exceptions are saved in {files_w_wrong_name}')

    file_properties = add_time2fileproperties(file_properties, length_dict)
    file_properties_df = file_properties2df(file_properties)
    combined_df = pd.concat([previous_metadata, file_properties_df])
    if new_metadata_path.suffix == '.pkl':
        combined_df.to_pickle(new_metadata_path)
    elif new_metadata_path.suffix == '.csv':
        combined_df.to_csv(new_metadata_path, index=False)
    else:
        raise ValueError('unknown file type for new_metadata_path, ' +
                         ' only .pkl and .csv are supported')
    print(f'updated metadata is saved to {new_metadata_path}')


def main():
    args = parse_args()
    main_logic(args)


if __name__ == '__main__':
    main()