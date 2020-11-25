"""Generating mock data that is used in general withing the project.
    Most common data passed around is file_properties_df, it holds information
    about original files.
    Usage:

"""
from typing import List, Dict, Callable, Tuple

import pandas as pd
import numpy as np
import random
import string
from pathlib import Path
import datetime
from nna import fileUtils

from nna.tests.mock_data_params import EXISTING_REGION_LOCATIONID  #,IGNORE_LOCATION_ID
from nna.tests.mock_data_params import EXISTING_YEARS, IGNORE_YEARS, EXISTING_SUFFIX
from nna.tests.mock_data_params import EXISTING_LONGEST_FILE_LEN

random.seed(44)


def mock_file_properties_df(
    index_length: int = 100,
    semantic_errors: bool = False,
    structural_errors: bool = False,
):
    """Generate a file_properties_df pd.DataFrame storing original file info.

        Args:
            index_length: number of items in the DataFrame
            semantic_errors: will there be semantic errors in the rows.
                wrong dates, negative duration time value etc.
            structural_errors: will there be structural errors with the
                DataFrame Missing columns, missing values, wrong type of values.
        file_properties_df's indexes are Path(a_file) and columns are:
            ['region','site_name', 'locationId','site_id', 'recorderId',
            'timestamp','year', 'month', 'day', 'hour_min_sec',
            'timestampEnd', 'durationSec',
            ]
            region: Name of the region (ex:prudhoe,anwr...)
            site_name: Deprecated name for region
            locationId: ID of the location within a site, not everywhere!
            site_id: Deprecated name for LocationId
            recorderId: ID of the recorder
            timestamp: when recording started
            year,month,day,hour_min_sec: values from filename, created timestamp
            timestampEnd: when recording ends
            durationSec: length of the recording
        There should not be two files covering the same location and at
            the same time.
        Returns: A pandas.DataFrame similar to file_properties_df.
    """
    # TODO: implement these
    del semantic_errors, structural_errors
    # example Path and info:
    # /tank/data/nna/real/anwr/40/2019/S4A10322_20190705_000000.flac
    # src_path = "/tank/data/nna/real/"
    # region = "anwr"
    # location_id = "40"
    # year_file = "2019"
    # stem_file = "S4A10322_20190705_000000"
    # suffix = ".flax"

    random_src_paths = [
        '/tank/data/nna/real/', '/scratch/enis/nna/real/', '/scratch/enis/nna/',
        '/nna/real/', './'
    ]
    acceptable_years = list(set(EXISTING_YEARS) - set(IGNORE_YEARS))

    # generate random list of recorder_id
    uppercase_letters = string.ascii_uppercase
    recorder_id_list = []
    distinct_recorder_id_count = 20
    for _ in range(distinct_recorder_id_count):
        recorder_id = ''.join([
            random.choice(uppercase_letters),
            str(random.randint(0, 9)),
            random.choice(uppercase_letters),
            str(random.randint(10000, 99999))
        ])
        recorder_id_list.append(recorder_id)

    data_df: Dict[Path():Dict] = {}
    for _ in range(index_length):
        random_path, mock_row = mock_file_properties_df_row(
            random_src_paths, acceptable_years, recorder_id_list)
        data_df[random_path] = mock_row

    df_dict = pd.DataFrame.from_dict(data_df, orient='index')
    df_dict = df_dict.astype({'durationSec': 'O'})
    return df_dict


def mock_file_properties_df_row(
        random_src_paths: List[str], acceptable_years: List[str],
        recorder_id_list: List[str]) -> Tuple[Path, Dict]:
    """Generate a random row of file_properties_df.

        Given options for src_path, years and recorder_id, create a random
        row to be used with file_properties_df.

        Returns:
            (index: Path,row: Dict)
    """
    src_path = random.choice(random_src_paths)
    region, location_id = random.choice(EXISTING_REGION_LOCATIONID)
    year_file = random.choice(acceptable_years)
    first_day_year = datetime.datetime(int(year_file), 1, 1, 00, 00)
    # # 31449600 seconds = 364 days
    random_point_in_year = datetime.timedelta(
        seconds=random.randint(0, 31449600))
    timestamp = first_day_year + random_point_in_year
    duration_sec = random.randint(1, EXISTING_LONGEST_FILE_LEN)
    timestamp_end = timestamp + datetime.timedelta(seconds=duration_sec)
    hour_min_sec = timestamp.strftime('%Y%m%d_%H%M%S')
    recorder_id = random.choice(recorder_id_list)
    stem_file = '_'.join([
        recorder_id,
        hour_min_sec,
    ])
    suffix = random.choice(EXISTING_SUFFIX)

    random_path = (Path(src_path) / region / location_id / year_file /
                   stem_file).with_suffix(suffix)
    # df_index_list.append(random_path)
    mock_row = {
        'region': region,
        'site_name': region,
        'locationId': location_id,
        'site_id': location_id,
        'recorderId': recorder_id,
        'timestamp': timestamp,
        'year': str(timestamp.year),
        'month': str(timestamp.month),
        'day': str(timestamp.day),
        'hour_min_sec': str(hour_min_sec),
        'timestampEnd': timestamp_end,
        'durationSec': float(duration_sec)
    }

    return random_path, mock_row


def mock_result_data_file(fill_value_func: Callable[[int], int],
                          output_file_path: Path,
                          file_length: int,
                          segment_len: float = 10.0,
                          channel_count: int = 1) -> np.array:
    """Create a .npy file and fill it with result values.
    """
    result_len = (file_length // segment_len)
    if file_length % segment_len != 0:
        result_len += 1
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    results = [fill_value_func(index) for index in range(int(result_len))]
    results = np.array(results)
    if channel_count > 1:
        results = np.repeat(np.array(results), channel_count)
        results = results.reshape(-1, channel_count)

    output_file_path = output_file_path.with_suffix('.npy')
    np.save(output_file_path, results)
    # print(output_file_path)

    return results


def mock_results_4input_files(file_properties_df: pd.DataFrame,
                              fill_value_func: Callable[[int], int],
                              output_path: Path,
                              results_tag_id: str = 'XXX',
                              file_length_limit: str = '01:00:00',
                              segment_len: int = 10,
                              channel_count: int = 1):
    """Mock result files for list of input files.

        Given a file properties, for each row, generate an output file and fill
        it with given function.

    """
    resulting_output_file_paths = []
    #
    # calculate file_length_limit in seconds
    hh_mm_ss = file_length_limit.split(':')
    hh_mm_ss = [int(i) for i in hh_mm_ss]
    file_length_limit_seconds = hh_mm_ss[0] * (
        3600) + hh_mm_ss[1] * 60 + hh_mm_ss[2]
    for _, row in file_properties_df.iterrows():
        # print(row)
        file_duration_sec = int(row.durationSec)
        # output file should be multiple segments if input file
        # longer than file_length_limit
        # calculate how many segments should be there
        segment_count = 1
        if file_duration_sec > file_length_limit_seconds:
            segment_count = (file_duration_sec // file_length_limit_seconds)
            file_length_segments = [
                file_length_limit_seconds for i in range(segment_count)
            ]
            file_length_segments.append(
                int(file_duration_sec % file_length_limit_seconds))
            segment_count += 1
        else:
            file_length_segments = [file_duration_sec]

        # go through each segment and create output files
        for file_index in range(segment_count):
            file_index_padded = '{:0>3}'.format(file_index)
            output_file_path = fileUtils.standard_path_style(
                output_path,
                row,
                sub_directory_addon=results_tag_id,
                file_name_addon=(results_tag_id + file_index_padded))

            _ = mock_result_data_file(fill_value_func,
                                      output_file_path,
                                      int(file_length_segments[file_index]),
                                      segment_len=segment_len,
                                      channel_count=channel_count)

            resulting_output_file_paths.append(output_file_path)

    resulting_output_file_paths = [
        i.with_suffix('.npy') for i in resulting_output_file_paths
    ]
    return resulting_output_file_paths


def mock_clipping_results_dict_file(file_properties_df, region_location_name,
                                    clipping_results_path):
    clipping_resultsdict_mock = {}
    for row in file_properties_df.iterrows():
        # print(row[0],row)
        clipping_mock = []
        for i in range(int(row[1].durationSec / 10)):
            channel1 = 1 / (i + 1)
            channel2 = channel1 * 1.1
            clipping_mock.append([channel1, channel2])

        clipping_mock_np = np.array(clipping_mock)
        clipping_resultsdict_mock[row[0]] = clipping_mock_np[:]

    filename = '{}_{}.pkl'.format(region_location_name, '1,0')

    output_file_path = clipping_results_path / filename
    with open(output_file_path, 'wb') as f:
        np.save(f, clipping_resultsdict_mock)
    return output_file_path


if '__init__' == '__main__':

    def ones(i):
        del i
        return 1

    mock_results_4input_files(
        mock_file_properties_df(2), ones,
        Path('/Users/berk/Documents/research/nna/src/nna/tests/job_logs'))
