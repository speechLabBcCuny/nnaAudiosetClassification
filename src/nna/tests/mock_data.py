"""Generating mock data that is used in general withing the project.
    Most common data passed around is file_properties_df, it holds information
    about original files.
    Usage:

"""
from typing import List, Dict, Callable

import pandas as pd
import numpy as np
import random
import string
from pathlib import Path
import datetime
from nna import fileUtils

random.seed(42)

# 14/Nov/2020
# from pathlib import Path
# a = glob.glob("/tank/data/nna/real/*/*")
# a1 = [(Path(i).stem) for i in a]
# a2 = [(Path(i).parent.stem,(Path(i).stem)) for i in a]
EXISTING_REGION_LOCATIONID = [('anwr', '31'), ('anwr', '32'), ('anwr', '33'),
                              ('anwr', '34'), ('anwr', '35'), ('anwr', '36'),
                              ('anwr', '37'), ('anwr', '38'), ('anwr', '39'),
                              ('anwr', '40'), ('anwr', '41'), ('anwr', '42'),
                              ('anwr', '43'), ('anwr', '44'), ('anwr', '45'),
                              ('anwr', '46'), ('anwr', '47'), ('anwr', '48'),
                              ('anwr', '49'), ('anwr', '50'), ('dalton', '01'),
                              ('dalton', '02'), ('dalton', '03'),
                              ('dalton', '04'), ('dalton', '05'),
                              ('dalton', '06'), ('dalton', '07'),
                              ('dalton', '08'), ('dalton', '09'),
                              ('dalton', '10'), ('dempster', '11'),
                              ('dempster', '12'), ('dempster', '13'),
                              ('dempster', '14'), ('dempster', '16'),
                              ('dempster', '17'), ('dempster', '19'),
                              ('dempster', '20'), ('dempster', '21'),
                              ('dempster', '22'), ('dempster', '23'),
                              ('dempster', '24'), ('dempster', '25'),
                              ('ivvavik', 'AR01'), ('ivvavik', 'AR02'),
                              ('ivvavik', 'AR03'), ('ivvavik', 'AR04'),
                              ('ivvavik', 'AR05'), ('ivvavik', 'AR06'),
                              ('ivvavik', 'AR07'), ('ivvavik', 'AR08'),
                              ('ivvavik', 'AR09'), ('ivvavik', 'AR10'),
                              ('ivvavik', 'SINP01'), ('ivvavik', 'SINP02'),
                              ('ivvavik', 'SINP03'), ('ivvavik', 'SINP04'),
                              ('ivvavik', 'SINP05'), ('ivvavik', 'SINP06'),
                              ('ivvavik', 'SINP07'), ('ivvavik', 'SINP08'),
                              ('ivvavik', 'SINP09'), ('ivvavik', 'SINP10'),
                              ('prudhoe', '11'), ('prudhoe', '12'),
                              ('prudhoe', '13'), ('prudhoe', '14'),
                              ('prudhoe', '15'), ('prudhoe', '16'),
                              ('prudhoe', '17'), ('prudhoe', '18'),
                              ('prudhoe', '19'), ('prudhoe', '20'),
                              ('prudhoe', '21'), ('prudhoe', '22'),
                              ('prudhoe', '23'), ('prudhoe', '24'),
                              ('prudhoe', '25'), ('prudhoe', '26'),
                              ('prudhoe', '27'), ('prudhoe', '28'),
                              ('prudhoe', '29'), ('prudhoe', '30'),
                              ('stinchcomb', '01-Itkillik'),
                              ('stinchcomb', '02-Colville2'),
                              ('stinchcomb', '03-OceanPt'),
                              ('stinchcomb', '04-Colville4'),
                              ('stinchcomb', '05-Colville5'),
                              ('stinchcomb', '06-Umiruk'),
                              ('stinchcomb', '07-IceRd'),
                              ('stinchcomb', '08-CD3'),
                              ('stinchcomb', '09-USGS'),
                              ('stinchcomb', '10-Nigliq1'),
                              ('stinchcomb', '11-Nigliq2'),
                              ('stinchcomb', '12-Anaktuvuk'),
                              ('stinchcomb', '13-Shorty'),
                              ('stinchcomb', '14-Rocky'),
                              ('stinchcomb', '15-FishCreek1'),
                              ('stinchcomb', '16-FishCreek2'),
                              ('stinchcomb', '17-FishCreek3'),
                              ('stinchcomb', '18-FishCreek4'),
                              ('stinchcomb', '19-Itkillik2'),
                              ('stinchcomb', '20-Umiat')]
IGNORE_LOCATION_ID = ['excerpts', 'dups']

EXISTING_YEARS = ['2010', '2013', '2016', '2018', '2019']
IGNORE_YEARS = ['2010', '2013', '2016']

EXISTING_SUFFIX = ['.flac', '.mp3', '.FLAC', '.mp3']

# in seconds, (4.5 hours + extra 30 minutes)
EXISTING_LONGEST_FILE_LEN = 5 * 60 * 60


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
            structural_errors: will there be structural errors with the DataFrame
                Missing columns, missing values, wrong type of values.
 

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
        '/nna/real/', "./"
    ]
    acceptable_years = list(set(EXISTING_YEARS) - set(IGNORE_YEARS))

    # generate random list of recorder_id
    uppercase_letters = string.ascii_uppercase
    recorder_id_list = []
    distinct_recorder_id_count = 20
    for _ in range(distinct_recorder_id_count):
        recorder_id = "".join([
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
    return (df_dict)


def mock_file_properties_df_row(random_src_paths: List[str],
                                acceptable_years: List[str],
                                recorder_id_list: List[str]):
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
                          segment_len: float = 10.0):
    """Create a file and fill it with result values.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    results = [fill_value_func(index) for index in range(int(file_length))]
    results = np.array(results)
    np.save(output_file_path, results)


def mock_results_4input_files(file_properties_df: pd.DataFrame,
                              fill_value_func: Callable[[int], int],
                              output_path: Path,
                              results_tag_id: str = "XXX",
                              file_length_limit: str = "01:00:00"):
    """Mock results for list of files.
    """
    results_tag_id = "_" + results_tag_id
    HH_MM_SS = file_length_limit.split(":")
    HH_MM_SS = [int(i) for i in HH_MM_SS]
    file_length_limit_seconds = HH_MM_SS[0] * (
        3600) + HH_MM_SS[1] * 60 + HH_MM_SS[2]
    for _, row in file_properties_df.iterrows():

        file_duration_sec = int(row.durationSec)
        file_count = 1
        if file_duration_sec > file_length_limit_seconds:
            file_count = (file_duration_sec // file_length_limit_seconds)
            file_count += 1
        total_file_length = 
        for file_index in range(file_count):
            file_index_padded = "{:0>3}".format(file_index)
            output_file_path = fileUtils.standard_path_style(
                output_path,
                row,
                sub_directory_addon=results_tag_id,
                file_name_addon=(results_tag_id + file_index_padded))

            mock_result_data_file(fill_value_func,
                                  output_file_path,
                                  int(row.durationSec),
                                  segment_len=10)

    return 1


def ones(i):
    return i


mock_results_4input_files(
    mock_file_properties_df(2), ones,
    Path('/Users/berk/Documents/research/nna/src/nna/tests/job_logs'))
