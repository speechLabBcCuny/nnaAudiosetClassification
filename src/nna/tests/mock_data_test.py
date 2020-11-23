"""Tests for mock_data.py.
"""

import numpy as np
import mock_data
import datetime
import glob
import random
import shutil
from pathlib import Path

from testparams import INPUTS_OUTPUTS_PATH

from nna import fileUtils

from mock_data_params import IGNORE_LOCATION_ID
from mock_data_params import EXISTING_YEARS, IGNORE_YEARS, EXISTING_SUFFIX
from mock_data_params import EXISTING_LONGEST_FILE_LEN

IO_mock_data_path = INPUTS_OUTPUTS_PATH / 'mock_data'

random.seed(42)


def test_mock_file_properties_df():
    """test for mock_data.mock_file_properties_df() func.
    """
    index_length_of_df = 101
    expected_columns = [
        'region',
        'site_name',
        'locationId',
        'site_id',
        'recorderId',
        'timestamp',
        'year',
        'month',
        'day',
        'hour_min_sec',
        'timestampEnd',
        'durationSec',
    ]
    expected_columns.sort()
    expected_columns_types = {
        'site_id': np.dtype('O'),
        'locationId': np.dtype('O'),
        'site_name': np.dtype('O'),
        'recorderId': np.dtype('O'),
        'hour_min_sec': np.dtype('O'),
        'year': np.dtype('O'),
        'month': np.dtype('O'),
        'day': np.dtype('O'),
        'region': np.dtype('O'),
        'timestamp': np.dtype('<M8[ns]'),
        'durationSec': np.dtype('O'),
        'timestampEnd': np.dtype('<M8[ns]')
    }
    mocked_file_properties_df = mock_data.mock_file_properties_df(
        index_length=index_length_of_df,
        semantic_errors=False,
        structural_errors=False,
    )
    mock_column_values = sorted(list(mocked_file_properties_df.columns.values))
    mock_column_dtypes = dict(mocked_file_properties_df.dtypes.items())
    assert len(mocked_file_properties_df.index) == index_length_of_df
    assert mock_column_values == expected_columns
    assert mock_column_dtypes == expected_columns_types


def test_mock_file_properties_df_row():
    """Test for mock_data.mock_file_properties_df_row function.
    """
    random_src_paths = ['/tank/test/test/', '/', '']
    acceptable_years = list(set(EXISTING_YEARS) - set(IGNORE_YEARS))
    recorder_id_list = ['S4A10322', 'M2B11342', 'K6B11242']
    random_path, mock_row = mock_data.mock_file_properties_df_row(
        random_src_paths, acceptable_years, recorder_id_list)

    for key, val in mock_row.items():
        if key == 'durationSec':
            assert isinstance(val, float)
        elif 'timestamp' in key:
            assert isinstance(val, datetime.datetime)
        else:
            assert isinstance(val, str)

    assert mock_row['region'] == mock_row['site_name']
    assert mock_row['locationId'] == mock_row['site_id']
    assert random_path.suffix in EXISTING_SUFFIX
    assert len(random_path.parts) >= 4
    assert random_path.parts[-2] == str(int(random_path.parts[-2]))
    assert len(mock_row.keys()) == 12


def test_mock_result_data_file():
    """Test for mock_data.mock_result_data_file function.
    """
### test with ones
    def ones(i):
        del i
        return 1

    func_output_path = IO_mock_data_path / 'mock_result_data_file' / 'outputs'
    output_file_path = func_output_path / 'data_file.npy'
    file_length = 10244
    segment_len = 10
    result_array = mock_data.mock_result_data_file(ones,
                                                   output_file_path,
                                                   file_length,
                                                   segment_len=segment_len)
    output_file_path = output_file_path.with_suffix('.npy')

    result_array_file = np.load(output_file_path)
    assert np.array_equal(result_array, result_array_file)

    reslen = file_length // segment_len
    if file_length % segment_len != 0:
        reslen += 1
    assert len(result_array) == reslen
    assert sum(result_array) == reslen
    output_file_path.unlink()
### test with index 
    def index_values(i):
        return i

    file_length = 102445
    segment_len = 10
    result_array = mock_data.mock_result_data_file(index_values,
                                                   output_file_path,
                                                   file_length,
                                                   segment_len=segment_len)
    output_file_path = output_file_path.with_suffix('.npy')

    result_array_file = np.load(output_file_path)
    assert np.array_equal(result_array, result_array_file)

    reslen = file_length // segment_len
    if file_length % segment_len != 0:
        reslen += 1
    assert len(result_array) == reslen
    assert sum(result_array) == sum(range(reslen))
    output_file_path.unlink()
### test channel count 
    channel_count = 2
    # def index_values(i):
        # return i

    file_length = 102445
    segment_len = 10
    result_array = mock_data.mock_result_data_file(index_values,
                                                   output_file_path,
                                                   file_length,
                                                   segment_len=segment_len,
                                                   channel_count=channel_count)
    output_file_path = output_file_path.with_suffix('.npy')

    result_array_file = np.load(output_file_path)
    assert np.array_equal(result_array, result_array_file)

    reslen = file_length // segment_len
    if file_length % segment_len != 0:
        reslen += 1
    assert len(result_array) == reslen
    assert np.sum(result_array) == sum(range(reslen))*channel_count
    assert result_array.shape == (reslen,channel_count)
    output_file_path.unlink()


def test_mock_results_4input_files():
    """Test for mock_files.mock_results_4input_files

        Checks if all there enough number of files.
        TODO add more tests such as consistency of stored values.
    """

    def ones(i):
        del i
        return 1

    row_count = 12
    # segment_len = 10
    results_tag = 'YYY'
    file_length_limit_str = '01:00:00'
    file_length_limit_seconds = 1 * 60 * 60
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    func_output_path = IO_mock_data_path / 'mock_results_4input_files' / 'outputs'

    resulting_output_file_paths = mock_data.mock_results_4input_files(
        file_properties_df,
        ones,
        func_output_path,
        results_tag_id=results_tag,
        file_length_limit=file_length_limit_str)

    assert resulting_output_file_paths
    assert len(resulting_output_file_paths)>=len(file_properties_df.index)

    found_output_files = fileUtils.list_files(search_path=str(func_output_path),
                                              ignore_folders=None,
                                              file_name='*.npy')
    for file_name in found_output_files[:10]:
        # check if there are two underscores before XXX
        file_name = Path(file_name)
        assert file_name.stem.split('_')[-2] != ''
        assert file_name.parent.stem.split('_')[-2] != ''

    for _, row in file_properties_df.iterrows():

        folder4row = fileUtils.standard_path_style(
            func_output_path,
            row,
            sub_directory_addon=results_tag,
        )
        assert folder4row.exists()

    expected_file_count = 0
    for _, row in file_properties_df.iterrows():
        expected_file_count_row = (row.durationSec // file_length_limit_seconds)
        if (row.durationSec % file_length_limit_seconds) != 0:
            expected_file_count_row += 1
        expected_file_count += expected_file_count_row

    assert expected_file_count == len(found_output_files)

    shutil.rmtree(str(func_output_path))
