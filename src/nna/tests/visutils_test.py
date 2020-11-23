"""Tests for visutils.py"""

import datetime
import numpy as np
import pytest
from nna import visutils
from nna.tests import mock_data
import shutil

#from matplotlib
import cycler
from nna.tests.testparams import INPUTS_OUTPUTS_PATH

IO_visutils_path = INPUTS_OUTPUTS_PATH / 'visutils'

from pandas._testing import assert_frame_equal


def test_get_cycle():

    output = visutils.get_cycle("tab20c")
    assert isinstance(output, cycler.Cycler)
    assert output.keys == {'color'}


test_data_create_time_index = [
    (
        100,
        '270min',
        ('270T',),
    ),
    (
        120,
        '2H',
        ('2H',),
    ),
]


@pytest.mark.parametrize('row_count,freq,expected', test_data_create_time_index)
def test_create_time_index(row_count, freq, expected):

    # row_count = 100
    # freq = '270min'

    file_properties_df = mock_data.mock_file_properties_df(row_count)
    selected_location_ids = set(file_properties_df.locationId.values)
    globalindex, all_start, all_end = visutils.create_time_index(
        selected_location_ids, file_properties_df, freq)

    assert all_start == globalindex[0]
    assert all_end <= globalindex[-1]
    assert globalindex.freq.freqstr == expected[0]


def test_load_results():
    """ test for visutils.load_npy_files.

        checks shape approximatly,
        checks shape of the array.
    """
    tag_name = 'XXX'
    channel_count = 2
    row_count = 10
    # file_length_limit_seconds = 1 * 60 * 60
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    func_output_path = IO_visutils_path / 'load_results' / 'outputs'

    def ones(i):
        del i
        return 1

    fill_value_func = ones

    resulting_output_file_paths = mock_data.mock_results_4input_files(
        file_properties_df,
        fill_value_func,
        func_output_path,
        results_tag_id=tag_name,
        channel_count=channel_count,
    )

    results = visutils.load_npy_files(resulting_output_file_paths)
    assert results.shape[0] > len(resulting_output_file_paths)
    assert results.shape[1] == channel_count

    shutil.rmtree(str(func_output_path))


def test_prob2binary():
    ones_array_len = 100
    ones_array = np.ones((ones_array_len, 2))
    segment_len = 10
    #  2-D array
    res_array = visutils.prob2binary(ones_array.copy(),
                                     segment_len=segment_len,
                                     channel=2)

    assert res_array.shape == (ones_array_len / segment_len,)
    assert np.sum(res_array) == ones_array_len / segment_len
    # with 1-D array
    ones_array_len = 100
    ones_array = np.ones((ones_array_len))
    segment_len = 10

    res_array = visutils.prob2binary(ones_array.copy(),
                                     segment_len=segment_len,
                                     channel=1)

    assert res_array.shape == (ones_array_len / segment_len,)
    assert np.sum(res_array) == ones_array_len / segment_len


#TODO
# # file length based time index
# if freq == "continous":
#     fileTimeIndexSeries = get_time_index_per_file(selected_area,
#                                                   file_properties_df,freq)
#     globalindex = fileTimeIndexSeries
# #fixed freq based  time index
# else:
# print(file_properties_df)
def test_file2TableDict():
    # selected_areas = ['anwr', 'prudhoe']
    model_tag_names = [
        'XXX',
    ]  #'YYY']

    row_count = 100
    # file_length_limit_seconds = 1 * 60 * 60
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    selected_areas = set(file_properties_df.locationId.values)
    func_output_path = IO_visutils_path / 'file2TableDict' / 'outputs'

    # selected_tag_name = ['_' + i for i in model_tag_names]

    def main_of_test(fill_value_func):

        for tag_name in model_tag_names:
            _ = mock_data.mock_results_4input_files(
                file_properties_df,
                fill_value_func,
                func_output_path,
                results_tag_id=tag_name,
            )

        global_columns = model_tag_names  #selected_areas+weather_cols
        freq = '270min'

        globalindex, all_start, all_end = visutils.create_time_index(
            selected_areas, file_properties_df, freq)
        del all_start, all_end

        df_dict, no_result_paths = visutils.file2TableDict(
            selected_areas,
            model_tag_names,
            globalindex,
            global_columns,
            file_properties_df,
            freq,
            # dataFreq="10S",
            # dataThreshold=0.5,
            # channel=1,
            # gathered_results_perTag=None,
            result_path=func_output_path,
            prob2binaryFlag=False,
        )

        shutil.rmtree(str(func_output_path))
        return df_dict, no_result_paths

    # test for filling array with ones
    def ones(i):
        del i
        return 1

    df_dict, no_result_paths = main_of_test(ones)
    #return df_dict, no_result_paths

    assert len(no_result_paths) == 0
    assert sorted(selected_areas) == sorted(list(df_dict.keys()))
    for location_id in df_dict.keys():
        counts, sums = df_dict[location_id]
        assert sum(counts.values)[0] == sum(sums.values)[0]

    # test for filling arrays with index of the file read.
    def index_index(i):
        return i

    df_dict, no_result_paths = main_of_test(index_index)
    # return df_dict, no_result_paths
    assert not no_result_paths
    for location_id in df_dict.keys():
        counts, sums = df_dict[location_id]
        # THAT DOES NOT WORK IF FILE SPLIT INTO two bins then bigger values
        # are in another bin which we assume starts from 0 again.

        # # (n * (n + 1)) / 2, sums should be found by this formula but
        # # since results are divided into smaller files, index starts from zero again
        # # so results are slightly less than that.
        # counts_to_approxim_sum = (counts[counts['XXX'] > 0] *
        #                           (counts[counts['XXX'] > 0] + 1)) / 2

        # approx_sum_flag = (
        #     counts_to_approxim_sum[counts_to_approxim_sum['XXX'] > 0] >=
        #     sums[sums['XXX'] > 0]).all()[0]
        # assert approx_sum_flag
        # sums should be bigger than counts,
        assert (counts[counts['XXX'] > 0] <= sums[sums['XXX'] > 0]).all()[0]


import pandas as pd


def test_reverse_df_dict():
    df_dict = {
        "area_name1": (pd.DataFrame({
            'XXX': [1, 2],
            'YYY': [3, 4]
        }), pd.DataFrame({
            'XXX': [10, 20],
            'YYY': [30, 40]
        })),
        "area_name2": (pd.DataFrame({
            'XXX': [5, 6],
            'YYY': [7, 8]
        }), pd.DataFrame({
            'XXX': [50, 60],
            'YYY': [70, 80]
        })),
    }
    df_dict_reverse_expected = {
        "XXX": (pd.DataFrame({
            'area_name1': [1, 2],
            'area_name2': [5, 6]
        }), pd.DataFrame({
            'area_name1': [10, 20],
            'area_name2': [50, 60]
        })),
        "YYY": (pd.DataFrame({
            'area_name1': [3, 4],
            'area_name2': [7, 8]
        }), pd.DataFrame({
            'area_name1': [30, 40],
            'area_name2': [70, 80]
        })),
    }

    df_dict_reverse = visutils.reverse_df_dict(df_dict)

    assert df_dict_reverse.keys() == df_dict_reverse_expected.keys()
    for key in df_dict_reverse:
        sum, count = df_dict_reverse[key]
        sum_exp, count_exp = df_dict_reverse_expected[key]
        assert_frame_equal(sum, sum_exp)
        assert_frame_equal(count, count_exp)


def test_time_index_by_close_recordings():
    """ Test for visutils.time_index_by_close_recordings.

    """
    row_count = 1000
    max_time_distance_allowed = datetime.timedelta(minutes=5)

    file_properties_df = mock_data.mock_file_properties_df(row_count)

    file_time_index_series = visutils.time_index_by_close_recordings(
        file_properties_df,)
    for m in range(len(file_time_index_series) - 1):
        diff = (file_time_index_series[m + 1] - file_time_index_series[m])
        assert diff >= max_time_distance_allowed
    assert len(file_properties_df.index) >= len(file_time_index_series)
    assert file_properties_df.timestampEnd[-1] == file_time_index_series.iloc[
        -1]
    assert file_properties_df.timestamp[0] == file_time_index_series.iloc[0]
    return file_time_index_series, file_properties_df
