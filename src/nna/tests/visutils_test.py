"""Tests for visutils.py"""
import faulthandler; faulthandler.enable()
import csv
import datetime
import shutil

#from matplotlib
import cycler
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
import pytest
import matplotlib.pylab as pl

from nna import visutils
from nna.tests import mock_data
from nna.tests.testparams import INPUTS_OUTPUTS_PATH

IO_visutils_path = INPUTS_OUTPUTS_PATH / 'visutils'


def test_get_cycle():
    output = visutils.get_cycle('tab20c')
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
    (
        120,
        'continous',
        ('10S',),
    ),
]


@pytest.mark.parametrize('row_count,output_data_freq,expected',
                         test_data_create_time_index)
def test_create_time_index(row_count, output_data_freq, expected):
    del expected
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    globalindex = visutils.create_time_index(file_properties_df,
                                             output_data_freq)

    # assert all_start == globalindex[0]
    # assert all_end <= globalindex[-1]
    # assert globalindex.freq.freqstr == expected[0]
    assert len(globalindex) > 1


def test_row_data_2_df():

    row = {
        'timestamp':
            datetime.datetime(year=2019,
                              month=5,
                              day=2,
                              hour=2,
                              minute=2,
                              second=2)
    }
    row = pd.Series(row)
    data = np.ones((500, 1))
    input_data_freq = '10S'
    model_tag_name = 'XXX'
    df_afile = visutils.row_data_2_df(row, data, input_data_freq,
                                      model_tag_name)

    assert df_afile.shape == data.shape
    assert df_afile.iloc[11].name - df_afile.iloc[
        10].name == pd.tseries.frequencies.to_offset(input_data_freq)
    assert list(df_afile.columns.values) == [model_tag_name]


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


def test_file2TableDict():  # pylint: disable=invalid-name
    # selected_areas = ['anwr', 'prudhoe']
    model_tag_names = [
        'XXX',
    ]  #'YYY']

    row_count = 100
    # file_length_limit_seconds = 1 * 60 * 60
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    # selected_areas = set(file_properties_df.locationId.values)
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

        output_data_freq = '270min'

        df_dict, no_result_paths = visutils.file2TableDict(
            model_tag_names,
            file_properties_df,
            input_data_freq='10S',
            output_data_freq=output_data_freq,
            prob2binary_threshold=0.5,
            # channel=1,
            # gathered_results_perTag=None,
            result_path=func_output_path,
            prob2binary_flag=False,
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
    # assert sorted(selected_areas) == sorted(list(df_dict.keys()))
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
        # # since results are divided into smaller files, index starts
        # # from zero again
        # # so results are slightly less than that.
        # counts_to_approxim_sum = (counts[counts['XXX'] > 0] *
        #                           (counts[counts['XXX'] > 0] + 1)) / 2

        # approx_sum_flag = (
        #     counts_to_approxim_sum[counts_to_approxim_sum['XXX'] > 0] >=
        #     sums[sums['XXX'] > 0]).all()[0]
        # assert approx_sum_flag
        # sums should be bigger than counts,
        for tag_name in model_tag_names:
            # counts[tag_name]==1 because when there is single item in a bin
            # sum is smaller than total count because item's value is 0
            assert ((counts[tag_name] <= sums[tag_name]) |
                    counts[tag_name] == 1).all()


def test_reverse_df_dict():
    df_dict = {
        'area_name1': (pd.DataFrame({
            'XXX': [1, 2],
            'YYY': [3, 4]
        }), pd.DataFrame({
            'XXX': [10, 20],
            'YYY': [30, 40]
        })),
        'area_name2': (pd.DataFrame({
            'XXX': [5, 6],
            'YYY': [7, 8]
        }), pd.DataFrame({
            'XXX': [50, 60],
            'YYY': [70, 80]
        })),
    }
    df_dict_reverse_expected = {
        'XXX': (pd.DataFrame({
            'area_name1': [1, 2],
            'area_name2': [5, 6]
        }), pd.DataFrame({
            'area_name1': [10, 20],
            'area_name2': [50, 60]
        })),
        'YYY': (pd.DataFrame({
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
        sum_df, count_df = df_dict_reverse[key]
        sum_exp, count_exp = df_dict_reverse_expected[key]
        assert_frame_equal(sum_df, sum_exp)
        assert_frame_equal(count_df, count_exp)


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


def test_export_raw_results_2_csv():
    """ Test for visutils.export_raw_results_2_csv.
    """

    func_output_path = IO_visutils_path / 'export_raw_results_2_csv' / 'outputs'
    func_input_path = IO_visutils_path / 'export_raw_results_2_csv' / 'inputs'
    row_count = 100
    tag_names = ['XXX']
    channel_count = 1

    file_properties_df = mock_data.mock_file_properties_df(row_count)
    selected_location_ids = list(set(file_properties_df.locationId.values))

    def ones(i):
        del i
        return 1

    fill_value_func = ones

    for tag_name in tag_names:
        resulting_output_file_paths = mock_data.mock_results_4input_files(
            file_properties_df,
            fill_value_func,
            func_input_path,
            results_tag_id=tag_name,
            channel_count=channel_count,
        )
        del resulting_output_file_paths

    csv_files_written, no_result_paths = visutils.export_raw_results_2_csv(
        func_output_path,
        tag_names,
        file_properties_df,
        # input_data_freq="10S",
        # output_data_freq="10S",
        # raw2prob_threshold=0.5,
        channel=channel_count,
        # gathered_results_per_tag=None,
        result_files_folder=func_input_path,
        # prob2binary_flag=True,
    )
    assert not no_result_paths
    assert len(csv_files_written) == len(selected_location_ids) * len(tag_names)

    with open(csv_files_written[0], newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)
        header, lines = lines[0], lines[1:]
    assert set(['1']) == set(i[1] for i in lines)
    second = datetime.datetime.strptime(lines[11][0], '%Y-%m-%d_%H:%M:%S')
    first = datetime.datetime.strptime(lines[10][0], '%Y-%m-%d_%H:%M:%S')
    assert (second - first).seconds == 10
    assert header[1] == csv_files_written[0].stem.split('_')[-1]

    shutil.rmtree(func_input_path)
    shutil.rmtree(func_output_path)
    return csv_files_written, no_result_paths


def test_vis_preds_with_clipping():
    '''Test vis generation.

        Requires manual inspection of generated graphs.
    
    '''
    func_output_path = IO_visutils_path / 'vis_preds_with_clipping' / 'outputs'
    func_input_path = IO_visutils_path / 'vis_preds_with_clipping' / 'inputs'
    clipping_results_path = func_input_path / 'clipping_files'
    clipping_results_path.mkdir(parents=True, exist_ok=True)

    row_count = 100
    tag_names = ['XXX']
    id2name = {'XXX': 'XXX_name'}

    channel_count = 1
    input_data_freq = '10S'
    output_data_freq = '270min'

    file_properties_df = mock_data.mock_file_properties_df(row_count,region_location_count=1)
    # selected_location_ids = list(set(file_properties_df.locationId.values))
    region_location = zip(list(file_properties_df.region.values),
                            list(file_properties_df.locationId.values))
    region_location = list(set(region_location))

    def ones(i):
        del i
        return 1

    fill_value_func = ones

    for tag_name in tag_names:
        resulting_output_file_paths = mock_data.mock_results_4input_files(
            file_properties_df,
            fill_value_func,
            func_input_path,
            results_tag_id=tag_name,
            channel_count=channel_count,
        )
        del resulting_output_file_paths

    cmap = pl.cm.tab10
    a_cmap = cmap
    my_cmaps = visutils.add_normal_dist_alpha(a_cmap)
    for region, location_id in region_location[0:1]:
        print(region, location_id)
        file_prop_df_filtered = file_properties_df[file_properties_df.region ==
                                                    region]
        file_prop_df_filtered = file_properties_df[file_properties_df.locationId
                                                    == location_id]
        region_location_name = '-'.join([region, location_id])
        _ = mock_data.mock_clipping_results_dict_file(
            file_prop_df_filtered,
            region_location_name,
            clipping_results_path,
        )
        visutils.vis_preds_with_clipping(region,
                                            location_id,
                                            file_prop_df_filtered,
                                            input_data_freq,
                                            output_data_freq,
                                            tag_names,
                                            my_cmaps,
                                            func_input_path,
                                            clipping_results_path,
                                            func_output_path,
                                            id2name,
                                            clipping_threshold=1.0)
