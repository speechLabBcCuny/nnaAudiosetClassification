"""Tests for visutils.py"""

import pytest
from nna import visutils
import mock_data

from testparams import INPUTS_OUTPUTS_PATH

IO_visutils_path = INPUTS_OUTPUTS_PATH / 'visutils'


def test_file2TableDict():
    #                           selected_areas,
    #                         model_tag_names,
    #                         globalindex,
    #                         globalcolumns,
    #                         file_properties_df,
    #                         freq,
    #                         dataFreq="10S",
    #                         dataThreshold=0.5,
    #                         channel=1,
    #                         gathered_results_perTag=None,
    #                         result_path=None,
    #                         file_name_addon="",
    #                         prob2binaryFlag=True

    # selected_areas = ['anwr', 'prudhoe']
    model_tag_names = [
        'XXX',
    ]  #'YYY']

    def ones(i):
        del i
        return 1

    row_count = 100
    # file_length_limit_seconds = 1 * 60 * 60
    file_properties_df = mock_data.mock_file_properties_df(row_count)
    selected_areas = set(file_properties_df.locationId.values)
    func_output_path = IO_visutils_path / 'file2TableDict' / 'outputs'
    selected_tag_name = ["_" + i for i in model_tag_names]

    for tag_name in selected_tag_name:
        _ = mock_data.mock_results_4input_files(
            file_properties_df,
            ones,
            func_output_path,
            results_tag_id=tag_name,
        )
    # print(file_properties_df)

    global_columns = selected_tag_name  #selected_areas+weather_cols
    freq = '270min'

    # # file length based time index
    # if freq == "continous":
    #     fileTimeIndexSeries = get_time_index_per_file(selected_area,
    #                                                   file_properties_df,freq)
    #     globalindex = fileTimeIndexSeries
    # #fixed freq based  time index
    # else:
    # print(file_properties_df)
    globalindex, all_start, all_end = visutils.createTimeIndex(
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
        file_name_addon=selected_tag_name[0],
        # prob2binaryFlag=True,
    )
    return df_dict, no_result_paths


# df_dict, no_result_paths = test_file2TableDict()
