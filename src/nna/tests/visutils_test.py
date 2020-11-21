"""Tests for visutils.py"""

import pytest
from nna import visutils
import mock_data
import shutil

#from matplotlib
import cycler

from testparams import INPUTS_OUTPUTS_PATH

IO_visutils_path = INPUTS_OUTPUTS_PATH / 'visutils'


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
    selected_tag_name = ['_' + i for i in model_tag_names]

    def main_of_test(fill_value_func):

        for tag_name in selected_tag_name:
            _ = mock_data.mock_results_4input_files(
                file_properties_df,
                fill_value_func,
                func_output_path,
                results_tag_id=tag_name,
            )

        global_columns = model_tag_names  #selected_areas+weather_cols
        freq = '270min'

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
            prob2binaryFlag=False,
        )

        shutil.rmtree(str(func_output_path))
        return df_dict, no_result_paths

    # def ones(i):
    #     del i
    #     return 1
    # df_dict, no_result_paths = main_of_test(ones)

    # assert len(no_result_paths) == 0
    # assert sorted(selected_areas)==sorted(list(df_dict.keys()))
    # for location_id in df_dict.keys():
    #     counts,sums = df_dict[location_id]
    #     assert sum(counts.values)[0]==sum(sums.values)[0]

    def index_index(i):
        return i

    df_dict, no_result_paths = main_of_test(index_index)

    for location_id in df_dict.keys():
        counts, sums = df_dict[location_id]
        # (n * (n + 1)) / 2, sums should be found by this formula but
        # since results are divided into smaller files, index starts from zero again
        # so results are slightly less than that.
        counts_to_approxim_sum = (counts[counts["XXX"] > 0] *
                                  (counts[counts["XXX"] > 0] + 1)) / 2

        assert (counts_to_approxim_sum[counts_to_approxim_sum["XXX"] > 0] >=
                sums[sums["XXX"] > 0]).all()[0]
        # sums should be bigger than counts,
        assert (counts[counts["XXX"] > 0] <= sums[sums["XXX"] > 0]).all()[0]

    # return df_dict, no_result_paths


def test_get_cycle():

    output = visutils.get_cycle("tab20c")
    assert isinstance(output, cycler.Cycler)
    assert output.keys == {'color'}