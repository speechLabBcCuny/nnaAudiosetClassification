"""Visualizations functions

"""
from typing import Dict, List, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import cycler

# from nna.fileUtils import list_files
from nna import fileUtils


# https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle
def get_cycle(
    cmap: Union[matplotlib.colors.Colormap, str, None],
    color_count: int = None,
    use_index: str = "auto",
) -> cycler.cycler:
    """Cycle colors to use with  matplotlib.

    Usage:
        Continous:
            import matplotlib.pyplot as plt
            N = 6
            plt.rcParams["axes.prop_cycle"] = get_cycle("viridis", N)

            fig, ax = plt.subplots()
            for i in range(N):
                ax.plot([0,1], [i, 2*i])

            plt.show()
        discrete case:
            import matplotlib.pyplot as plt

            plt.rcParams["axes.prop_cycle"] = get_cycle("tab20c")

            fig, ax = plt.subplots()
            for i in range(15):
                ax.plot([0,1], [i, 2*i])

            plt.show()
    """
    if isinstance(cmap, str):
        if use_index == "auto":
            use_index = cmap in [
                "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1",
                "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"
            ]
        cmap = matplotlib.cm.get_cmap(cmap)
    if not color_count:
        color_count = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(color_count)) % cmap.N
        return plt.cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0.2, 1, color_count))
        return plt.cycler("color", colors)


def time_index_by_close_recordings(file_properties_df,
                                   max_time_distance_allowed=5):
    """Cal time indexes of bins with recordings not far to each other more than.

        This function is for creating a time index of bins used to group data
        but specifically grouping close recordings. So that each bin represents
        a group of recording that was continous or close enough.
        Args:
            file_properties_df: see file_properties_df
            max_time_distance_allowed: how far one recording's ending can be
                                        further than next one's start
    """
    #TODO add sensitivity
    file_properties_df.sort_values(by=["timestamp"], inplace=True)
    rowiterator = file_properties_df.iterrows()
    # use first item as initialization, ahead of for loop
    n = next(rowiterator)
    start = n[1].timestamp
    beginning = start
    end = n[1].timestampEnd
    file_time_index = []
    is_head = False
    for row in rowiterator:
        # if end of previous file not equal to start of the second one
        if (row[1].timestamp -
                end) > timedelta(minutes=max_time_distance_allowed):
            # add previous one to list and make new one the beginning of
            # continous recording
            file_time_index.append(beginning)
            beginning = row[1].timestamp
            is_head = True
    # if they are equal, they should be in the same bin, so keep going
        else:
            is_head = False
        start = row[1].timestamp
        end = row[1].timestampEnd
    # If last one is a head add to the list

    if is_head:
        file_time_index.append(beginning)
    # add end since last bin border should be bigger than all data
    file_time_index.append(end)
    file_time_index_series = pd.Series(file_time_index)
    return file_time_index_series


def create_time_index(
        file_properties_df,
        output_data_freq) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Create time index boundaries of the bins with size of frequency.

        Generate a time index as boundary of the bins to gather data by time.
        Bins have the length of the freq ex: 270 minutes
        Start and end times are earliest and latest points in time within the
        recordings of the selected_areas.

        Args:
            file_properties_df: See file_properties_df.
            freq: frequency of the time to be used to divide time into bins
        Returns:
            Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]
    """
    # FIND earliest and latest time for time scale
    # lists in selected_areas_dict is ordered by time
    if output_data_freq == "continous":
        globalindex = time_index_by_close_recordings(file_properties_df)
    else:
        times = sorted(file_properties_df["timestamp"].values)
        all_start = times[0]
        all_end = times[-1]
        all_end_padded = all_end + pd.tseries.frequencies.to_offset(
            output_data_freq)
        globalindex = pd.date_range(all_start,
                                    all_end_padded,
                                    freq=output_data_freq)

    return globalindex


# result_path="/scratch/enis/data/nna/real/"
def load_npy_files(all_segments: Union[List[Union[str, Path]], Union[str,
                                                                     Path]],):
    """Load list of numpy files and concatanate in order.
        args:
            all_segments: list of npy file paths or single path
        return:
            np.Array
    """
    if not isinstance(all_segments, list):
        all_segments = [all_segments]

    results = []
    for filename in all_segments:
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"file does not exists: {str(filename)} ")
        else:
            data = np.load(filename)
        # if prob2binary_flag:
        # data = prob2binary(data, threshold=threshold, channel=channel)
        results.append(data)

    results = np.concatenate(results)

    return results


def prob2binary(
    result: np.array,
    threshold: float = 0.5,
    channel: int = 1,
    segment_len=10,
) -> np.array:
    """Applies threshold to group of values, calculate single.

            ! padds array so it is length is divisible by segment_len

        args:
            result: numpy array to be modified
            threshld: threshold value
            channel: last dimension of the array, how many channels sound has
                     axis=0
    """
    if len(result.shape) > 1 and result.shape[-1] != channel:
        raise ValueError(
            "input array should have same dimension size with channel")
    if channel == 2:
        result = np.max(result, axis=1)
    if channel > 2:
        raise NotImplementedError(
            f"channel bigger than 2 is not implemented given {channel}")
    result[result > threshold] = 1
    result[result <= threshold] = 0
    remainder = (result.size % segment_len)
    if remainder > 0:
        pad_width = segment_len - remainder
        result = np.pad(result, (0, pad_width),
                        "constant",
                        constant_values=(0, 0))

    result = result[:(result.size // segment_len) * segment_len]
    result = result.reshape(segment_len, -1).max(axis=0)
    return result


def load_data_of_row(
    result_path,
    row,
    model_tag_name,
    gathered_results_per_tag=None,
    afile=None,
):
    """Load data belonging to a sound file (row from file_properties_df).
    """
    if gathered_results_per_tag:
        data = gathered_results_per_tag[model_tag_name].get(afile,
                                                            np.empty(0))[:]
    else:
        check_folder = fileUtils.standard_path_style(
            result_path,
            row,
            sub_directory_addon=model_tag_name,
        )
        check_folder_str = str(check_folder) + "/"
        all_segments = fileUtils.list_files(check_folder_str)
        all_segments.sort()
        if not all_segments:
            data = np.empty(0)
        else:
            data = load_npy_files(all_segments)

    return data


def row_data_2_df(file_object, data, input_data_freq, model_tag_name):
    """Creates DF from data belong to the same file from file_properties_df.
    """
    if data.shape[-1] != 1 and len(data.shape) == 2:
        raise ValueError("data should have single channel," +
                         "double channel not supported")

    # file_object = file_properties_df.loc[afile]
    start = file_object.timestamp
    input_freq_offset = pd.tseries.frequencies.to_offset(input_data_freq)
    end = start + (input_freq_offset * (len(data) - 1))
    index = pd.date_range(start, end, freq=input_data_freq)
    df_afile = pd.DataFrame(data, index=index, columns=[model_tag_name])
    return df_afile


def afile_df_2_counts_sums(globalindex, df_afile, df_count, df_sums):
    """ Adds data from df_afile to general count and sums dataframes.
    """
    # old version
    # df_afile_grouped = df_afile.groupby([pd.Grouper(freq=freq)])
    # counts=df_afile_grouped.count()
    # sums=df_afile_grouped.sum()
    # print(df_afile.index)
    global_index_start = globalindex.searchsorted(df_afile.index[0],)
    if global_index_start != 0:
        global_index_start = global_index_start - 1
    global_index_end = globalindex.searchsorted(df_afile.index[-1])
    if global_index_end == global_index_start:
        global_index_end = global_index_end + 1
    the_bins = pd.cut(df_afile.index,
                      globalindex[global_index_start:global_index_end + 1])

    # the_bins=pd.cut(df_afile.index,globalindex)
    df_afile_grouped = df_afile.groupby(the_bins)
    sums = df_afile_grouped.agg("sum")
    counts = df_afile_grouped.agg("count")
    sums.set_index(sums.index.categories.left, inplace=True)
    counts.set_index(counts.index.categories.left, inplace=True)

    df_count = df_count.add(counts, fill_value=0)  # df_count.update(counts)
    df_sums = df_sums.add(sums, fill_value=0)  # df_sums.update(sums)
    return df_count, df_sums


# TODO BUG HANDLE same locationId from multiple regions
def file2TableDict(  # pylint: disable=invalid-name
        model_tag_names: List[str],
        #globalcolumns: List[str],
        file_properties_df: pd.DataFrame,
        input_data_freq: str = "10S",
        output_data_freq: str = "10S",
        prob2binary_threshold: float = 0.5,
        channel: int = 1,
        gathered_results_per_tag: Union[Dict, None] = None,
        result_path: Union[str, None] = None,
        prob2binary_flag: bool = True) -> Tuple[Dict, List]:
    """Reduce results by dataFreq from multiple files into a pd.DataFrame.

        For all selected_areas (locationId), finds results for each tag from
        gathered_results_perTag: Dict or path (result_path) to pkl/npy files
        storing results. Reduces them by dataFreq and stores in pd.DataFrame
        Each result file named after it's original input file.
        Results are numpy arrays storing binary or decimal values.

        Args:
            model_tag_names: Tags (XXX,CARS) to search results for.
                            Name of the columns for the resulting DataFrame
                            actually same with model_tag_name.
            file_properties_df: Original input file's properties database to
                                be used a reference to find all files in the
                                given areas.
            input_data_freq = "10S": Freq of the data provided.
            output_data_freq = "10S": Freq of the output data.
            dataThreshold=0.5: if results are in probability, then threshold
                            used to calculate 0/1.
            channel=1: number of the channels of the data, # of dimensions
            gathered_results_perTag=None: Results stored in a
                                        dict["tag_name":
                                            {Path(a_file):np.Array,},]
            result_path=None: Location to look for results for each orig file
            prob2binaryFlag=True: Should results converted to binary.

        Returns:
            Results dictionary and list of file_paths with no results.
            Dictionary is structured as:
                {"area_name1":(df_count:pd.DataFrame,df_sums:pd.DataFrame),
                 "area_name2":(df_count:pd.DataFrame,df_sums:pd.DataFrame),}
            df_count and df_sums have global_index and model_tag_names as their
            indexes and column names.

    """
    global_index = create_time_index(file_properties_df, output_data_freq)

    selected_location_ids = set(file_properties_df.locationId.values)
    df_dict = {key: None for key in selected_location_ids}

    area_prev = None
    no_result_paths = []
    for location_id, _, df_afile in load_data_yield(
            model_tag_names, file_properties_df, input_data_freq,
            prob2binary_threshold, channel, gathered_results_per_tag,
            result_path, prob2binary_flag):
        if location_id is None:
            no_result_paths.append(df_afile)
            continue
        if area_prev != location_id:
            if area_prev is not None:
                df_dict[area_prev] = (
                    df_count.copy(),  #type: ignore
                    df_sums.copy(),  #type: ignore
                )  #type: ignore

            area_prev = location_id
            df_sums = pd.DataFrame(index=global_index,
                                   columns=model_tag_names).fillna(0)
            df_count = pd.DataFrame(index=global_index,
                                    columns=model_tag_names).fillna(0)

        df_count, df_sums = afile_df_2_counts_sums(
            global_index,
            df_afile,
            df_count,  #type: ignore
            df_sums)  #type: ignore

    # last results
    df_dict[area_prev] = (df_count.copy(), df_sums.copy())  #type: ignore
    return df_dict, no_result_paths


def load_data_yield(tag_names: List[str],
                    file_properties_df: pd.DataFrame,
                    input_data_freq: str = "10S",
                    prob2binary_threshold: float = 0.5,
                    channel: int = 1,
                    gathered_results_per_tag: Union[Dict, None] = None,
                    result_path: Union[str, None] = None,
                    prob2binary_flag: bool = True):
    """Iterates location_ids, tag_names then yield the loaded file.
    """
    if (gathered_results_per_tag is None) and (result_path is None):
        print("ERROR: gathered_results_perTag or" +
              "(result_path and subDirectoryAddon )should be defined")
        return (None, None)

    selected_location_ids = set(file_properties_df.locationId.values)
    # no_result_paths = []

    for _, location_id in enumerate(selected_location_ids):
        filtered_by_location_id = file_properties_df[file_properties_df.site_id
                                                     == location_id]
        for a_tag_name in tag_names:
            for afile, row in filtered_by_location_id.iterrows():
                afile = Path(afile)
                data = load_data_of_row(result_path, row, a_tag_name,
                                        gathered_results_per_tag, afile)

                if data.size == 0:
                    yield (None, None, afile)

                if prob2binary_flag:
                    data = prob2binary(data,
                                       threshold=prob2binary_threshold,
                                       channel=channel)

                df_afile = row_data_2_df(row, data, input_data_freq, a_tag_name)

                yield location_id, a_tag_name, df_afile


def reverse_df_dict(df_dict):
    """Switch TAG and AREA keys in the dataframe dict.
        df_dict is
                    {"area_name1":(df_count:pd.DataFrame,df_sums:pd.DataFrame),
                    "area_name2":(df_count:pd.DataFrame,df_sums:pd.DataFrame),}
                    then DataFrame has columns tag_name1, tag_name2 ex:(XXX)
        df_dict_reverse is
                    {"tag_name1":(df_count:pd.DataFrame,df_sums:pd.DataFrame),
                    "tag_name2":(df_count:pd.DataFrame,df_sums:pd.DataFrame),}
                    then DataFrame has columns area_name1, area_name2
    """

    df_dict_reverse = {}  # type: ignore
    location_ids = list(df_dict.keys())
    model_tag_names = df_dict[location_ids[0]][0].columns
    for area in df_dict.keys():
        df_count, df_sums = df_dict[area]
        for tagname in model_tag_names:
            df_dict_reverse.setdefault(tagname, [[], []])
            df_dict_reverse[tagname][0].append(df_count[tagname])
            df_dict_reverse[tagname][1].append(df_sums[tagname])

    for tagname in model_tag_names:
        df_count = pd.concat(df_dict_reverse[tagname][0], axis=1)
        df_sums = pd.concat(df_dict_reverse[tagname][1], axis=1)
        df_count.columns, df_sums.columns = location_ids, location_ids
        df_dict_reverse[tagname] = df_count, df_sums

    return df_dict_reverse


def export_raw_results_2_csv(output_folder_path,
                             tag_names,
                             file_properties_df,
                             input_data_freq="10S",
                             output_data_freq="10S",
                             raw2prob_threshold=0.5,
                             channel=1,
                             gathered_results_per_tag=None,
                             result_files_folder=None,
                             prob2binary_flag=True):
    """Export results to a csv file without any grouping.
    """
    if output_data_freq != "10S":
        raise ValueError(
            "this function does not do aggregation, set dataFreq to 10S")

    def export_list_of_df_2csv(df_raw_list, csv_files_written):
        if df_raw_list:
            df_raw = pd.concat(df_raw_list)
            df_raw = df_raw.sort_index()
            csv_file_name = "_".join([location_id, model_tag_name + ".csv"])  #pylint: disable=undefined-loop-variable
            output_folder_path.mkdir(exist_ok=True, parents=True)
            df_raw.to_csv((output_folder_path / csv_file_name),
                          index_label="TimeStamp",
                          header=[model_tag_name])  #pylint: disable=undefined-loop-variable
            csv_files_written.append((output_folder_path / csv_file_name))
        return csv_files_written

    csv_files_written = []
    no_result_paths = []

    model_tag_name_prev = None
    location_id_prev = None
    df_raw_list = []

    for location_id, model_tag_name, df_afile in load_data_yield(
            tag_names, file_properties_df, input_data_freq, raw2prob_threshold,
            channel, gathered_results_per_tag, result_files_folder,
            prob2binary_flag):
        # no results, yields None,None,df_afile
        if location_id is None:
            no_result_paths.append(df_afile)

        # if location_id or model_tag_name changed, save the buffer
        # in the first one df_raw_list will be empty anyway
        if (location_id_prev, model_tag_name_prev) != (location_id,
                                                       model_tag_name):
            if df_raw_list:
                csv_files_written = export_list_of_df_2csv(
                    df_raw_list, csv_files_written)
            model_tag_name_prev = model_tag_name
            location_id_prev = location_id
            df_raw_list = []

        df_raw_list.append(df_afile)

    # save last yielded batch
    csv_files_written = export_list_of_df_2csv(df_raw_list, csv_files_written)
    return csv_files_written, no_result_paths


def add_normal_dist_alpha(a_cmap):
    # Choose colormap
    # cmap = pl.cm.tab10
    # cmap = aCmap
    # Get the colormap colors
    my_cmap = a_cmap(np.arange(a_cmap.N))
    my_cmaps = []
    for clr in my_cmap:
        r, g, b, _ = clr
        cdict = {
            "red": [[0.0, r, r], [1.0, r, r]],
            "green": [[0.0, g, g], [1.0, g, g]],
            "blue": [[0.0, b, b], [1.0, b, b]],
            "alpha": [[0, 0.9, 0.9], [1, 0.1, 0.1]]
        }

        newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=100)
        my_cmaps.append(newcmp)
    return my_cmaps


def load_clipping_2dict(
    region_location_name,
    clipping_results_path,
    threshold: float = 1.0,
    gathered_results=None,
):
    """Load clipping results into a dictionary.
file_properties_df, region_location_name,
                                    clipping_results_path
    """
    if not gathered_results:
        gathered_results = {}
    clipping_threshold_str = str(threshold)
    clipping_threshold_str = clipping_threshold_str.replace(".", ",")
    file_name = (clipping_results_path /
                 (region_location_name + f"_{clipping_threshold_str}.pkl"))
    results_dict = np.load(file_name, allow_pickle=True)
    results_dict = results_dict[()]
    gathered_results.update(results_dict)
    return gathered_results


def vis_preds_with_clipping(region,
                            location_id,
                            file_prop_df_filtered,
                            input_data_freq,
                            output_data_freq,
                            model_tag_names,
                            my_cmaps,
                            result_path,
                            clipping_results_path,
                            vis_file_path,
                            id2name,
                            clipping_threshold: float = 1.0):

    ########     LOAD tag data     #########

    df_dict, no_result_paths = file2TableDict(
        model_tag_names,
        file_prop_df_filtered,
        input_data_freq=input_data_freq,
        output_data_freq=output_data_freq,
        result_path=result_path,
        prob2binary_flag=False,
    )
    df_count, df_sums = df_dict[location_id]
    df_freq = df_sums / df_count
    df_freq = df_freq * 100

    ########     LOAD Clipping     #########
    gathered_results = {}

    region_location_name = "_".join([region, location_id])
    gathered_results = load_clipping_2dict(
        region_location_name,
        clipping_results_path,
        threshold=clipping_threshold,
        gathered_results=gathered_results,)
    gathered_results_per_tag = {"Clipping": gathered_results}

    df_dict_clipping, no_result_paths = file2TableDict(
        ["Clipping"],
        file_prop_df_filtered,
        input_data_freq=input_data_freq,
        output_data_freq=output_data_freq,
        prob2binary_threshold=0.01,
        channel=2,
        gathered_results_per_tag=gathered_results_per_tag,
        result_path=None,
    )

    if len(no_result_paths) != 0:
        print("{} number of files do not have results".format(
            len(no_result_paths)))

    df_count_clipping, df_sums_clipping = df_dict_clipping[location_id]
    df_freq_clipping = df_sums_clipping / df_count_clipping
    df_freq_clipping = df_freq_clipping * 100

    ### add Clipping data to predictions

    df_freq = pd.concat([df_freq, df_freq_clipping], axis=1, sort=False)

    ### Divide data into months
    data_figure_parts_by_year = divide_data_into_months(df_freq,)
    ### create figure per year
    cord_list = [(i, (0, 0)) for i in df_freq.columns]
    for year, months_time_in_ayear, months_in_ayear in data_figure_parts_by_year:  #pylint: disable=line-too-long
        create_figure(location_id, months_in_ayear, months_time_in_ayear,
                      my_cmaps, cord_list, vis_file_path, region, year,
                      output_data_freq, id2name)

    return no_result_paths


def divide_data_into_months(df_freq,):

    months_time = pd.unique(df_freq.index.strftime("%Y-%m-01"))
    months_time = [pd.Timestamp(i) for i in months_time]

    months_time_str = [
        "{}-{}".format(month.year, month.month) for month in months_time
    ]
    months = [df_freq.loc[month:month] for month in months_time_str]
    ##### align all months
    for i, month in enumerate(months):
        months[i] = month.rename(index=lambda x: x.replace(month=7, year=2019))

    unique_years = np.unique([month.year for month in months_time])
    data_figure_parts_by_year = []
    for year in unique_years:
        months_in_ayear = [
            months[i]
            for i, month in enumerate(months_time)
            if month.year == year
        ]
        months_time_in_ayear = [
            months_time[i]
            for i, month in enumerate(months_time)
            if month.year == year
        ]
        data_figure_parts_by_year.append(
            (year, months_time_in_ayear, months_in_ayear))

    return data_figure_parts_by_year


def create_figure(selected_area, months, months_time, my_cmaps, cord_list,
                  vis_file_path, region_name, year, freq, id2name):
    #     plt.rcParams["axes.prop_cycle"] = get_cycle("tab10",N=8)
    vmin, vmax = 0, 100
    normalize = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(80, len(months) * 9),
                           nrows=len(months),
                           sharex=True,
                           sharey=True,
                           gridspec_kw={"hspace": 0})
    ax = np.array(ax).reshape(
        -1)  # subplot returns single element for single row

    # markers = itertools.cycle((',', '+', '.', 'o', '*'))

    # weather_colors = [
    #     "firebrick", "darkorange", "green", "seagreen", "lightpink"
    # ]

    for monthi, month in enumerate(months):
        # for col in df_freq.columns:
        for i, (col, (lat, long)) in enumerate(cord_list):
            del lat, long
            if col == "Clipping":
                continue
            #convert dates to numbers first
            inxval = mdates.date2num(month[col].index.to_pydatetime())
            points = np.array([inxval, month[col].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=my_cmaps[i],
                norm=normalize,
                linewidth=3,
            )
            # set color to date values
            lc.set_array(month["Clipping"])
            # note that you could also set the colors according to y values
            # lc.set_array(s.values)
            # add collection to axes
            ax[monthi].add_collection(lc)


#             break

# add legend and set names of the lines
    ax[0].legend(labels=[id2name.get(x[0], x[0][1:]) for x in cord_list],
                 loc="upper left",
                 borderpad=0.2,
                 labelspacing=0.2,
                 fontsize=28,
                 frameon=True)  # frameon=False to remove frame.

    # set colours of the lines on the legend
    leg = ax[0].get_legend()
    for i, (col, (lat, long)) in enumerate(cord_list):
        if col == "Clipping":
            continue
        leg.legendHandles[i].set_color(my_cmaps[i](vmin)[:-1])

    ax[-1].set_xlabel("Day Number", fontsize=32)

    #     uniqueYears=pd.unique([month.year for month in monthsTime])
    #     uniqueYears.size

    for i, an_ax in enumerate(ax):
        an_ax.set_ylabel("{}".format(months_time[i].strftime("%Y-%B")),
                         fontsize=48)  #, fontweight='black')

        locator = mdates.DayLocator()
        an_ax.xaxis.set_minor_locator(locator)
        an_ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d\n"))

        an_ax.xaxis.grid(True, which="minor")
        an_ax.xaxis.grid(True, which="major")

        an_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        an_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n"))

        an_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        an_ax.yaxis.grid()
        an_ax.tick_params(labelsize=22, which="minor")
        an_ax.tick_params(labelsize=25, which="major")

        # TODO figure out why we need to autoscale_view
        an_ax.autoscale_view()

    plt.tight_layout()
    plt.margins(x=0)
    plt.subplots_adjust(top=0.90)

    fig.suptitle(
        "Site {}, Normalized Bi-270min Frequency [%]".format(selected_area),
        fontsize=48)
    #     plt.show()

    fig_dir = Path(vis_file_path) / ("Freq-" + freq) / region_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / ("_".join([selected_area, str(year)]) + "." + "png")

    fig.savefig(fig_path)
    #     fig.show()
    #     fig.savefig("test" +'.png')
    plt.close(fig)
