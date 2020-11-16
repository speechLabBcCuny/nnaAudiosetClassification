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

# from nna.fileUtils import list_files
from nna import fileUtils
import itertools


# https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle
def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            use_index = cmap in [
                "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1",
                "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"
            ]

        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return plt.cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0.2, 1, N))
        return plt.cycler("color", colors)


def createTimeIndex(selected_areas, file_properties_df, freq):
    times = []
    # FIND earliest and latest time for time scale
    # lists in selected_areas_dict is ordered by time
    for _, area in enumerate(selected_areas):
        # get timestamp values from file_properties
        area_filtered = file_properties_df[file_properties_df.locationId ==
                                           area]
        # print(area)
        # print(area_filtered)
        if len(area_filtered.index) > 0:
            start = area_filtered.iloc[0]["timestamp"]
            end = area_filtered.iloc[-1]["timestamp"]

            times.extend([start, end])
        else:
            print("{}, do not have any files".format(area))

    times.sort()
    all_start = times[0].replace(hour=0, minute=0, second=0)
    all_end = times[-1].replace(hour=23, minute=59, second=59)

    # def days_hours_minutes(td):
    #     return td.days, td.seconds // 3600, (td.seconds // 60) % 60

    # create date axis indexes depending on start,end along with frequency
    if "H" in freq:
        number_hours = 3600
        count = int(freq[:-1])
        # extra = math.ceil(3 / count)
        periods = ((all_end - all_start).total_seconds() //
                   (number_hours * count) + 48)
        globalindex = pd.date_range(all_start, periods=periods, freq=freq)
    elif "D" in freq:
        periods = (all_end - all_start).days + 3
        globalindex = pd.date_range(all_start, periods=periods, freq=freq)
    elif "T" in freq or "min" in freq:
        #‘m’ / ‘minute’ / ‘min’ / ‘minutes’ / ‘T’
        time_digits = "".join([i for i in freq if i.isdigit()])
        buffer = timedelta(minutes=int(time_digits))
        all_end = all_end + buffer
        globalindex = pd.date_range(all_start, all_end, freq=freq)

    else:
        globalindex = pd.date_range(all_start, all_end, freq=freq)

    return globalindex, all_start, all_end


# result_path="/scratch/enis/data/nna/real/"
def loadResults(all_segments, prob2binaryFlag, threshold=0.5, channel=1):
    # try:
    if type(all_segments) != list:
        all_segments = [all_segments]

    results = []
    for filename in all_segments:
        filename = Path(filename)
        if not filename.exists():
            data = []
        data = np.load(filename)
        if prob2binaryFlag == True:
            data = prob2binary(data, threshold=threshold, channel=channel)
        results.append(data)

    results = np.concatenate(results)
    return results


def prob2binary(result, threshold=0.5, channel=1):
    if channel == 2:
        result = np.min(result, axis=1)
    result[result > threshold] = 1
    result[result <= threshold] = 0
    result = result[:(result.size // 10) * 10]
    result = result.reshape(10, -1).max(axis=0)
    return result


# with open("/home/enis/projects/nna/data/8tags_on_8sites_DF.pkl", 'ab') as  dffile:
#             # source, destination
#     pickle.dump(df_dict, dffile)
# TODO BUG HANDLE same locationId from multiple regions
def file2TableDict(selected_areas: List[str],
                   model_tag_names: List[str],
                   globalindex: pd.DataFrame,
                   globalcolumns: List[str],
                   file_properties_df: pd.DataFrame,
                   freq: str,
                   dataFreq: str = "10S",
                   dataThreshold: float = 0.5,
                   channel: int = 1,
                   gathered_results_perTag: Union[Dict, None] = None,
                   result_path: Union[str, None] = None,
                   file_name_addon: str = "",
                   prob2binaryFlag: bool = True) -> Tuple[Dict, List]:
    """Reduce results by dataFreq from multiple files into a pd.DataFrame.

        For all selected_areas (locationId), finds results for each tag from
        gathered_results_perTag: Dict or path (result_path) to pkl/npy files
        storing results. Reduces them by dataFreq and stores in pd.DataFrame
        
        Each result file named after it's original input file.
        Results are numpy arrays storing binary or decimal values.

        Args:
            selected_areas: Areas (locationId) to search results for.
            model_tag_names: Tags (_XXX,_CARS) to search results for.
            globalindex: Dataframe index of timestamps, from start to end with
                        dataFreq.
            globalcolumns: Name of the columns for the resulting DataFrame
                            actually same with model_tag_name.
            file_properties_df: Original input file's properties database to
                                be used a reference to find all files in the
                                given areas.
            freq: Frequency results are stored. (Not Used.)
            dataFreq="10S": Freq data will be reduced by.
            dataThreshold=0.5: if results are in probability, then threshold
                            used to calculate 0/1.
            channel=1: number of the channels of the data, # of dimensions
            gathered_results_perTag=None: Results stored in a
                                        dict["tag_name":
                                            {Path(a_file):np.Array,},]
            result_path=None: Location to look for results for each orig file
            file_name_addon="": Addon (_XXX) used with result files
            prob2binaryFlag=True: Should results converted to binary.

        Returns:
            Results dictionary and list of file_paths with no results.
            Dictionary is structured as:
                {"area_name1":(df_count:pd.DataFrame,df_sums:pd.DataFrame),
                 "area_name2":(df_count:pd.DataFrame,df_sums:pd.DataFrame),}
            df_count and df_sums have globalindex and globalcolumns as their
            indexes and column names.

    """
    del freq
    df_dict = {key: None for key in selected_areas}
    no_result_paths = []

    # we need to load it from files
    # print(gathered_results_perTag, result_path)
    if (gathered_results_perTag is None) and (result_path is None):
        print("ERROR: gathered_results_perTag or" +
              "(result_path and subDirectoryAddon )should be defined")
        return (None, None)

    for _, area in enumerate(selected_areas):
        df_sums = pd.DataFrame(index=globalindex,
                               columns=globalcolumns).fillna(0)
        df_count = pd.DataFrame(index=globalindex,
                                columns=globalcolumns).fillna(0)

        for model_tag_name in model_tag_names:
            #         for afile in selected_areas_dict[area]:
            area_filtered = file_properties_df[file_properties_df.site_id ==
                                               area]
            for afile, row in area_filtered.iterrows():
                #         data=gathered_results[afile][0]
                afile = Path(afile)
                # we either load data from multiple files or from single one
                if gathered_results_perTag is None:
                    # TODO, make _FCmodel variable
                    check_folder = fileUtils.standard_path_style(
                        result_path,
                        row,
                        sub_directory_addon=file_name_addon,
                        #BUG check if this is used correctly
                        # checked: it should not be used, otherwise,
                        # it does not return a folder
                        # file_name_addon=file_name_addon,
                    )
                    check_folder_str = str(check_folder) + "/"
                    all_segments = fileUtils.list_files(check_folder_str)
                    all_segments.sort()
                    if not all_segments:
                        data = np.empty(0)
                    else:
                        data = loadResults(all_segments,
                                           prob2binaryFlag=prob2binaryFlag,
                                           threshold=dataThreshold,
                                           channel=channel)
                        # gathered_results[file]=result[:]
                else:
                    data = gathered_results_perTag[model_tag_name].get(
                        afile, np.empty(0))[:]
                    if data.size != 0 and prob2binaryFlag:
                        data = prob2binary(
                            data,
                            threshold=dataThreshold,
                            channel=channel,
                        )

                if data.size == 0:
                    no_result_paths.append(afile)
                    continue

                start = file_properties_df.loc[afile]["timestamp"]
                end = start + timedelta(seconds=(10 * (len(data) - 1)))
                index = pd.date_range(start, end, freq=dataFreq)
                df_afile = pd.DataFrame(data,
                                        index=index,
                                        columns=[model_tag_name])
                # df_afile_grouped = df_afile.groupby([pd.Grouper(freq=freq)])
                # counts=df_afile_grouped.count()
                # sums=df_afile_grouped.sum()
                global_index_start = globalindex.searchsorted(
                    df_afile.index[0],)
                if global_index_start != 0:
                    global_index_start = global_index_start - 1
                global_index_end = globalindex.searchsorted(df_afile.index[-1])
                if global_index_end == global_index_start:
                    global_index_end = global_index_end + 1

                the_bins = pd.cut(
                    df_afile.index,
                    globalindex[global_index_start:global_index_end + 1])

                # the_bins=pd.cut(df_afile.index,globalindex)
                df_afileGrouped = df_afile.groupby(the_bins)
                sums = df_afileGrouped.agg("sum")
                counts = df_afileGrouped.agg("count")
                sums.set_index(sums.index.categories.left, inplace=True)
                counts.set_index(counts.index.categories.left, inplace=True)

                df_count = df_count.add(counts,
                                        fill_value=0)  # df_count.update(counts)
                df_sums = df_sums.add(sums,
                                      fill_value=0)  # df_sums.update(sums)

        df_dict[area] = (df_count.copy(), df_sums.copy())  # type: ignore

    return df_dict, no_result_paths


def reverseTableDict(selected_areas, df_dict, model_tag_names):
    # Reverse order of TAG and AREA in the dataframe
    # This graph for each tag for all areas

    df_dict_reverse = {}  # type: ignore
    for _, area in enumerate(selected_areas):
        df_count, df_sums = df_dict[area]
        for tagname in model_tag_names:
            df_dict_reverse.setdefault(tagname, [[], []])
            df_dict_reverse[tagname][0].append(df_count[tagname])
            df_dict_reverse[tagname][1].append(df_sums[tagname])

    for tagname in model_tag_names:

        df_count = pd.concat(df_dict_reverse[tagname][0], axis=1)
        df_sums = pd.concat(df_dict_reverse[tagname][1], axis=1)
        df_count.columns, df_sums.columns = selected_areas, selected_areas
        df_dict_reverse[tagname] = df_count, df_sums

    return df_dict_reverse


def rawFile2Csv(csvPath,
                selected_areas,
                model_tag_names,
                globalindex,
                globalcolumns,
                file_properties_df,
                freq,
                dataFreq="10S",
                dataThreshold=0.5,
                channel=1,
                gathered_results_perTag=None,
                result_path=None,
                file_name_addon="",
                prob2binaryFlag=True):

    del globalindex, globalcolumns, freq
    # using gathered_results_perTag dictionary or  result_path to create
    # a pandas dataframe for visualizations

    # dataFreq is sampling frequency of the data,
    # most of the time we have predictions for each 10 second
    if dataFreq != "10S":
        print(
            "ERROR: this function does not do aggregation, set dataFreq to 10S")
        return None

    csvFilesWritten = []
    no_result_paths = []

    # we need to load it from files
    if gathered_results_perTag == None and (result_path == None):
        print("ERROR: gathered_results_perTag or" +
              " (result_path and subDirectoryAddon )should be defined")
        return (None, None)

    for _, area in enumerate(selected_areas):
        #         df_sums = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)
        #         df_count = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)

        for model_tag_name in model_tag_names:
            df_raw_list = []
            #         for afile in selected_areas_dict[area]:
            area_filtered = file_properties_df[file_properties_df.site_id ==
                                               area]
            for afile, row in area_filtered.iterrows():
                ####################
                afile = Path(afile)

                # we either load data from multiple files or from single one
                if gathered_results_perTag == None:
                    check_folder = fileUtils.standard_path_style(
                        result_path,
                        row,
                        sub_directory_addon=model_tag_name,
                        file_name_addon=file_name_addon)
                    check_folder_str = str(check_folder) + "/"
                    all_segments = fileUtils.list_files(check_folder_str)
                    if not all_segments:
                        data = np.empty(0)
                    else:
                        data = loadResults(all_segments,
                                           prob2binaryFlag=prob2binaryFlag,
                                           threshold=dataThreshold,
                                           channel=channel)
                        # gathered_results[file]=result[:]
                else:
                    data = gathered_results_perTag[model_tag_name].get(
                        afile, np.empty(0))[:]
                    if data.size != 0 and prob2binaryFlag == True:
                        data = prob2binary(data, threshold=0.5, channel=channel)

                if data.size == 0:
                    no_result_paths.append(afile)
                    continue

                start = file_properties_df.loc[afile]["timestamp"]
                end = start + timedelta(seconds=(10 * (len(data) - 1)))
                index = pd.date_range(start, end, freq=dataFreq)
                df_afile = pd.DataFrame(data,
                                        index=index,
                                        columns=[model_tag_name])
                ####################
                df_raw_list.append(df_afile)
            if df_raw_list:
                df_raw = pd.concat(df_raw_list)
                df_raw = df_raw.sort_index()
                csv_file_name = "_".join([area, model_tag_name[1:] + ".csv"])
                df_raw.to_csv((csvPath / csv_file_name),
                              index_label="TimeStamp",
                              header=[model_tag_name[1:]])
                csvFilesWritten.append((csvPath / csv_file_name))

    return csvFilesWritten, no_result_paths


def add_normal_dist_alpha(aCmap):
    # Choose colormap
    # cmap = pl.cm.tab10
    cmap = aCmap
    # Get the colormap colors
    my_cmap = aCmap(np.arange(aCmap.N))
    my_cmaps = []
    for clr in my_cmap:
        r, g, b, _ = clr
        cdict = {
            'red': [[0.0, r, r], [1.0, r, r]],
            'green': [[0.0, g, g], [1.0, g, g]],
            'blue': [[0.0, b, b], [1.0, b, b]],
            'alpha': [[0, 0.9, 0.9], [1, 0.1, 0.1]]
        }

        newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=100)
        my_cmaps.append(newcmp)
    return my_cmaps


def load_clipping_2dict(clippingResultsPath, selected_areas, selected_tag_name):
    gathered_results_perTag = {selected_tag_name: {}}
    gathered_results = {}
    selected_areas_files = {}
    for i, area in enumerate(selected_areas):
        to_be_deleted = []
        fileName = (clippingResultsPath + area + "_1.pkl")
        resultsDict = np.load(fileName, allow_pickle=True)
        resultsDict = resultsDict[()]
        gathered_results_perTag[selected_tag_name].update(resultsDict)
    return gathered_results_perTag


# def loadClipping(clippingInfoFile):

#     clippingInfo=np.load(clippingInfoFile,allow_pickle=True)
#     clippingInfo = clippingInfo[()]
#     clippingInfo2={}
#     clippingInfoArray=[]
#     cc=0
#     for clipFile,clipping in clippingInfo.items():
#         locID=Path(clipFile).stem.split("_")[:2]
#         locID = tuple(locID)
#         clippingInfo2[locID] = clipping
#         if clipping.shape==(1,2):
#             clipping = clipping[0]
#         elif clipping.shape==(2,2):
#             clipping = clipping[0]
#         elif clipping.shape==(2,):
#             pass
#         else:
#             print("ERROR",clipping)
#         clippingInfoArray.append(clipping)
#         clippingInfo2[locID] = clipping

#     clippingInfoArray = np.concatenate(clippingInfoArray).reshape(-1,2)
#     return clippingInfo2,clippingInfoArray


def get_time_index_per_file(selected_area,
                            file_properties_df,
                            freq,
                            timeDifference=5):
    #TODO add sensitivity
    rowiterator = file_properties_df[file_properties_df.site_id ==
                                     selected_area].iterrows()
    # use first item as initialization, ahead of for loop
    n = next(rowiterator)
    start = n[1].timestamp
    beginning = start
    end = n[1].timestampEnd
    fileTimeIndex = []
    IsHead = False
    #     print(start)
    for row in rowiterator:
        # if end of previous file not equal to start of the second one
        if (row[1].timestamp - end) > timedelta(minutes=timeDifference):
            # add previous one to list and make new one the beginning of continous recording
            fileTimeIndex.append(beginning)
            beginning = row[1].timestamp
            IsHead = True
    #             pass
#             print("noteq",row[1].timestamp-end)
#         print(row[1].timestamp,end)
# if they are equal, they should be in the same bin, so keep going
        else:
            IsHead = False
    #             pass
    #         print("equal",row[1].timestampEnd-start)
    #         fileTimeIndex.append(start)
    #         print(row[1].timestampEnd,start)
        start = row[1].timestamp
        end = row[1].timestampEnd
    # If last one is a head add to the list


#     print(IsHead)
    if IsHead:
        fileTimeIndex.append(beginning)
    # add end since last bin border should be bigger than all data
    fileTimeIndex.append(end)
    fileTimeIndexSeries = pd.Series(fileTimeIndex)
    return fileTimeIndexSeries


def vis_preds_with_clipping(selected_area, file_properties_df, freq,
                            model_tag_names, my_cmaps, result_path, data_folder,
                            vis_file_path, id2name):
    selected_areas = [
        selected_area,
    ]
    # file length based time index
    if freq == "continous":
        fileTimeIndexSeries = get_time_index_per_file(selected_area,
                                                      file_properties_df, freq)
        globalindex = fileTimeIndexSeries
    #fixed freq based  time index
    else:
        globalindex, all_start, all_end = createTimeIndex(
            selected_areas, file_properties_df, freq)
        del all_start, all_end

    selected_tag_name = ["_" + i for i in model_tag_names]
    globalcolumns = selected_tag_name  #selected_areas+weather_cols

    #     print(globalindex)
    df_dict, no_result_paths = file2TableDict(selected_areas,
                                              selected_tag_name,
                                              globalindex,
                                              globalcolumns,
                                              file_properties_df,
                                              freq,
                                              dataFreq="10S",
                                              result_path=result_path,
                                              prob2binaryFlag=False)
    if len(no_result_paths) != 0:
        print("{} number of files do not have results".format(
            len(no_result_paths)))


#         print(no_result_paths[:1])
#         print(no_result_paths[-1:])

    regionName = file_properties_df[file_properties_df.site_id ==
                                    selected_area][:1].region[0]

    # we are not using this for visualizations
    # df_dict_reverse=reverseTableDict(selected_areas,df_dict,model_tag_names)
    df_count, df_sums = df_dict[selected_area]

    df_freq = df_sums / df_count
    # del df_freq['UMIAT']
    df_freq = df_freq * 100

    ########     LOAD Clipping     #########
    clippingResultsPath = data_folder + "clipping_results_old/"
    selected_tag_name = "Clipping"
    #     model_tag_names=[selected_tag_name]
    globalcolumns = [selected_tag_name]  #selected_areas+weather_cols

    gathered_results_perTag = load_clipping_2dict(clippingResultsPath,
                                                  selected_areas,
                                                  selected_tag_name)

    df_dict_clipping, no_result_paths = file2TableDict(
        selected_areas,
        globalcolumns,
        globalindex,
        globalcolumns,
        file_properties_df,
        freq,
        dataFreq="10S",
        dataThreshold=0.01,
        channel=2,
        gathered_results_perTag=gathered_results_perTag,
        result_path=None)

    df_count_clipping, df_sums_clipping = df_dict_clipping[selected_area]
    df_freq_clipping = df_sums_clipping / df_count_clipping
    df_freq_clipping = df_freq_clipping * 100

    ### add Clipping data to predictions

    df_freq = pd.concat([df_freq, df_freq_clipping], axis=1, sort=False)
    if len(no_result_paths) != 0:
        print("{} number of files do not have results".format(
            len(no_result_paths)))

    ### Divide data into months

    cord_list = [(i, (0, 0)) for i in df_freq.columns]

    monthsTime = pd.unique(df_freq.index.strftime("%Y-%m-01"))
    monthsTime = [pd.Timestamp(i) for i in monthsTime]

    monthsTimeStr = [
        "{}-{}".format(month.year, month.month) for month in monthsTime
    ]
    months = [df_freq.loc[month:month] for month in monthsTimeStr]
    ##### align all months
    for i, month in enumerate(months):
        months[i] = month.rename(index=lambda x: x.replace(month=7, year=2019))

    uniqueYears = np.unique([month.year for month in monthsTime])
    for year in uniqueYears:
        monthsInAYear = [
            months[i]
            for i, month in enumerate(monthsTime)
            if month.year == year
        ]
        monthsTimeInAYear = [
            monthsTime[i]
            for i, month in enumerate(monthsTime)
            if month.year == year
        ]
        create_figure(selected_area, monthsInAYear, monthsTimeInAYear, my_cmaps,
                      cord_list, vis_file_path, regionName, year, freq, id2name)


def create_figure(selected_area, months, monthsTime, my_cmaps, cord_list,
                  vis_file_path, regionName, year, freq, id2name):
    #     plt.rcParams["axes.prop_cycle"] = get_cycle("tab10",N=8)
    vmin, vmax = 0, 100
    normalize = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(80, len(months) * 9),
                           nrows=len(months),
                           sharex=True,
                           sharey=True,
                           gridspec_kw={'hspace': 0})
    ax = np.array(ax).reshape(
        -1)  # subplot returns single element for single row

    markers = itertools.cycle((',', '+', '.', 'o', '*'))

    weather_colors = [
        "firebrick", "darkorange", "green", "seagreen", "lightpink"
    ]

    for monthi, month in enumerate(months):
        # for col in df_freq.columns:
        for i, (col, (lat, long)) in enumerate(cord_list):
            #             if col in weather_cols:
            #                 index=weather_cols.index(col)
            #                 ax[monthi].plot_date(month.index.to_pydatetime(), month[col],linestyle="-",marker=" ",color=weather_colors[index])
            #             else:
            #             ax[monthi].plot_date(month.index.to_pydatetime(), month[col],linestyle="-",marker=" ")
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
                 loc='upper left',
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

    ax[-1].set_xlabel('Day Number', fontsize=32)

    #     uniqueYears=pd.unique([month.year for month in monthsTime])
    #     uniqueYears.size

    for i, an_ax in enumerate(ax):
        an_ax.set_ylabel('{}'.format(monthsTime[i].strftime("%Y-%B")),
                         fontsize=48)  #, fontweight='black')

        locator = mdates.DayLocator()
        an_ax.xaxis.set_minor_locator(locator)
        an_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n'))

        an_ax.xaxis.grid(True, which="minor")
        an_ax.xaxis.grid(True, which="major")

        an_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        an_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n'))

        an_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        an_ax.yaxis.grid()
        an_ax.tick_params(labelsize=22, which="minor")
        an_ax.tick_params(labelsize=25, which="major")

        # TODO figure out why we need to autoscale_view
        an_ax.autoscale_view()

    plt.tight_layout()
    plt.margins(x=0)
    plt.subplots_adjust(top=0.90)

    fig.suptitle(
        'Site {}, Normalized Bi-270min Frequency [%]'.format(selected_area),
        fontsize=48)
    #     plt.show()

    figDir = Path(vis_file_path) / ("Freq-" + freq) / regionName
    figDir.mkdir(parents=True, exist_ok=True)
    figPath = figDir / ("_".join([selected_area, str(year)]) + '.' + "png")

    fig.savefig(figPath)
    #     fig.show()
    #     fig.savefig("test" +'.png')
    plt.close(fig)
