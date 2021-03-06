"""Operations around files by using unix style path."""

import csv
import datetime
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import IPython
import pandas as pd
from IPython.display import display
from pydub import AudioSegment

# from PIL import Image


def save_to_csv(file_name: Union[str, Path], lines: List[Sequence]) -> None:
    """Append list of lines to a file in csv format.
    Args:
        file_name: Name of the csv file to append.
        lines: List of iterables; one iterable per line.
               do NOT seperate items by comma yourself for csv.
    """
    file_name = Path(file_name).with_suffix(".csv")
    with open(str(file_name), mode="a") as labels_file:
        label_writer = csv.writer(labels_file,
                                  delimiter=",",
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for line in lines:
            label_writer.writerow(line)


def standard_path_style(
        parent_path: Union[str, Path],
        row: Union[pd.Series, Dict],
        sub_directory_addon: Optional[str] = "",
        file_name_addon: Optional[str] = "") -> Union[Path, None]:
    """Generate a path for a file given sub directory and file.

    NNA files are organized in folder structure of region/locationId/year,
    and they have a file name add-on following an underscore except original
    files.
    Add-on is a identifier for process applied to original file.
    Such as 'vgg' which means vgg embeddings of the original file.
    Path might have an extra folder in case there are multiple outputs
    for a original file.
    This folder named as original_file_name + sub_Directory_Addon.
    Usually sub_Directory_Addon is same with file_name_addon.
    
    !!sub_directory_addon and file_name_addon are merged with "_"

    Args:
        parent_Path: parent path for the generated path.
        row: Pandas Series object or dict with properties of
                    region,locationId,year. If Series then it should have a name
                    and Dict should have a key called "name"
        sub_Directory_Addon: (Optional) id for the output folder ex-> "XXX"
        file_name_addon:(Optional)  id for the output file ex-> "XXX"
    Returns:
        generated_path: a Path object following NNA folder ordering, has extra
            folder if sub_Directory_Addon is given, if file_name_addon given
            returns a file, otherwise just a folder.
    """
    row_region = row.get("region", "")
    row_location_id = row.get("locationId", "")
    row_year = row.get("year", "")
    if isinstance(row, pd.Series):
        row_name = row.name
        if row_name is None:
            row_name = ""
    elif isinstance(row, dict):
        row_name = row.get("name", "")
    else:
        raise TypeError("row should be dict or pd.Series")
    file_name = Path(row_name)

    if (not row_region) or (not row_location_id) or (not row_year):
        print("Empty info for the path from row: {}".format(row))
        return None
    generated_path = Path(parent_path) / str(row_region) / str(
        row_location_id) / str(row_year)

    if sub_directory_addon:
        sub_directory_name = "_".join([file_name.stem, sub_directory_addon])
        generated_path = generated_path / sub_directory_name
    if file_name_addon:
        file_name = "_".join([file_name.stem, file_name_addon])
        generated_path = generated_path / file_name
    return generated_path


def parse_file_path(file_path: Union[str, Path], debug: int = 0) -> Dict:
    """Parse info of timestamp,region,locationId,year from file path.

    expected in the string: region/locationId/YYYY/name_YYYYMMDD_HHMMSS
    possibilities:
            XYZ/random/region/locationId/YYYY/name_YYYYMMDD_HHMMSS_XXX000.stuf

    Args:
        file_path: A file path.
        debug: Debugging level.

    Returns:
        A dict mapping timestamp,region,locationId,year to the corresponding
        information from file path.
        example:

        {"timestamp": "20190531_060000",
         "region": "Anwr",
         "locationId": "12",
         "year": "2019",
         "part_index":0 }

    """
    del debug
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # relative2main = file_path.relative_to(source_folder)
    file_name = file_path.stem

    file_path_parts = file_path.parts[-4:]
    file_path_parts = file_path_parts[1:] if file_path_parts[
        0] == "/" else file_path_parts

    year_folder = ""
    location_id = ""
    region = ""
    if len(file_path_parts) >= 2:
        year_folder = file_path_parts[-2]
        if len(file_path_parts) >= 3:
            location_id = file_path_parts[-3]
            if len(file_path_parts) >= 4:
                region = file_path_parts[-4]

    file_name_parts = file_name.split("_")

    output_index: Union[int, None] = None
    #OUTPUT_HAS_PARTS = True # such as _XXX001.py
    if len(file_name_parts) > 3:
        output_index = int("".join(
            c for c in file_name_parts[-1] if c.isdigit()))
        # here [1:3] does not work for 0813_091810_embeddings025.npy
        # [-3:-1] works for both S4A10327_20190531_060000_embeddings000.npy
        timestamp = "_".join(file_name_parts[-3:-1])
    else:
        timestamp = "_".join(file_name_parts[-2:])

    yearfile_name = timestamp.split("_")[0][0:4]

    # Special Case
    if yearfile_name != year_folder and (region != "stinchcomb" and
                                         location_id != "20-Umiat"):
        print("ERROR File is in the wrong year folder ", file_path)
    if region == "stinchcomb" and location_id == "20-Umiat":
        year = yearfile_name
    else:
        year = year_folder

    return {
        "timestamp": timestamp,
        "region": region,
        "locationId": location_id,
        "year": year,
        "part_index": output_index
    }


def match_path_info2row(path_info: Dict,
                        file_properties_df: pd.DataFrame,
                        debug: int = 0) -> Union[Path, int]:
    """From timestamp,region,locationId,year, find row from the database.

    Args:
        path_info: A dict mapping timestamp,region,locationId,year to the
                    corresponding information from file path.
            example:
                {"timestamp": "20190531_060000",
                 "region": "Anwr",
                 "locationId": "12",
                 "year": "2019",
                 "part_index":0 }

        file_properties_df: A dataframe with info about original files.
        debug: Optional; Debugging level.

    Returns:
        A pandas Series from file_properties_df.
            Example:
                site_id                          12
                locationId                       12
                site_name
                recorderId                 S4A10274
                hour_min_sec                 103000
                year                           2019
                month                            08
                day                              11
                region                      prudhoe
                timestamp       2019-08-11 10:30:00
                durationSec                    4560
                timestampEnd    2019-08-11 11:46:00
                Name: /tank/data/nna/real/prudhoe/12/\
                    2019/S4A10274_20190811_103000.flac, dtype: object
    """

    is_region = file_properties_df.region == path_info.get("region", "")
    is_location_id = file_properties_df.locationId == str(
        path_info.get("locationId", ""))
    is_year = file_properties_df.year == path_info.get("year", "")

    truth_table = is_region & is_location_id & is_year
    filtered_properties = file_properties_df[truth_table]

    for row in filtered_properties.iterrows():
        if path_info.get("timestamp", "") in str(row[0]):
            return row
    if debug > 0:
        print(path_info.get("timestamp", ""), filtered_properties)

    return -1


# TODO: deprecate it
def npy2originalfile(file_path, source_folder, file_properties_df, debug=0):

    #     file_path.parents[parentDistance]
    # relative2main = file_path.relative_to(source_folder)
    # file_name = relative2main.parents[0].stem
    #
    # # find possible files in the file properties
    # region = relative2main.parts[0]
    # location_id = relative2main.parts[1]
    # # here [1:3] does not work for 0813_091810_embeddings025.npy
    # # [-3:-1] works for both S4A10327_20190531_060000_embeddings000.npy
    # timestamp = "_".join(file_name.split("_")[-3:-1])
    # yearfile_name = timestamp.split("_")[0][0:4]
    # year_folder = relative2main.parts[2]
    # if yearfile_name != year_folder and (region != "stinchcomb"
    #                                      and location_id != "20-Umiat"):
    #     print("ERROR File is in the wrong year folder ", file_path)
    # if region == "stinchcomb" and location_id == "20-Umiat":
    #     year = yearfile_name
    # else:
    #     year = year_folder
    # isRegion = file_properties_df.region == region
    # islocationID = file_properties_df.locationId == location_id
    # isYear = file_properties_df.year == year
    #
    # truthTable = isRegion & islocationID & isYear
    # filteredProperties = file_properties_df[truthTable]
    #
    # for row in filteredProperties.iterrows():
    #     if timestamp in str(row[0]):
    #         return row[0]
    del source_folder
    path_info = parse_file_path(file_path, debug=debug)
    row = match_path_info2row(path_info, file_properties_df, debug=debug)

    return row


# TODO ,work with relative paths not absolute
def read_file_properties(mp3_files_path_list):
    if not isinstance(mp3_files_path_list, list):
        with open(str(mp3_files_path_list)) as f:
            lines = f.readlines()
            mp3_files_path_list = [line.strip() for line in lines]

    site_names = []
    hours = []
    exceptions = []
    file_properties = {}
    for apath in mp3_files_path_list:
        apath = Path(apath)
        name = apath.stem
        if len(apath.parents) == 6:
            site_name = apath.parent.stem
        else:
            site_name = " ".join(apath.parent.parent.stem.split(" ")[1:])
    #     print(site_name)
    # file_id = name
        name = name.split("_")
        # ones without date folder
        if len(apath.parents) == 7 and len(name) == 3:
            site_name_tmp = apath.parent.stem.split(" ")
            if len(site_name_tmp) == 1:
                site_name = site_name_tmp[0]
            else:
                #                 print("2")
                site_name = " ".join(site_name_tmp[1:])
    #         print(site_name)
    #         print(apath)
            site_id = name[-3]
            site_names.append(site_name)
            date = name[-2]
            hour_min_sec = name[-1]
            hour = hour_min_sec[0:2]
            hours.append(hour)
            year, month, day = date[0:4], date[4:6], date[6:8]
        # usual ones
        elif len(name) == 3:
            site_id = name[-3]
            site_names.append(site_name)
            date = name[-2]
            hour_min_sec = name[-1]
            hour = hour_min_sec[0:2]
            hours.append(hour)
            year, month, day = date[0:4], date[4:6], date[6:8]
        # stem does not have site_id in it
        elif len(name) == 2:
            site_id = "USGS"
            site_names.append(site_name)
            date = name[-2]
            hour_min_sec = name[-1]
            hour = hour_min_sec[0:2]
            hours.append(hour)
            month, day = date[0:2], date[2:4]
    #         year=Path(apath).parent.stem.split(" ")[0]

    # files with names that does not have fixed rule
        else:
            exceptions.append(apath)
        file_properties[apath] = {
            "site_id": site_id,
            "site_name": site_name,
            "hour_min_sec": hour_min_sec,
            "year": year,
            "month": month,
            "day": day
        }
        str2timestamp(file_properties[apath])
    return file_properties, exceptions


# TODO ,work with relative paths not absolute IMPORTANT fix that
# then update
def read_file_properties_v2(mp3_files_path_list, debug=0):
    if isinstance(mp3_files_path_list, str):
        if debug > 0:
            print("using txt file at {}".format(mp3_files_path_list))
        with open(str(mp3_files_path_list)) as f:
            lines = f.readlines()
            mp3_files_path_list = [line.strip() for line in lines]

    def inner_loop(apath, exceptions):
        if debug > 0:
            print(apath)
        apath = Path(apath)
        # usual ones
        if len(apath.parents) == 8:
            recorderId_startDateTime = apath.stem

            recorderId_startDateTime = recorderId_startDateTime.split("_")
            recorderId = recorderId_startDateTime[0]
            if debug > 0:
                print(recorderId_startDateTime)
            date = recorderId_startDateTime[1]
            if debug > 0:
                print(date)
            year, month, day = date[0:4], date[4:6], date[6:8]

            hour_min_sec = recorderId_startDateTime[2]
            if hour_min_sec is None:
                print(apath)
            # hour = hour_min_sec[0:2]
            location_id = apath.parts[6]
            region = apath.parts[5]

            site_name = ""

            file_properties[apath] = str2timestamp({
                "site_id": location_id,
                "locationId": location_id,
                "site_name": site_name,
                "recorderId": recorderId,
                "hour_min_sec": hour_min_sec,
                "year": year,
                "month": month,
                "day": day,
                "region": region
            })

        else:
            exceptions.append(apath)

    exceptions: List[str] = []
    file_properties = {}
    for apath in mp3_files_path_list:
        try:
            inner_loop(apath, exceptions)
        except:
            exceptions.append(apath)

    return file_properties, exceptions


# example usage in ../notebooks/Labeling/save_file_properties.ipynb
def getLength(
    input_video,
    ffprobe_path="/scratch/enis/conda/envs/speechEnv/bin/ffprobe",
):
    input_video = str(input_video)

    cmd = []
    cmd.extend([
        ffprobe_path, "-i", "{}".format(input_video), "-show_entries",
        "format=duration", "-v", "quiet"
    ])
    result = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_b = result.communicate(b"\n")
    output = [i.decode("ascii") for i in output_b]
    length: Union[float] = -1.0
    if output[0] == "" or output[0] == "N/A":
        print("ERROR file is too short {}".format(input_video))
        print("command run with ERROR: {}".format(cmd))
    else:
        output_stdout = output[0]
        length = float(output_stdout.split("\n")[1].split("=")[1])

    return length


def list_files(search_path: str = "/search_path/",
               ignore_folders: Union[List, None] = None,
               file_name: str = "*.*"):
    if ignore_folders is None:
        ignore_folders = []
    if search_path[-1] != "/":
        search_path += "/"
    all_path_set = set(
        glob.glob(search_path + "**/" + file_name, recursive=True))

    for folder in ignore_folders:
        ignore_paths = set(glob.glob(folder + "**/*.*", recursive=True))
        all_path_set = all_path_set.difference(ignore_paths)
    all_path = sorted(list(all_path_set))
    return all_path


def str2timestamp(fileinfo_dict):
    # x=file_properties[file]
    #         print(x)
    hour_min_sec = fileinfo_dict["hour_min_sec"]
    hour = int(hour_min_sec[:2])
    minute = int(hour_min_sec[2:4])
    second = int(hour_min_sec[4:6])
    year = int(fileinfo_dict["year"])

    timestamp = datetime.datetime(year,
                                  int(fileinfo_dict["month"]),
                                  int(fileinfo_dict["day"]),
                                  hour=hour,
                                  minute=minute,
                                  second=second,
                                  microsecond=0)
    fileinfo_dict["timestamp"] = timestamp
    return fileinfo_dict


# start_time='01-07-2016_18:33:00' # or datetime object
def find_files(location, start_time, end_time, length, file_properties_df):

    file_properties_df = file_properties_df.sort_values(by=["timestamp"])
    if location in file_properties_df["site_id"].values:
        loc_key = "site_id"
    elif location in file_properties_df["site_name"].values:
        loc_key = "site_name"
    else:
        print("Location not found")
        print("Possible names and ids:")
        for site_name, site_id in set(
                zip(file_properties_df.site_name, file_properties_df.site_id)):
            print(site_name, "---", site_id)

    if isinstance(start_time, str):
        start_time = datetime.datetime.strptime(start_time, "%d-%m-%Y_%H:%M:%S")

    site_filtered = file_properties_df[file_properties_df[loc_key] == location]
    # print(site_filtered)
    if length != 0:
        end_time = start_time + datetime.timedelta(seconds=length)
    else:
        end_time = datetime.datetime.strptime(end_time, "%d-%m-%Y_%H:%M:%S")

    if not start_time or not end_time:
        print("time values should be given")

    # first and last recordings from selected site
    first, last = site_filtered["timestamp"][0], site_filtered["timestamp"][-1]
    # make sure start or end time time are withing possible range
    beginning, end = max(start_time, first), min(end_time, last)

    start_file = site_filtered[
        site_filtered["timestamp"] <= beginning].iloc[-1:]

    time_site_filtered = site_filtered[site_filtered["timestamp"] > beginning]

    time_site_filtered = time_site_filtered[
        time_site_filtered["timestamp"] < end]

    time_site_filtered = pd.concat([time_site_filtered, start_file])

    sorted_filtered = time_site_filtered.sort_values(by=["timestamp"])
    # print(time_site_filtered)
    if len(sorted_filtered.index) == 0:
        print("No records for these times at {} ".format(location))
        print("Earliest {}  and latest {}".format(first, last))

    return sorted_filtered, start_time, end_time


def find_filesfunc_inputs(location, region, start_time, end_time, length,
                          buffer, file_properties_df):
    del region

    if location in file_properties_df["site_id"].values:
        loc_key = "site_id"
    elif location in file_properties_df["site_name"].values:
        loc_key = "site_name"
    else:
        print("Location not found")
        print("Possible names and ids:")
        for site_name, site_id in set(
                zip(file_properties_df.site_name, file_properties_df.site_id)):
            print(site_name, "---", site_id)
        return None, None, None, None, None,

    if not start_time:
        print("start time value should be given")
    if not (end_time) and length <= 0:
        print(
            "end time value should be given or lenght should be bigger than 0")

    if isinstance(start_time, str):
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d_%H:%M:%S")

    # if length is given then overwrite end time
    if length > 0:
        end_time = start_time + datetime.timedelta(seconds=length)
    elif isinstance(end_time, str):
        end_time = datetime.datetime.strptime(start_time, "%Y-%m-%d_%H:%M:%S")

    # buffer changes start and end times, we keep original values
    if buffer > 0:
        start_time_org, end_time_org = start_time, end_time
        start_time_buffered = start_time - datetime.timedelta(seconds=buffer)
        end_time_buffered = start_time + datetime.timedelta(seconds=buffer +
                                                            length)
        start_time, end_time = start_time_buffered, end_time_buffered
    else:
        start_time_org, end_time_org = start_time, end_time

    # print("start:",start_time,"end",end_time)
    try:
        return start_time, end_time, loc_key, start_time_org, end_time_org
    except:
        return None


# from nna.fileUtils import find_filesv2
def find_filesv2(location, region, start_time, end_time, length, buffer,
                 file_properties_df,only_continuous=True):
    #     print(start_time,end_time)
    output = find_filesfunc_inputs(location, region, start_time, end_time,
                                   length, buffer, file_properties_df)
    start_time, end_time, loc_key, start_time_org, end_time_org = output

    # sorted here
    file_properties_df = file_properties_df.sort_values(by=["timestamp"])

    site_filtered = file_properties_df[file_properties_df['region'] == region]
    site_filtered = site_filtered[site_filtered[loc_key] == location]

    # first and last recordings from selected site
    first = site_filtered["timestamp"][0]
    last = site_filtered["timestampEnd"][-1]
    #     print("first,last",first,last)
    #######
    # if query before all recordings, return empty
    if first > end_time:
        return (site_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)
#    if query all after recordings, return empty
    elif last < start_time:
        return (site_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)

# if   TODO thing about this, if we wanna keep overlapping parts !
    elif start_time < first:
        print('start<first')
        return (site_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)
    elif end_time > last:
        print('end>last')
        return (site_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)

    # make sure start or end time are within possible range
    beginning, end = max(start_time, first), min(end_time, last)
    start_file = site_filtered[
        site_filtered["timestamp"] <= beginning].iloc[-1:]
    time_site_filtered = site_filtered[site_filtered["timestamp"] > beginning]

    time_site_filtered = time_site_filtered[
        time_site_filtered["timestamp"] < end]
    time_site_filtered = pd.concat([time_site_filtered, start_file])

    # remove ones that has an end before our start
    time_site_filtered = time_site_filtered[
        time_site_filtered["timestampEnd"] >= beginning]

    sorted_filtered = time_site_filtered.sort_values(by=["timestamp"])
    if len(sorted_filtered.index) == 0:
        # print('56')
        return (sorted_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)
    # if   TODO thing about this, if we wanna keep overlapping parts !
    if len(sorted_filtered.index) > 1 and only_continuous:

        timestamps, timestamp_ends = site_filtered["timestamp"], site_filtered[
            "timestampEnd"]
        i = 0
        while i < len(sorted_filtered.index):
            # if recordings are not continues
            if timestamps[i + 1] != timestamp_ends[i]:
                print('123')
                return site_filtered[
                    0:0], start_time, end_time, start_time_org, end_time_org
            i += 1

    first_result_start, last_result_end = sorted_filtered["timestamp"][0], sorted_filtered[
        "timestampEnd"][-1]
    # if query starts after first result does and ends before last result does
    if first_result_start <= start_time and last_result_end >= end_time:
        print('...')
        return (sorted_filtered, start_time, end_time, start_time_org,
                end_time_org)
    else:
        # print('elseeee')
        return (sorted_filtered[0:0], start_time, end_time, start_time_org,
                end_time_org)


def get_audio(sorted_filtered,
              start_t,
              end_t,
              display_flag=True,
              save=True,
              file_name="output",
              tmpfolder="./trap_photo_audio/"):
    del save
    total_seconds = 0
    for _, f in enumerate(sorted_filtered.iterrows()):

        # in audio file, where should excerpt starts and ends
        start_seconds = max(((start_t - f[-1].timestamp)).total_seconds(), 0)
        end_seconds = max((end_t - f[-1].timestamp).total_seconds(), 0)

        excerpt_length = end_seconds - start_seconds

        # these are minutes and seconds are for naming
        # which seconds of the query audio in this file
        start_minute, start_second = divmod(int(total_seconds), 60)
        start_minute, start_second = str(start_minute), str(start_second)
        end_minute, end_second = divmod(int(total_seconds + excerpt_length), 60)
        end_minute, end_second = str(end_minute), str(end_second)

        file_name_specific = file_name + "_" + start_minute + "m_" + \
            start_second + "s__" + end_minute + "m_" + end_second + "s"

        mp3_file_path = f[0]
        file_extension = str(Path(mp3_file_path).suffix)

        save_audiofile(mp3_file_path, file_extension, file_name_specific,
                       start_seconds, end_seconds, tmpfolder)

        if display_flag:
            display_audio(tmpfolder, file_name_specific, file_extension)
        # if not save:
        # delete the file
        total_seconds += excerpt_length


def save_audiofile(mp3_file_path, file_extension, file_name, start_seconds,
                   end_seconds, tmpfolder):
    from nna.labeling_utils import ffmpeg_split_mp3

    # if end_seconds bigger than file, ffmpeg ignores it, if both out
    # of order than output is emtpy
    ffmpeg_split_mp3(mp3_file_path,
                     start_seconds,
                     end_seconds,
                     tmpfolder=tmpfolder)

    try:
        os.rename(tmpfolder + "output" + file_extension,
                  tmpfolder + file_name + file_extension)

    except:
        print("{}".format(sys.exc_info()[0]))


def display_audio(tmpfolder, file_name, file_extension):
    try:
        if file_extension in [".mp3", ".MP3"]:
            mp3file = AudioSegment.from_mp3(
                Path(tmpfolder + file_name + file_extension))
        else:
            mp3file = AudioSegment.from_file(
                Path(tmpfolder + file_name + file_extension))
        display(mp3file)
    except:
        print("{}".format(sys.exc_info()[0]))


def query_audio(location,
                region,
                start_time,
                end_time,
                length,
                buffer,
                file_properties_df,
                file_name,
                display_flag=True,
                save=True,
                tmp_folder="./tmp_audio_excerpt/"):
    '''Find audio segment,trim segment and name according to found time.

        Audio can be found in given 'exact' time. 
            If buffer is bigger than 0 then it can be found 'earlier' or 'later'
            on of these words is placed in tmp audio file name accordingly.
            ex: f'{filename}_exact_{start_time.strftime(%Y-%m-%d_%H:%M:%S)}'

        Args:

        Returns: File df with entries corresponding to the query.

    '''
    output = find_filesv2(location, region, start_time, end_time, length, 0,
                          file_properties_df,only_continuous=False)
    sorted_filtered, start_time, end_time, start_time_org, end_time_org = output

    # if there is no file without buffer then search again with buffer
    if len(sorted_filtered.index) == 0 and buffer > 0:

        output = find_filesv2(location, region, start_time_org, end_time_org,
                              length, buffer, file_properties_df,only_continuous=False)
        (sorted_filtered, start_time, end_time, start_time_org,
         end_time_org) = output

        closestLeft = sorted_filtered[
            sorted_filtered["timestampEnd"] < start_time_org][-1:]
        closestRight = sorted_filtered[
            sorted_filtered["timestamp"] > end_time_org][:1]
        if len(sorted_filtered.index) == 0:
            return sorted_filtered
            print("Recording not found")
        elif len(closestLeft.index) == 0:
            start_time = closestRight["timestamp"][0]
            end_time = closestRight["timestamp"][0] + datetime.timedelta(
                seconds=length)
            file_name += "_earlier_" + start_time.strftime("%Y-%m-%d_%H:%M:%S")

            get_audio(closestRight,
                      start_time,
                      end_time,
                      display_flag=display_flag,
                      save=save,
                      file_name=file_name,
                      tmpfolder=tmp_folder)
        elif len(closestRight.index) == 0:
            start_time = closestLeft["timestampEnd"][0] - datetime.timedelta(
                seconds=length)
            end_time = closestLeft["timestampEnd"][0]
            file_name += "_later_" + start_time.strftime("%Y-%m-%d_%H:%M:%S")
            get_audio(closestLeft,
                      start_time,
                      end_time,
                      display_flag=display_flag,
                      save=save,
                      file_name=file_name,
                      tmpfolder=tmp_folder)

    else:
        #     print(sorted_filtered)
        file_name += "_exact_" + start_time_org.strftime("%Y-%m-%d_%H:%M:%S")
        get_audio(sorted_filtered,
                  start_time_org,
                  end_time_org,
                  display_flag=display_flag,
                  save=save,
                  file_name=file_name,
                  tmpfolder=tmp_folder)
    # print(file_name)
    return sorted_filtered


def findPhoto(location,
              timestamp,
              imgOnlyDate,
              buffer=datetime.timedelta(seconds=1)):
    """Find photo in a location with given timestamp
    
    Does not support region filtering.

  Example
  --------
  import datetime

  from IPython.display import display, Image
  import pickle

  # information about images
  with open("../../data/imgOnlyDateV1.pkl", "rb") as f:
      imgOnlyDate=pickle.load(f)


  # query
  location="35"
  start_time='2019-06-05_00:00:00' # YYYY-MM-DD_HH:MM:SS or datetime object
  # if there is no recording in given timestamp, it searches before and after,
  # buffer is how far to look in seconds
  buffer=1800


  timestamp=datetime.datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
  end_time=None

  imgTime,imgPath=findPhoto(location,timestamp,imgOnlyDate,buffer=buffer)


  if imgTime!=-1:
      display(Image(imgPath,width=600))
  dataPoint="'{},{},{},{}',".format(location,start_time,imgPath)
  print(dataPoint)
  """
    from bisect import bisect_left

    index = bisect_left(imgOnlyDate[location], (timestamp, ""))
    if index == len(imgOnlyDate[location]):
        index -= 1

    if index == -1:
        index = 1

    left = imgOnlyDate[location][index - 1]
    right = imgOnlyDate[location][index]
    #     print(index)
    #     print(left[0],right[0])

    if timestamp == left[0]:
        return left
    if timestamp == right[0]:
        return right
    leftDistance = abs(timestamp - left[0])
    rightDistance = abs(right[0] - timestamp)
    #     print(leftDistance,rightDistance)
    if leftDistance <= rightDistance and leftDistance <= buffer:
        print(leftDistance)
        return left
    if rightDistance <= buffer:
        print(rightDistance)
        return right
    return (-1, -1)


def QuickQuery_audio(location,
                     region,
                     start_time,
                     imgOnlyDate,
                     file_properties_df,
                     length=30):

    end_time = None  # or datetime object

    # how to name the file
    file_name = "original2"

    # if there is no recording in given timestamp, it searches before and after,
    # buffer is how far to look in seconds
    buffer_audio = 1800

    # where to save audio files
    tmp_folder = "/home/enis/projects/nna/data/tmp/"

    output = query_audio(location,
                         region,
                         start_time,
                         end_time,
                         length,
                         buffer_audio,
                         file_properties_df,
                         file_name,
                         display_flag=True,
                         save=True,
                         tmp_folder=tmp_folder)
    if len(output.index) == 0:
        print("No Audio")
    else:
        print(output.iloc[0].timestamp, "---", output.iloc[0].durationSec)

    if isinstance(start_time, str):
        timestamp = datetime.datetime.strptime(start_time, "%Y-%m-%d_%H:%M:%S")
    else:
        timestamp = start_time
    buffer = datetime.timedelta(minutes=5)

    imgTime, imgPath = findPhoto(location,
                                 timestamp,
                                 imgOnlyDate,
                                 buffer=buffer)

    if imgTime != -1:
        display(IPython.display.Image(imgPath, width=600))
    data_point = "'{},{},{},{}',".format(location, start_time, length, imgPath)
    print(data_point)
