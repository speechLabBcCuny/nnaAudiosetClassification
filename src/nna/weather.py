''' This script is used to generate weather data for the NNA dataset.
'''
import os

# import itertools

import glob

from pathlib import Path
import csv
import random
import datetime
import pandas as pd
import numpy as np

TIMESTAMP_FORMAT = '%Y-%m-%d_%H:%M:%S'
WEATER_DATA_FREQ = 3600 * 3  # 3 hours

LOCAL = False

if LOCAL:
    path = '/Users/berk/Documents/research/nna/src/scripts/'
else:
    path = '/home/enis/projects/nna/src/scripts/'
os.chdir(path)


def load_rows(csv_fname, region, short_locations, long_locations):
    with open(csv_fname, newline='', encoding='utf-8') as csvfile:
        # reader = csv.DictReader(csvfile,fieldnames=fieldnames)
        csv_reader = list(csv.reader(csvfile))

        short = len(csv_reader[0]) == 14

        if region in short_locations and not short:
            # repladce follpowing with fstring
            raise Exception(f'short location has long csv {region}')
        if region in long_locations and short:
            raise Exception(f'long location has short csv {region}')

        return csv_reader, short


def get_random_rows(reader, file_per_location, station_years):
    # filter rows for available years
    rows_4_available_years = [
        row for row in reader if int(row[0]) in station_years
    ]
    rows_picked = random.choices(rows_4_available_years, k=file_per_location)
    return rows_picked


def parse_row(row, location, region, input_csv_headers):

    # Extract the year, month, day, and hour from the row
    year, month, day, hour = [int(row[x]) for x in range(4)]

    pd_row = {}

    # Add the location and region to the row
    pd_row['location'] = location.lower()
    pd_row['region'] = region.lower()

    # Compute the timestamp for the current offset
    timestamp = datetime.datetime(year, month, day, hour=hour)
    pd_row['TIMESTAMP'] = timestamp

    # Convert the data values to floats and add them to the row
    row[4:] = [float(x) for x in row[4:]]
    for label, data in zip(input_csv_headers, row[4:]):
        pd_row[label] = data

    return pd_row


def parse_rows(rows_picked, location, region, short, short_headers,
               long_headers):

    # Determine which headers to use
    input_csv_headers = short_headers if short else long_headers

    # Use list comprehension to parse each row
    pd_rows = [
        parse_row(row, location, region, input_csv_headers)
        for row in rows_picked
    ]

    return pd_rows


def shift_row_timestamp_2_beginning_of_window(row,
                                              weather_data_freq=WEATER_DATA_FREQ
                                             ):
    """Shift the timestamp of a row to the beginning of the 3-hour window."""
    timestamp = row['TIMESTAMP']
    timestamp = timestamp - datetime.timedelta(seconds=weather_data_freq)
    return row


def generate_extended_rows(pd_rows, timestamps_per_row):
    """Generate additional rows with equally
    spaced timestamps within the 3-hour window.

    if timestamps_per_row = 1, then timestamps are
         shifted to the middle of the window

    """

    if timestamps_per_row == 0:
        raise ValueError('timestamps_per_row must be greater than 0.')

    new_rows = []

    for row in pd_rows:
        timestamp = row['TIMESTAMP']

        timestamp_offsets = [
            datetime.timedelta(seconds=WEATER_DATA_FREQ * offset /
                               (timestamps_per_row + 1))
            for offset in range(1, timestamps_per_row + 1)
        ]

        for offset in timestamp_offsets:
            new_row = row.copy()
            new_timestamp = timestamp + offset
            new_row['TIMESTAMP'] = new_timestamp
            new_rows.append(new_row.copy())

    return new_rows


def csv_path_per_regloc(data_folder):

    station_csv = {}
    for region_path in glob.glob(f'{data_folder}/*'):
        locations = glob.glob(region_path + '/sm_products_by_station/*')
        region = Path(region_path).name.lower()
        for location_path in locations:
            location = Path(location_path).stem.split('_')[-1]
            if region != location[:-2].lower():
                print(region, location)
            location = location[-2:]
            # print(location_path)
            if region == 'ivvavik':
                location = 'SINP' + location
            station_csv[(region, location)] = location_path
            # print(region,location)
        # print(len(locations))
    return station_csv


def year_per_regloc(station_csv, file_properties_df):

    station_years = {}
    for region, location in station_csv.keys():
        region_filtered = file_properties_df[file_properties_df['region'] ==
                                             region]
        loc_reg_filtered = region_filtered[region_filtered['locationId'] ==
                                           location]

        # print(region,location)
        unique_years = (loc_reg_filtered.year.unique())
        unique_years = [int(year) for year in unique_years if int(year) > 2018]
        # print(unique_years)
        station_years[(region, location)] = unique_years
    return station_years


def load_weather_data(region, location, fname, short_locations, long_locations,
                      short_input_csv_headers, long_input_csv_headers):
    csv_reader, short = load_rows(fname, region, short_locations,
                                  long_locations)
    pd_rows = parse_rows(csv_reader, location, region, short,
                         short_input_csv_headers, long_input_csv_headers)
    pd_rows = [
        shift_row_timestamp_2_beginning_of_window(row) for row in pd_rows
    ]
    data = pd.DataFrame(pd_rows)
    data['rain_precip_mm'] = data['rain_precip'] * 1000
    return data


# Find the bin/row index for each timestamp in the list
def get_bin_index(interval_index, timestamp):
    try:
        index = interval_index.get_loc(timestamp)
    except KeyError:
        index = -1
    return index


def get_weather_index(weather_data,
                      timestamp_fromat=TIMESTAMP_FORMAT,
                      weather_data_freq=WEATER_DATA_FREQ):

    if not pd.api.types.is_datetime64_dtype(weather_data['TIMESTAMP']):
        # Do nothing, the column is already in datetime type
        weather_data['timestamp_start'] = weather_data['TIMESTAMP'].apply(
            lambda x: pd.to_datetime(x, format=timestamp_fromat))
    else:
        weather_data['timestamp_start'] = weather_data['TIMESTAMP']

    weather_data['timestamp_end'] = weather_data[
        'timestamp_start'] + pd.Timedelta(seconds=weather_data_freq)
    weather_data.sort_values(by='timestamp_start', inplace=True)

    interval_index = pd.IntervalIndex.from_arrays(
        weather_data['timestamp_start'],
        weather_data['timestamp_end'],
        closed='left')
    return interval_index


def get_random_timestamp(start, end):
    # Generate a random number of seconds between 0 and the duration of the audio clip (in seconds)
    audio_duration = int((end - start).total_seconds())
    random_seconds = random.randint(0, audio_duration)

    # Add the random number of seconds to the start time and subtract 10 seconds
    # to get a random timestamp within the range of start and end with a buffer of 10 seconds
    random_timestamp = start + pd.Timedelta(
        seconds=random_seconds) - pd.Timedelta(seconds=10)

    # Ensure that the random timestamp is within the range of start and end
    random_timestamp = np.clip(random_timestamp, start, end)

    return random_timestamp


def get_weather_rows(filtered_files,
                     weather_data,
                     file_per_location,
                     timestamp_fromat=TIMESTAMP_FORMAT,
                     weather_data_freq=WEATER_DATA_FREQ):
    interval_index = get_weather_index(weather_data,
                                       timestamp_fromat=timestamp_fromat,
                                       weather_data_freq=weather_data_freq)
    valid_indices = []
    valid_timestamps = []

    while len(valid_indices) < file_per_location:
        audio_sample = filtered_files.sample(n=1)
        random_timestamp = get_random_timestamp(
            audio_sample['timestamp'].iloc[0],
            audio_sample['timestampEnd'].iloc[0])
        weather_bind_index = get_bin_index(interval_index, random_timestamp)

        if weather_bind_index != -1:
            valid_indices.append(weather_bind_index)
            valid_timestamps.append(random_timestamp)

    weather_rows = weather_data.iloc[valid_indices].copy()
    weather_rows.loc[:, 'timestamp_orig_weather'] = weather_rows['TIMESTAMP']
    weather_rows.loc[:, 'TIMESTAMP'] = valid_timestamps
    return weather_rows


def load_neon_data(tool_weather_data_path,
                   length=5,
                   location='',
                   region='',
                   local_time_zone='America/Anchorage'):
    neon_files = glob.glob(f'{tool_weather_data_path}/*TOOL*/*{length}min*.csv')
    neon_df = []
    for file in neon_files:
        w_d = pd.read_csv(file)
        w_d['startDateTime'] = pd.to_datetime(
            w_d['startDateTime']).dt.tz_convert(local_time_zone).dt.tz_localize(
                None)
        w_d['endDateTime'] = pd.to_datetime(w_d['endDateTime']).dt.tz_convert(
            local_time_zone).dt.tz_localize(None)
        if location != '':
            w_d['location'] = location.lower()
        if region != '':
            w_d['region'] = region.lower()

        w_d['TIMESTAMP'] = w_d['startDateTime']
        neon_df.append(w_d)
    # priPrecipFinalQF is the quality flag for precipitation
    # 0 is good, 1 is bad
    neon_df = pd.concat(neon_df)
    neon_df = neon_df.sort_values(by=['startDateTime'])
    neon_df = neon_df.reset_index(drop=True)

    return neon_df
