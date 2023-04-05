''' This script is used to generate weather data for the NNA dataset.
'''
import os

# import itertools

import glob

from pathlib import Path
import csv
import random
import datetime

TIMESTAMP_FORMAT = '%Y-%m-%d_%H:%M:%S'
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


def parse_rows(rows_picked, location, region, short, short_headers,
               long_headers, timestamps_per_row):
    # Determine which headers to use
    input_csv_headers = short_headers if short else long_headers

    pd_rows = []

    # Iterate through each row
    for row in rows_picked:
        # Extract the year, month, day, and hour from the row
        year, month, day, hour = [int(row[x]) for x in range(4)]

        # Compute equally spaced timestamps within the 3-hour window
        timestamp_offsets = [
            datetime.timedelta(hours=3 * offset / (timestamps_per_row + 1))
            for offset in range(1, timestamps_per_row + 1)
        ]

        # Iterate through each timestamp offset
        for timestamp_offset in timestamp_offsets:
            # Create a new dictionary to store the row data
            pd_row = {}

            # Add the location and region to the row
            pd_row["location"] = location
            pd_row["region"] = region.lower()

            # Compute the timestamp for the current offset
            timestamp = datetime.datetime(year, month, day, hour=hour)
            timestamp += timestamp_offset
            pd_row["TIMESTAMP"] = timestamp.strftime(TIMESTAMP_FORMAT)

            # Convert the data values to floats and add them to the row
            row[4:] = [float(x) for x in row[4:]]
            for label, data in zip(input_csv_headers, row[4:]):
                pd_row[label] = data

            # Add the row to the list of rows
            pd_rows.append(pd_row)

    return pd_rows


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
