'''Calculate clipping info of all files from a specific location.

    TODO: ffmpeg executable hard coded in clippint utils for pydub
    

    Ex:
        # single location
        python clipping_cli.py --region 'anwr' --location '11' --file_database '../../data/prudhoeAndAnwr4photoExp_dataV1.pkl'
        # from list of locations, run it in parallel 

        cat '/scratch/enis/data/nna/flow_tracking/region-location-list_all-fields_datav4_without-stinchcomb.txt' | \
            parallel --csv -P 40 -n 1 -q python clipping_cli.py \
            --region {1} --location {2} --output_folder \
            '/scratch/enis/data/nna/clipping_info/all-merged_2021-02-10/' \
            --file_database '/scratch/enis/data/nna/database/allFields_dataV4.pkl' >>logs_2021-02-10.txt 2>&1 

    # How to filter a dataframe by another and create list of locations to process
        import pandas as pd

        latest_path = "/scratch/enis/data/nna/database/allFields_dataV4.pkl"
        latest = pd.read_pickle(latest_path)

        older_path = "/home/enis/projects/nna/data/prudhoeAndAnwr4photoExp_dataV1.pkl"
        older = pd.read_pickle(older_path)

        index_diff=list(set(latest.index)-set(older.index))
        df_diff=latest.loc[index_diff]

        df_diff.to_pickle("/scratch/enis/data/nna/database/all-fields_datav4_without-prudhoe-anwr.pkl")

        region_loc_diff = list(zip(df_diff.region,df_diff.locationId))
        region_loc_diff_set = list(set(region_loc_diff))
        region_loc_diff_set.sort()
        with open('/scratch/enis/data/nna/flow_tracking/region-location-list_all-fields_datav4_without-prudhoe-anwr.txt',"w") as ff:
            for line in region_loc_diff_set:
                line = ','.join(line)
                _ = ff.write(line+"\n")
'''
import argparse

# file_properties_df_path = '../../data/prudhoeAndAnwr4photoExp_dataV1.pkl'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--region',
        help='region-location_id such as anwr or stinchcomb etc',
        required=True)
    parser.add_argument(
        '--location',
        help='location_id such as 11 or 14-Rocky etc',
        required=True)
    
    parser.add_argument('-O',
                        '--output_folder',
                        help='output folder',
                        default='./clipping_output/')
    parser.add_argument('-d',
                        '--file_database',
                        help='path to file_properties_df_path',
                        required=True)
    parser.add_argument('-t',
                        '--clipping_threshold',
                        help='path to file_properties_df_path',
                        type=float,
                        default=1.0)

    args = parser.parse_args()

    from nna import clippingutils
    import pandas as pd

    clipping_threshold = args.clipping_threshold

    file_properties_df_path = args.file_database
    file_properties_df = pd.read_pickle(file_properties_df_path)
    clipping_results_path = args.output_folder
    location = args.location
    region = args.region
    region_location = f'{region}-{location}'
    # file_name = region_location
    # print(region,location)
    region_filtered = file_properties_df[file_properties_df.region == region]
    # print(len(region_filtered))
    location_id_filtered = region_filtered[region_filtered.locationId ==
                                           location]
    # print(len(location_id_filtered))
    duration_filtered = location_id_filtered[location_id_filtered.durationSec > 0]
    # print(len(duration_filtered.index))

    all_results_dict, files_w_errors = clippingutils.run_task_save(
        duration_filtered.index,
        region_location,
        clipping_results_path,
        clipping_threshold,
    )

    if files_w_errors:
        for file_path in files_w_errors:
            print(file_path)
