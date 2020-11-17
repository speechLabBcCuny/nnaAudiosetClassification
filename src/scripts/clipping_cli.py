import argparse

# file_properties_df_path = "../../data/prudhoeAndAnwr4photoExp_dataV1.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--location_id",
                        help="location_id such as 11,12 etc",
                        required=True)
    parser.add_argument("-O",
                        "--output_folder",
                        help="output folder",
                        default="./clipping_output/")
    parser.add_argument("-d",
                        "--file_database",
                        help="path to file_properties_df_path",
                        required=True)
    parser.add_argument("-t",
                        "--clipping_threshold",
                        help="path to file_properties_df_path",
                        type=float,
                        default=1.0)

    args = parser.parse_args()

    from nna import clippingutils
    import pandas as pd

    clipping_threshold = args.clipping_threshold

    file_properties_df_path = args.file_database
    file_properties_df = pd.read_pickle(file_properties_df_path)
    clipping_results_path = args.output_folder
    location_id = args.location_id
    location_id_filtered = file_properties_df[file_properties_df.locationId ==
                                              location_id]
    all_results_dict, files_w_errors = clippingutils.run_task_save(
        location_id_filtered.index, location_id, clipping_results_path,
        clipping_threshold)

    if files_w_errors:
        for file_path in files_w_errors:
            print(file_path)

# python clipping_cli.py --location_id "11" --file_database "../../data/prudhoeAndAnwr4photoExp_dataV1.pkl"
# cat locationIds.txt | parallel -P 40 -n 1 -q python clipping_cli.py --location_id {} --file_database "../../data/prudhoeAndAnwr4photoExp_dataV1.pkl"
