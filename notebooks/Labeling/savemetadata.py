from pathlib import Path
import pandas as pd
import datetime

from collections import Counter

from nna.fileUtils import list_files, concurrent_get_media_duration, read_file_properties_v2


### Functions
def count_file_extensions(path_list):
    file_extensions = [Path(i).suffix.lower() for i in path_list]
    file_extensions_counter = Counter(file_extensions)
    return file_extensions_counter


# Load previous data
def load_previous_data(current_metadata_file):
    if not current_metadata_file:
        return None
    current_metadata_file = Path(current_metadata_file)
    if current_metadata_file.exists():
        if current_metadata_file.suffix == '.pkl':
            current_file_properties_df = pd.read_pickle(
                str(current_metadata_file))
        elif current_metadata_file.suffix == '.csv':
            current_file_properties_df = pd.read_csv(str(current_metadata_file),
                                                     index_col=0)
        else:
            raise ValueError('unknown file type for current_metadata_file')
        return current_file_properties_df
    else:
        print(f'{current_metadata_file} does NOT exists')


def get_new_files(files_path_list, current_file_properties_df):
    # remove files we already know about
    currentFileSet = set(current_file_properties_df.index)
    currentFileSet = set([str(m) for m in currentFileSet])
    foundFileSet = set(files_path_list)
    foundFileSet = foundFileSet.difference(currentFileSet)
    new_files_path_list = list(foundFileSet)
    print(
        f'new {len(new_files_path_list)} previously {len(currentFileSet)} total {len(files_path_list)}'
    )
    return new_files_path_list


# External Programs
# ffprobe version >= 4.3.1
ffprobe_path = '/home/enis/sbin/ffprobe'

# PARAMETERS for new DATABASE

# increase version number accordinly
previous_database_ver_str = 'V101'
new_database_ver_str = 'V102'
# where to save txt file storing length info
data_folder = '/scratch/enis/data/nna/database/'
#/scratch/enis/data/nna/database

# where
# path to search for audio files
search_path = '/tank/data/nna/real/'
ignore_folders = [
    '/tank/data/nna/real/stinchcomb/dups/',
    "/tank/data/nna/real/stinchcomb/excerpts/",
    "/tank/data/nna/real/stinchcomb/"
]

# create Relative Path names

# if we already have a list of files we can load them
# files_list_path=data_folder+"stinchcomb_files_pathV1.txt"
files_list_path = data_folder + f"allFields_path{new_database_ver_str}.txt"

# if we calculated audio lengths and saved them into text file,
# we can load them
fileswlen_path = data_folder + f"allFields_wlen_f{new_database_ver_str}.txt"
filesWError_out = data_folder + f"allFields_wERROR_f{new_database_ver_str}.txt"
files_w_wrong_name = data_folder + f"wrong_name_f{new_database_ver_str}.txt"
# do NOT add pkl extension at the end
pkl_file_name = f"allFields_data{new_database_ver_str}"

# this is the current info we have so we can check if we already processed a file before
current_pkl_file = data_folder + f"allFields_data{previous_database_ver_str}.pkl"

# Find files
# in given search path ignoring given directories
if Path(fileswlen_path).exists():
    with open(files_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        files_path_list = [line.strip() for line in lines]
else:
    files_path_list = list_files(search_path, ignore_folders)

print('total number of files:', len(files_path_list))
if Path(files_path_list[0]).is_dir():
    raise Exception('first file found is a directory, exiting, there is a bug')

files_suffix_counts = count_file_extensions
print('file extensions counts:\n')
print(files_suffix_counts(files_path_list))
print('\n')
previous_metadata = load_previous_data(current_pkl_file)
new_files_path_list = get_new_files(files_path_list, previous_metadata)

length_dict, files_werror = concurrent_get_media_duration(new_files_path_list,
                                                          ffprobe_path,
                                                          verbose=False)

if len(files_werror) > 0:
    print(
        f'there are {len(files_werror)} number of files with errors reading length'
    )
    print(files_werror)
    with open(filesWError_out, 'w', encoding='utf-8') as f:
        lines = [line[0] for line in files_werror]
        f.write("\n".join(lines) + "\n")
    print(f'files with errors are saved in {filesWError_out}')

file_properties, exceptions = read_file_properties_v2(list(length_dict.keys()),
                                                      debug=0)

if len(exceptions) > 0:
    print(f'there are {len(exceptions)} number of ' +
          'files with errors reading metadata from filename')
    print(exceptions)
    with open(files_w_wrong_name, 'a', encoding='utf-8') as f:
        lines = [line[0] for line in exceptions]
        f.write("\n".join(lines) + "\n")
    print(f'exceptions files are saved in {files_w_wrong_name}')
