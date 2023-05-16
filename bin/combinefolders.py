import os
import shutil
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Merge 'Data' directories.")

# Add the positional argument for the main directory
parser.add_argument('main_dir', type=str, help="Main directory path")

# Parse the command-line arguments
args = parser.parse_args()
main_dir = args.main_dir
# Get the list of all folders
folders = os.listdir(main_dir)

# Iterate over the list of folders
for folder in folders:
    # Check if folder ends with 'B'
    if folder.endswith('B'):
        # Define the source directory (Data folder in B)
        src_dir = os.path.join(main_dir, folder, 'Data')

        # Define the destination directory (Data folder in corresponding A)
        dest_dir = os.path.join(main_dir, folder[:-1] + 'A', 'Data')

        # Make sure both directories exist
        if os.path.exists(src_dir) and os.path.exists(dest_dir):
            # Get the list of files in the source directory
            files = os.listdir(src_dir)

            # Iterate over the list of files
            for file in files:
                # Define the full file paths
                full_file_path = os.path.join(src_dir, file)
                destination = os.path.join(dest_dir, file)
                # Move the file if the destination directory exists
                if not os.path.exists(destination):
                    shutil.move(full_file_path, destination)
                else:
                    print(f'ERROR: Destination file does exist {destination}')
        else:
            # Print a message when the source directory does not exist
            print(f'ERROR: Source directory does not exist {src_dir}')

# find ./ -depth -type d -name "Data" -execdir mv {} 2022 \;


sudo /scratch/enis/conda/envs/expenv1/bin/python /home/enis/projects/nna/bin/copywav2flac.py --src_dir  /tank/data/nna/recover_files/2022/ --dst_dir /mnt/seagate/region/ \
&& /scratch/enis/conda/envs/speechEnv/bin/python \
/home/enis/projects/nna/src/nna/slack_message.py \
    -t "cpu job ended" -m 'flac recover finished ' \
|| /scratch/enis/conda/envs/speechEnv/bin/python \
/home/enis/projects/nna/src/nna/slack_message.py \
    -t "cpu job FAILED" -m 'flac recover FAILED'