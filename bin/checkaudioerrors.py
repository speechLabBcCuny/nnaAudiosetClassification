import sys
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

CSV_FILE = "audio_error_files.csv"

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"]


def check_audio_files(folder_path, ffprobe_path="/home/enis/sbin/ffprobe"):
    files = sorted(list(Path(folder_path).rglob("*")))
    audio_files = sorted(
        [path for path in files if path.suffix.lower() in AUDIO_EXTENSIONS])

    for audio_file in audio_files:
        check_audio_file(audio_file, ffprobe_path=ffprobe_path)

    print_extension_counts(files)


def check_audio_file(wav_file, ffprobe_path="/home/enis/sbin/ffprobe"):
    command = [
        ffprobe_path, "-v", "error", "-show_error", "-show_format",
        "-show_streams", "-print_format", "json",
        str(wav_file)
    ]

    try:
        output = subprocess.check_output(command,
                                         stderr=subprocess.STDOUT,
                                         text=True)
    except subprocess.CalledProcessError as e:
        output = e.output

    if "error" in output.lower():
        print(f"Error in {wav_file}:")
        print(output)
        write_to_csv(wav_file, output)


def write_to_csv(wav_file, error_message):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(wav_file), error_message])


def print_extension_counts(files):
    extension_counts = defaultdict(int)

    for path in files:
        if path.is_file():
            extension = path.suffix.lower()
            extension_counts[extension] += 1

    print("File extension counts:")
    for extension, count in extension_counts.items():
        print(f"{extension}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} folder_path")
        sys.exit(1)

    folder_path_g = sys.argv[1]
    ffprobe_path_g = sys.argv[2] if len(
        sys.argv) > 2 else "/home/enis/sbin/ffprobe"

    # Initialize the CSV file with headers if it doesn't exist
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile_g:
            csv_writer_g = csv.writer(csvfile_g)
            csv_writer_g.writerow(["File", "Error"])

    check_audio_files(folder_path_g, ffprobe_path=ffprobe_path_g)

# import os
# import shutil
# import pandas as pd

# def move_files_with_structure(file_list, target_folder):
#     for file_path in file_list:
#         relative_path = os.path.relpath(
#             file_path,
#             '/mnt/seagate/dempster')  # Get the relative path of the file
#         destination = os.path.join(target_folder, relative_path)
#         # os.makedirs(os.path.dirname(destination), exist_ok=True)  # Create the destination directory
#         # shutil.move(file_path, destination)
#         print(f"Moved {file_path} to {destination}")

# # Example usage
# a = pd.read_csv('/mnt/seagate/dempster/audio_error_files.csv')
# files = a['File'].to_list()
# target_folder = '/tank/data/nna/corrupted_files/dempster'

# move_files_with_structure(files, target_folder)
