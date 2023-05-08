import sys
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

CSV_FILE = "audio_error_files.csv"

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"]


def check_audio_files(folder_path):
    files = sorted(list(Path(folder_path).rglob("*")))
    audio_files = sorted(
        [path for path in files if path.suffix.lower() in AUDIO_EXTENSIONS])

    for audio_file in audio_files:
        check_audio_file(audio_file)

    print_extension_counts(files)


def check_audio_file(wav_file):
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format", "-of", "csv=p=0",
        str(wav_file)
    ]

    try:
        output = subprocess.check_output(command,
                                         stderr=subprocess.STDOUT,
                                         text=True)
    except subprocess.CalledProcessError as e:
        output = e.output

    if output:
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

    # Initialize the CSV file with headers if it doesn't exist
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile_g:
            csv_writer_g = csv.writer(csvfile_g)
            csv_writer_g.writerow(["File", "Error"])

    check_audio_files(folder_path_g)
