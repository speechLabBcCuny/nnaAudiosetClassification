"""
Copy wav files in a directory tree to another directory tree of flac files.

"""
import csv
import argparse
import concurrent.futures
from pathlib import Path
import threading
import subprocess
import shlex
import signal

CSV_FILE = "processed_files.csv"
# Create a global lock for CSV file writing
CSV_FILE_LOCK = threading.Lock()


def run_cmd(cmd, dry_run=False, verbose=True, timeout=None):
    if dry_run:
        return ''.join(cmd), '\n cmd not run, dry_run is True', 'no run'

    try:
        # Execute the command
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   shell=False)

        # Wait for the command to complete with an optional timeout
        stdout, stderr = process.communicate(timeout=timeout)
        # Get the return code of the command
        returncode = process.returncode

        # Retrieve the output and error messages
        output = stdout.strip()
        error = stderr.strip()

        if returncode != 0 and verbose:
            print('---------')
            print(cmd)
            print('Output:\n' + output)
            print('Error:\n' + error)

        return output, error, returncode

    except subprocess.TimeoutExpired:
        # Handle timeout if specified
        process.kill()
        return None, 'Command execution timed out', -1

    except Exception as e:
        # Handle any other exceptions
        return None, str(e), -1


# def run_cmd(cmd, dry_run=False, verbose=True):
#     if dry_run:
#         return ''.join(cmd), '\n cmd not run, dry_run is True', 'no run'

#     proc = subprocess.Popen(cmd,
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE,
#                             text=True)
#     output, error = proc.communicate()

#     if proc.returncode != 0 and verbose:
#         print('---------')
#         print(cmd)
#         print('Output: \n' + output)
#         print('Error: \n' + error)

#     return output, error, proc.returncode


def load_processed_files():
    processed_files = set()

    if Path(CSV_FILE).exists():
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                if row:
                    processed_files.add(row[0])

    return processed_files


def get_wav_files_left(src_dir):
    wav_files = set(map(str,
                        get_wav_files(src_dir)))  # Convert wav_files to a set
    processed_files = load_processed_files()

    # Use set subtraction to find files that haven't been processed yet
    wav_files_to_process = wav_files - processed_files
    wav_files_to_process = map(Path, wav_files_to_process)
    wav_files_to_process = sorted(list(wav_files_to_process))

    return wav_files_to_process


def process_wav_files(src_dir, dst_dir, dry_run=False):
    errors = []

    wav_files_to_process = get_wav_files_left(src_dir)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_wav = {
            executor.submit(process_single_wav_file, src_dir, dst_dir, fullwav,
                            dry_run): fullwav
            for fullwav in wav_files_to_process
        }
        for future in concurrent.futures.as_completed(future_to_wav):
            fullwav = future_to_wav[future]
            try:
                error = future.result()
                if error:
                    errors.append((fullwav, error))
            except Exception as exc:
                print(f"{fullwav!r} generated an exception: {exc}")

    print_errors(errors)


def get_wav_files(src_dir):
    wav_files = sorted([
        path for path in Path(src_dir).rglob("*")
        if path.suffix.lower() == ".wav"
    ])
    return wav_files


def process_single_wav_file(src_dir,
                            dst_dir,
                            fullwav,
                            dry_run,
                            ffmpeg_path="/home/enis/sbin/ffmpeg"):

    wav = fullwav.relative_to(src_dir)
    outfile = (dst_dir / wav).with_suffix(".flac")
    create_directory(dst_dir, wav)
    _, error, returncode = run_cmd(
        [
            ffmpeg_path, "-y", "-nostdin", "-loglevel", "error", "-i",
            str(fullwav),
            str(outfile)
        ],
        dry_run=dry_run,
    )
    if returncode != 0:
        print(f"Error processing {fullwav}: {error}")
        if error == '':
            error = 'Error: no error message'

    append_to_csv(fullwav, outfile, error, returncode)
    return error


def create_directory(dst_dir, wav):
    subdir = wav.parent
    (dst_dir / subdir).mkdir(parents=True, exist_ok=True)


def append_to_csv(fullwav, outfile, error, returncode):
    with CSV_FILE_LOCK:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [str(fullwav),
                 str(outfile), error or "", returncode])


def print_errors(errors):
    if errors:
        print("The following files had errors:")
        for fullwav, error in errors:
            print(f"{fullwav}: {error}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="Source directory path")
    parser.add_argument("--dst_dir",
                        type=str,
                        help="Destination directory path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")

    args = parser.parse_args()
    return args.src_dir, args.dst_dir, args.dry_run


if __name__ == "__main__":
    # Initialize the CSV file with headers if it doesn"t exist
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile_g:
            csv_writer_g = csv.writer(csvfile_g)
            csv_writer_g.writerow(
                ["Source File", "Destination File", "Error", "Return Code"])

    process_wav_files(*parse_arguments())
