"""
Copy wav files in a directory tree to another directory tree of flac files.
If third and fourth arguments (K and N) are provided, then take every Nth
filename starting at position K (for parallelization across several
simultaneous runs).
"""
import sys
import csv
import fcntl
from pathlib import Path
from nna.labeling_utils import run_cmd

CSV_FILE = "processed_files.csv"


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


def process_wav_files(src_dir, dst_dir, k=1, n=1, dry_run=False):
    wav_files = get_wav_files(src_dir, k, n)
    errors = []
    processed_files = load_processed_files()

    for fullwav in wav_files:
        error = process_single_wav_file(src_dir, dst_dir, fullwav,
                                        processed_files, dry_run)
        if error:
            errors.append((fullwav, error))

    print_errors(errors)


def get_wav_files(src_dir, k, n):
    wav_files = sorted([
        path for path in Path(src_dir).rglob("*")
        if path.suffix.lower() == ".wav"
    ])
    wav_files = wav_files[k - 1::n]
    return wav_files


def process_single_wav_file(src_dir, dst_dir, fullwav, processed_files,
                            dry_run):
    if str(fullwav) in processed_files:
        # print(f"Skipping {fullwav} as it's already processed.")
        return None

    wav = fullwav.relative_to(src_dir)
    outfile = (dst_dir / wav).with_suffix(".flac")

    create_directory(dst_dir, wav)
    _, error, returncode = run_cmd(
        ["ffmpeg", "-y", "-nostdin", "-i",
         str(fullwav),
         str(outfile)],
        dry_run=dry_run,
    )

    if returncode != 0:
        print(f"Error processing {fullwav}: {error}")
    # else:
    #     print(f"Successfully processed {fullwav}")

    append_to_csv(fullwav, outfile, error)
    return error


def create_directory(dst_dir, wav):
    subdir = wav.parent
    (dst_dir / subdir).mkdir(parents=True, exist_ok=True)


def append_to_csv(fullwav, outfile, error):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(fullwav), str(outfile), error or ""])
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)


def print_errors(errors):
    if errors:
        print("The following files had errors:")
        for fullwav, error in errors:
            print(f"{fullwav}: {error}")


def parse_arguments():
    argc = len(sys.argv)
    if argc < 3 or argc > 6:
        print(f"Usage: {sys.argv[0]} src_dir dst_dir [k n] [--dry-run]")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    k = int(sys.argv[3]) if argc > 3 and sys.argv[3].isdigit() else 1
    n = int(sys.argv[4]) if argc > 4 and sys.argv[4].isdigit() else 1
    dry_run = "--dry-run" in sys.argv

    return src_dir, dst_dir, k, n, dry_run


if __name__ == "__main__":
    # Initialize the CSV file with headers if it doesn't exist
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile_g:
            csv_writer_g = csv.writer(csvfile_g)
            csv_writer_g.writerow(["Source File", "Destination File", "Error"])

    process_wav_files(*parse_arguments())
