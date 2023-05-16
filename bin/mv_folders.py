import os
import shutil
from pathlib import Path
import argparse


def move_and_rename_folder(parent_folder, folder_name):
    # Create tmp directory
    tmp_dir = Path(parent_folder) / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Move ./folder_name/Data to ./tmp/ and rename it to folder_name
    data_dir = Path(parent_folder) / folder_name / "Data"
    shutil.move(str(data_dir), str(tmp_dir / folder_name))

    # Remove ./folder_name directory
    shutil.rmtree(str(Path(parent_folder) / folder_name))

    # Move ./tmp/folder_name to ./folder_name
    shutil.move(str(tmp_dir / folder_name),
                str(Path(parent_folder) / folder_name))

    # Remove tmp directory if it's empty
    if not any(tmp_dir.iterdir()):  # Check if directory is empty
        tmp_dir.rmdir()
    else:
        print(f"Directory {tmp_dir} is not empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Move and rename folders.')
    parser.add_argument(
        'parent_folder',
        type=str,
        help='The parent folder containing subfolders to be moved and renamed')

    args = parser.parse_args()

    parent_folder_path = Path(args.parent_folder).resolve()
    for subfolder in parent_folder_path.iterdir():
        if subfolder.is_dir():
            move_and_rename_folder(str(parent_folder_path), subfolder.name)
