"""Tests for pre_process_func.py"""

import os
import urllib.request
import shutil

import pytest

from pathlib import Path
from subprocess import Popen, PIPE

import numpy as np

from nna import pre_process_func
from nna.params import PRE_PROCESSING_queue, PRE_PROCESSED_queue
# from nna.params import LOGS_FILE
from testparams import INPUTS_OUTPUTS_PATH
from testparams import SOUND_EXAMPLES
from testparams import LOGS_FILE
from testparams import CONDA_ENV_NAME
IO_clippingutils_path = INPUTS_OUTPUTS_PATH / "pre_process_func"

# TEST_DIR = Path("tests/")
# EXAMPLE_MODELS_DIR = TEST_DIR / "example_models/"
# EXAMPLE_OUTPUT_DIR = TEST_DIR / "example_outputs"
# OUTPUT_DIR = Path("aggregates")


@pytest.fixture
def settings_mock():
    return


class TestAggregator:
    def example_test(self):
        assert True


@pytest.fixture
def download_vggish():
    file_path = "tests/example_models/vggish/vggish_model.ckpt"
    file_url = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
    if not os.path.exists(file_path):
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(file_url, file_path)
    return file_path


def test_relative2outputdir():
    #params.INPUT_DIR_PARENT = "/home/data/nna/stinchcomb/NUI_DATA/"
    input_dir_parent = Path("/home/data/nna/stinchcomb/NUI_DATA/")

    mp3_file_path = input_dir_parent / "18 Fish Creek 4/July 2016/my_music.mp3"
    # relative_path = "18 Fish Creek 4/July 2016"
    output_dir = "/output/directory"

    expected_absolute_output_dir = Path(
        "/output/directory/18 Fish Creek 4/July 2016/")

    absolute_output_dir = pre_process_func.relative2outputdir(
        mp3_file_path, output_dir, input_dir_parent)
    assert expected_absolute_output_dir == absolute_output_dir


def test_preb_names():
    mp3_file_path = ("/home/data/nna/stinchcomb/NUI_DATA/" +
                     "18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3")
    output_dicretory = "/scratch/enis/data/nna/NUI_DATA/"
    abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"

    output = pre_process_func.preb_names(mp3_file_path, output_dicretory,
                                         abs_input_path)
    common_path = ("/scratch/enis/data/nna/NUI_DATA/" +
                   "18 Fish Creek 4/July 2016/FSHCK4_20160629_194935")
    segments_folder = common_path + "_segments/"
    embeddings_file_name = common_path + "_embeddings.npy"
    pre_processed_folder = common_path + "_preprocessed/"
    expected_output = (segments_folder, embeddings_file_name,
                       pre_processed_folder)
    expected_output = [Path(path_str) for path_str in expected_output]
    assert tuple(expected_output) == output


def test_divide_mp3():
    mp3_file_path = SOUND_EXAMPLES / "3hours30min.mp3"
    segments_folder = IO_clippingutils_path / "divide_mp3" / "outputs"
    Path(segments_folder).mkdir(parents=True, exist_ok=True)

    mp3_segments = pre_process_func.divide_mp3(mp3_file_path,
                                               segments_folder,
                                               segment_len="01:00:00",
                                               conda_env_name=CONDA_ENV_NAME)
    assert len(mp3_segments) == 4

    command_list = ["conda", "run", "-n", CONDA_ENV_NAME]
    command_list.extend([
        "ffprobe", "-show_entries", "format=duration", "-i",
        str(segments_folder / "output000.mp3")
    ])
    popen_instance = Popen(command_list, stdout=PIPE, stderr=PIPE)
    output_b = popen_instance.communicate(b"\n")
    output = [i.decode("ascii") for i in output_b]

    if output[0] == "" or output[0] == "N/A":
        print("command run with ERROR: {}".format(command_list))
        print(output[1])
    print(output)
    length_min = str(output[0]).split("\n")[1].split("=")[1].split(".")[0]
    assert length_min == "3600"

    command_list = ["conda", "run", "-n", CONDA_ENV_NAME]
    command_list.extend([
        "ffprobe", "-show_entries", "format=duration", "-i",
        str(segments_folder / "output003.mp3")
    ])
    popen_instance = Popen(command_list, stdout=PIPE, stderr=PIPE)
    output_b = popen_instance.communicate(b"\n")
    output = [i.decode("ascii") for i in output_b]
    if output[0] == "" or output[0] == "N/A":
        print("command run with ERROR: {}".format(command_list))
        print(output[1])
    length_min = str(output[0]).split("\n")[1].split("=")[1].split(".")[0]
    assert length_min == "1799"
    pre_process_func.rmv_folder(segments_folder)


def test_load_mp3():
    input_file_path = SOUND_EXAMPLES / "10minutes.mp3"
    wav_data, sr = pre_process_func.load_mp3(input_file_path)
    assert isinstance(wav_data, np.ndarray)
    assert wav_data.shape == (26459136, 2)
    assert sr == 44100


# also tests mp3file_to_examples
def test_iterate_for_waveform_to_examples():
    input_file_path = SOUND_EXAMPLES / "10minutes.mp3"
    # wav_data,sr=pre_process_func.load_mp3(input_file_path)
    # sound = pre_process_func.iterate_for_waveform_to_examples(wav_data,sr)
    sound = pre_process_func.mp3file_to_examples(input_file_path)
    assert isinstance(sound, np.ndarray)
    # 10 minute = 600 seconds = 600 samples
    assert sound.shape == (600, 96, 64)
    assert sound.dtype == np.float32
    assert np.count_nonzero(sound) / sound.size > 0.98
    assert np.count_nonzero(sound) == sound.size


@pytest.fixture(scope="function")
def log_folder():
    Path(PRE_PROCESSING_queue).unlink(missing_ok=True)
    Path(PRE_PROCESSED_queue).unlink(missing_ok=True)


def test_parallel_pre_process(log_folder):  #pylint: disable=W0621,W0613

    func_output_dir = IO_clippingutils_path / "parallel_pre_process" / "outputs"
    Path(func_output_dir).mkdir(parents=True, exist_ok=True)

    input_path_list = [
        SOUND_EXAMPLES / "1hour.mp3",
    ]

    pre_process_func.parallel_pre_process(input_path_list,
                                          output_dir=func_output_dir,
                                          input_dir_parent=SOUND_EXAMPLES,
                                          cpu_count=2,
                                          segment_len="00:20:00",
                                          logs_file_path=LOGS_FILE,
                                          conda_env_name=CONDA_ENV_NAME)

    expected_file2 = func_output_dir / (
        input_path_list[0].stem + "_preprocessed/output002_preprocessed.npy")
    expected_file1 = func_output_dir / (
        input_path_list[0].stem + "_preprocessed/output001_preprocessed.npy")
    part2 = np.load(expected_file2)
    part1 = np.load(expected_file1)

    assert np.sum(part2) == -18499336.0
    assert np.sum(part1) == -24182264.0
    assert part2.shape == (1200, 96, 64)
    assert part1.shape == (1200, 96, 64)
    assert np.count_nonzero(part1) == part1.size
    assert np.count_nonzero(part2) == part2.size
    try:
        shutil.rmtree(func_output_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
