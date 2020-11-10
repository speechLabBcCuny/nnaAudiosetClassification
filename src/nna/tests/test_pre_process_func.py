from unittest.mock import Mock

import pytest

from pathlib import Path
from subprocess import Popen, PIPE

import numpy as np

from nna import pre_process_func

TEST_DIR = Path("tests/")
EXAMPLE_MODELS_DIR = TEST_DIR / "example_models/"
EXAMPLE_OUTPUT_DIR = TEST_DIR / "example_outputs"
OUTPUT_DIR = Path("aggregates")


@pytest.fixture
def settings_mock():
    return


class TestAggregator:
    def example_test(self):
        assert True


@pytest.fixture
def download_vggish():
    import os
    import urllib.request
    file_path = "tests/example_models/vggish/vggish_model.ckpt"
    file_url = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
    if not os.path.exists(file_path):
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(file_url, file_path)
    return file_path


def test_relative2outputdir():
    #params.INPUT_DIR_PARENT = "/home/data/nna/stinchcomb/NUI_DATA/"
    input_dir_parent = "/home/data/nna/stinchcomb/NUI_DATA/"

    mp3_file_path = "/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/my_music.mp3"
    # relative_path = "18 Fish Creek 4/July 2016"
    output_dir = "/output/directory"

    expected_absolute_output_dir = Path(
        "/output/directory/18 Fish Creek 4/July 2016/")

    absolute_output_dir = pre_process_func.relative2outputdir(
        mp3_file_path, output_dir, input_dir_parent)
    assert (expected_absolute_output_dir == absolute_output_dir)


def test_preb_names():
    mp3_file_path = "/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    output_dicretory = "/scratch/enis/data/nna/NUI_DATA/"
    abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"

    output = pre_process_func.preb_names(mp3_file_path, output_dicretory,
                                         abs_input_path)
    common_path = "/scratch/enis/data/nna/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935"
    segments_folder = common_path + "_segments/"
    embeddings_file_name = common_path + "_embeddings.npy"
    pre_processed_folder = common_path + "_preprocessed/"
    expected_output = (segments_folder, embeddings_file_name,
                       pre_processed_folder)
    expected_output = tuple([Path(path_str) for path_str in expected_output])
    assert (expected_output == output)


def test_divide_mp3():
    mp3_file_path = "tests/data/3hours30min.mp3"
    segments_folder = "tests/data/segments/"
    Path(segments_folder).mkdir(parents=True, exist_ok=True)
    mp3_segments = pre_process_func.divide_mp3(mp3_file_path,
                                               segments_folder,
                                               segment_len="01:00:00",
                                               conda_env_name="soundEnv")
    assert (len(mp3_segments) == 4)

    args = ("ffprobe", "-show_entries", "format=duration", "-i",
            "tests/data/segments/output000.mp3")
    popen = Popen(args, stdout=PIPE)
    popen.wait()
    output = popen.stdout.read()
    length_min = str(output).split("\\n")[1].split("=")[1].split(".")[0]
    assert (length_min == "3600")

    args = ("ffprobe", "-show_entries", "format=duration", "-i",
            "tests/data/segments/output003.mp3")
    popen = Popen(args, stdout=PIPE)
    popen.wait()
    output = popen.stdout.read()
    length_min = str(output).split("\\n")[1].split("=")[1].split(".")[0]
    assert (length_min == "1799")
    pre_process_func.rmv_folder(segments_folder)


def test_load_mp3():
    input_file_path = "tests/data/10minutes.mp3"
    wav_data, sr = pre_process_func.load_mp3(input_file_path)
    assert (type(wav_data) == np.ndarray)
    assert (wav_data.shape == (26459136, 2))
    assert (sr == 44100)

also tests mp3file_to_examples
def test_iterate_for_waveform_to_examples():
    input_file_path = "tests/data/10minutes.mp3"
    # wav_data,sr=pre_process_func.load_mp3(input_file_path)
    # sound = pre_process_func.iterate_for_waveform_to_examples(wav_data,sr)
    sound = pre_process_func.mp3file_to_examples(input_file_path)
    assert (type(sound) == np.ndarray)
    # 10 minute = 600 seconds = 600 samples
    assert (sound.shape == (600, 96, 64))
    assert (sound.dtype == np.float32)
    assert (np.count_nonzero(sound) / sound.size > 0.98)
    assert (np.count_nonzero(sound) == sound.size)


def test_parallel_pre_process():
    #TODO get this with os.cwd

    TEST_DIR = Path("tests/")
    EXAMPLE_MODELS_DIR = TEST_DIR / "example_models/"
    EXAMPLE_OUTPUT_DIR = TEST_DIR / "example_outputs"
    OUTPUT_DIR = Path("aggregates")

    root_dir = "/Users/berk/Documents/workspace/speech_audio_understanding/src/"
    # input_path_list=[root_dir+"tests/data/3hours30min.mp3",]

    input_path_list = [
        root_dir + "tests/data/1hour20min.flac",
    ]

    pre_process_func.parallel_pre_process(input_path_list,
                                          output_dir="./tests/data/output",
                                          input_dir_parent=root_dir,
                                          cpu_count=2,
                                          segment_len="02:00:00",
                                          logs_file_path=LOGS_FILE)

    expected_file3 = "tests/data/output/tests/data/3hours30min_preprocessed/output003_preprocessed.npy"
    expected_file2 = "tests/data/output/tests/data/3hours30min_preprocessed/output002_preprocessed.npy"
    last_part = np.load(expected_file3)
    second_part = np.load(expected_file2)
    assert (last_part.shape == (1800, 96, 64))
    assert (second_part.shape == (3600, 96, 64))
    assert (np.count_nonzero(second_part) == second_part.size)
    assert (np.count_nonzero(last_part) == last_part.size)
