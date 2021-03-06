"""Experiment running. Can only be imported from the experiment folder.

"""

# from genericpath import exists
import os

# # import run
# # import nna
# import torch
# import torch.nn as nn
# from torchvision import transforms

import numpy as np
import pytest
import torch
# from pathlib import Path
# from collections import Counter

# import wandb

# from ignite.metrics import Accuracy, Loss
# from ignite.contrib.metrics import ROC_AUC

# import runconfigs # type: ignore
# import modelarchs # type: ignore

# import nna.exp.megan as megan
# from nna.exp import runutils
from nna import dataimport

# @pytest.fixture(scope="function")
# def output_folder(request):
#     # print("setup")
#     file_name, lines, expected = request.param
#     # print(file_name.exists())
#     file_name = Path(file_name).with_suffix(".csv")
#     file_name.parent.mkdir(parents=True, exist_ok=True)
#     yield (file_name, lines, expected)
#     print("teardown")
#     print(file_name)
#     file_name.unlink(missing_ok=True)

# @pytest.mark.parametrize(
#     "output_folder",
#     test_data_save_to_csv,
#     indirect=True,
# )
# def test_save_to_csv(
#         # file_name,
#         # lines,
#         # expected,
#     output_folder):  #pylint:disable=W0621
#     file_name, lines, expected = output_folder


@pytest.fixture(scope='module')
def chdir_to_local_folder():
    '''change directory, we are testing a script from local directory. 
    '''
    # os.chdir('/home/enis/projects/nna/src/nna/exp/megan/run-2/')
    os.chdir('/Users/berk/Documents/research/nna/src/nna/exp/megan/run-3/')


test_data_data_to_samples = [
    (chdir_to_local_folder, 10, 54, 48000.0, 'random/path/to/audio/file'),
    (chdir_to_local_folder, 10, 5.1, 48000.0, 'random/path/to/audio/file'),
    (chdir_to_local_folder, 10.0, 5.1, 48000.0, 'random/path/to/audio/file'),
    (chdir_to_local_folder, 10.0, 58.1, 48000, 'random/path/to/audio/file'),
    (chdir_to_local_folder, 10.0, 54.1, 48000, 'random/path/to/audio/file'),
]


@pytest.mark.parametrize(
    "chdir_to_local_folder,excerpt_len,audio_file_len,audio_sr,audio_file_path",
    test_data_data_to_samples,
    # indirect=True,
)
def test_data_to_samples(chdir_to_local_folder, excerpt_len, audio_file_len,
                         audio_sr, audio_file_path):
    import run
    print(excerpt_len, audio_file_len, audio_sr, audio_file_path)
    # excerpt_len,audio_file_len,audio_sr,audio_file_path  = inputs

    sound_ins = dataimport.Audio(audio_file_path, audio_file_len)
    sound_ins.sr = audio_sr
    sound_ins.data = np.ones(int(audio_file_len * audio_sr))
    run.data_to_samples(sound_ins, excerpt_len)

    assert isinstance(sound_ins.samples, list)
    if audio_file_len > 10:
        sample_count = audio_file_len // excerpt_len
        trim_point = int(sample_count*excerpt_len*sound_ins.sr )
        if audio_file_len % excerpt_len>=5:
            assert np.sum(sound_ins.samples[-1][trim_point:])==0
            sample_count+=1
        
        assert len(sound_ins.samples)==sample_count
        assert sound_ins.samples[0].size == excerpt_len * audio_sr
    else:
        assert len(sound_ins.samples) == 1
        assert np.sum(sound_ins.samples[0][int(audio_file_len *
                                               audio_sr):]) == 0
    assert sound_ins.samples[0].size == audio_sr * excerpt_len

    bb = np.array(sound_ins.samples)
    _ = torch.from_numpy(bb)
