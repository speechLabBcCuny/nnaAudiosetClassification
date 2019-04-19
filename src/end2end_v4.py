from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pydub import AudioSegment

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class NatureDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,mp3_file_path,segments_folder, mp3_segments, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mp3_segments = mp3_segments
        self.segments_folder = segments_folder
        self.transform = transform
        self.input_file_path=mp3_file_path

    def __len__(self):
        return len(self.mp3_segments)

    def __getitem__(self, idx):

        wav_data = AudioSegment.from_mp3(self.input_file_path)
        sr=wav_data.frame_rate
        wav_data = wav_data.get_array_of_samples()
        wav_data = np.array(wav_data)


        sample = {'wav_data': wav_data, 'sr': sr}

        if self.transform:
            sample = self.transform(sample)

        return sample


class waveform(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __call__(self, sample):
        wav_data, sr = sample['wav_data'], sample['sr']
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        wav_data = wav_data/  32768.0  # Convert to [-1.0, +1.0]
        sound= vggish_input.waveform_to_examples(wav_data, sr)
        sound=sound.astype(np.float32)

        return {'wav_data': wav_data, 'sr': sr}
