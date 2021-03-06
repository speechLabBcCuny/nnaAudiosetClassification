"""Configs for the run-1.
"""
from pathlib import Path
from unittest import expectedFailure

from nna.params import EXCERPT_LENGTH

PROJECT_NAME = 'megan'
DATASET_NAME_V = 'megan_v1'

# #   DEFAULT values
# default_config = {
#     'batch_size': 58,
#     'epochs': 150,
#     'patience': -1,
#     # 'weight_decay': 0.001,
#     # 'augmentadSize': 200,
#     'CNNLayer_count': 1,
#     'CNN_filters_1': 45,
#     'CNN_kernel_size':9,
#     # 'CNN_filters_2': 64,
#     'fc_1_size':32,
#     'device': 0,
#     # augmentation params
#     # 'pitch_shift_n_steps': [3.5, 2.5, 0, -2.5, -3.5],
#     # 'time_stretch_factor': [0.8, 1.2],
#     # 'noise_factor': 0.001,
#     # 'roll_rate': 1.1,
#     # ['pitch_shift':0,'time_stretch':1,'noise_factor':2, 'roll_rate':3]
#     # 'aug_ID': 3,
#     #     'lr': lr,
#     #     'momentum': momentum,
# }

#   DEFAULT values
default_config = {
    'CNNLayer_count': 1,
    'CNN_filters_1': 2,
    'CNN_kernel_size':8,
    'device': 0,
    'batch_size': 38,
    'epochs': 150,
    'patience': -1,
    'fc_1_size':120,
}


TAXO_COUNT_LIMIT = 25
SAMPLE_LENGTH_LIMIT = 2
SAMPLING_RATE = 48000
EXP_DIR = Path('/home/enis/projects/nna/src/nna/exp/megan/run-3/')
CATEGORY_COUNT = 2
EXCERPT_LENGTH = 10
MAX_MEL_LEN = 938 # old 850

TAXONOMY_FILE_PATH = Path(
    '/home/enis/projects/nna/src/nna/assets/taxonomy/taxonomy.yaml')

DATASET_FOLDER = Path('/scratch/enis/data/nna/labeling/megan/AudioSamplesPerSite/')
MEGAN_LABELED_FILES_INFO_PATH = DATASET_FOLDER / 'meganLabeledFiles_wlenV1.txt'
AUDIO_DATA_CACHE_PATH = DATASET_FOLDER / 'files_as_np_filtered_v3_int16.pkl'
# resources_folder = ('/scratch/enis/archive/' +
#                     'forks/cramer2020icassp/resources/')
CSV4MEGAN_EXCELL_CLEANED  = (DATASET_FOLDER / 'Sheet1_my_copy_v1.csv')

FILES_AS_NP_FILTERED = DATASET_FOLDER / 'files_as_np_filtered_v1.pkl'

IGNORE_FILES = set([
    'S4A10268_20190610_103000_bio_anth.wav',  # has two topology bird/plane
])




EXCELL_NAMES2CODE = {
        'anth': '0.0.0',
        'auto': '0.1.0',
        'bio': '1.0.0',
        'bird': '1.1.0',
        'bug': '1.3.0',
        'dgs': '1.1.7',
        'flare': '0.4.0',
        'fox': '1.2.4',
        'geo': '2.0.0',
        'grouse': '1.1.8',
        'loon': '1.1.3',
        'mam': '1.2.0',
        'plane': '0.2.0',
        'ptarm': '1.1.8',
        'rain': '2.1.0',
        'seab': '1.1.5',
        'silence': '3.0.0',
        'songbird': '1.1.10',
        'unknown': 'X.X.X',
        'water': '2.2.0',
        'x': 'X.X.X',
    }
    
    