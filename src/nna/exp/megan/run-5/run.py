"""Experiment running. Can only be imported from the experiment folder.

"""

from genericpath import exists
import os

# import run
# import nna
import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np

from pathlib import Path
from collections import Counter

import wandb

from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC

import runconfigs  # type: ignore
import modelarchs  # type: ignore

import nna.exp.megan as megan
from nna.exp import runutils
from ignite.contrib.handlers import wandb_logger


def prepare_dataset():

    taxo_count_limit = runconfigs.TAXO_COUNT_LIMIT
    sample_length_limit = runconfigs.SAMPLE_LENGTH_LIMIT
    taxonomy_file_path = runconfigs.TAXONOMY_FILE_PATH

    megan_labeled_files_info_path = runconfigs.MEGAN_LABELED_FILES_INFO_PATH

    csv4megan_excell_clenaed = runconfigs.CSV4MEGAN_EXCELL_CLEANED

    ignore_files = runconfigs.IGNORE_FILES

    excerpt_length = runconfigs.EXCERPT_LENGTH
    excell_names2code = runconfigs.EXCELL_NAMES2CODE
    dataset_name_v = runconfigs.DATASET_NAME_V

    audio_dataset, deleted_files = megan.preparedataset.run(  # type: ignore
        megan_labeled_files_info_path,
        taxonomy_file_path,
        csv4megan_excell_clenaed,
        ignore_files,
        excerpt_length,
        sample_length_limit,
        taxo_count_limit,
        excell_names2code=excell_names2code,
        dataset_name_v=dataset_name_v)

    audio_dataset.load_audio_files(runconfigs.AUDIO_DATA_CACHE_PATH)
    audio_dataset.pick_channel_by_clipping()

    return audio_dataset, deleted_files


def setup_config(config, wandb_project_name):

    # wandb.init(config=runconfigs.default_config, project=runconfigs.PROJECT_NAME)
    # config = wandb.config
    config = runconfigs.default_config
    # wandb.config.update(args) # adds all of the arguments as config variables
    # config['batch_size'] = 64

    os.chdir(runconfigs.EXP_DIR)

    device = torch.device(f"cuda:{config['device']}" if
                          torch.cuda.is_available() else "cpu")  # type: ignore

    config['device'] = device

    random_seed: int = 42
    # stable results
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False  # type: ignore

    # wandb.init(config=config, project=wandb_project_name) # type: ignore
    # config = wandb.config # type: ignore

    return config


def dataset_generate_samples(audio_dataset, excerpt_len):
    '''divida into chunks by expected_len seconds.
        Repeats data if smaller than expected_len.

    TODO move this function to audio_dataset's class
    '''
    for sound_ins in audio_dataset.values():
        data_to_samples(sound_ins, excerpt_len)
    return audio_dataset


def data_to_samples(sound_ins, excerpt_len):
    excerpt_sample_size = excerpt_len * sound_ins.sr

    data_len_sec = sound_ins.length
    min_expected_element_count = excerpt_len * sound_ins.sr
    if data_len_sec < excerpt_len:
        missing_element_count = int(min_expected_element_count -
                                    sound_ins.data.size)
        padded_data = np.pad(sound_ins.data, (0, missing_element_count),
                             'constant',
                             constant_values=(0, 0))
        sound_ins.samples = [padded_data]
    elif data_len_sec == 10:
        sound_ins.samples = [sound_ins.data]
    else:
        excerpt_count = data_len_sec // excerpt_len
        # first process samples that are at least excerpt_len long.
        data_trim_point = int(excerpt_count * excerpt_len * sound_ins.sr)
        samples = sound_ins.data[:data_trim_point].reshape(
            -1, int(excerpt_sample_size))
        sample_list = []
        for sample in samples:
            sample_list.append(sample)
        # if left over is >= 5 seconds, pad with zeros to make a new sample
        left_over_len = data_len_sec % excerpt_len
        if left_over_len >= 5:
            missing_element_count = int(min_expected_element_count -
                                        (sound_ins.data.size - data_trim_point))
            padded_data = np.pad(sound_ins.data[data_trim_point:],
                                 (0, missing_element_count),
                                 'constant',
                                 constant_values=(0, 0))
            sample_list.append(padded_data)

        sound_ins.samples = sample_list


def put_samples_into_array(target_taxo, other_taxo, audio_dataset):
    # sound_ins[1].taxo_code
    # classA = 1.1.7 #'duck-goose-swan'
    # classB = 0.2.0 # other-aircraft
    # 3.0.0 : 0.48, 0.26, 0.26, 46 # silence
    # 2.1.0 : 0.22, 0.56, 0.22, 18 # rain
    # 1.3.0 1.3.0 : 0.52, 0.4, 0.087, 161 # insect
    # 1.1.8 : 0.49, 0.19, 0.32, 88 # grouse-ptarmigan

    x_data = []
    y = []
    location_id_info = []

    for sound_ins in audio_dataset.values():
        if sound_ins.taxo_code in target_taxo + other_taxo:
            for sample in sound_ins.samples:
                y.append(sound_ins.taxo_code)
                location_id_info.append(sound_ins.location_id)
                x_data.append(sample)

    return x_data, y, location_id_info


def find_upper_taxo(taxo):

    if '.' in taxo:
        taxo_a = taxo.split('.')
    else:
        taxo_a = taxo[:]

    if set(taxo_a) == set('X'):
        return '.'.join(taxo_a)
    if 'X' in taxo_a:
        taxo_a = [x if x != 'X' else '0' for x in taxo_a]
    # -1 because we do not change first bit
    for i in range(len(taxo_a) - 1):
        if taxo_a[-(i + 1)] == '0':
            continue
        else:
            taxo_a[-(i + 1)] = '0'
            break

    taxo_a = '.'.join(taxo_a)
    return taxo_a


def get_root_taxos(org_taxo):
    root_taxos = []
    upper_taxo = find_upper_taxo(org_taxo)
    previous_taxo = org_taxo
    while upper_taxo != previous_taxo:
        root_taxos.append(upper_taxo)
        previous_taxo = upper_taxo
        upper_taxo = find_upper_taxo(previous_taxo)
    return root_taxos


assert ['1.1.0', '1.0.0'] == get_root_taxos('1.1.1')


def create_multi_label_vector(alphabet, y_data):
    # define input string
    # define universe of possible input values
    # alphabet = ['1.1.10','1.1.7']

    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    integer_encoded = []
    for point in y_data:
        root_taxos = get_root_taxos(point)

        root_taxos.append(point)
        #         print(root_taxos)
        int_values = [char_to_int.get(taxo, None) for taxo in root_taxos]
        int_values = [x for x in int_values if x is not None]
        integer_encoded.append(int_values)

    onehot_encoded = list()
    #     print(integer_encoded)
    for values in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        for value in values:
            #             print(value)
            letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


def create_one_hot_vector(alphabet, y_data):
    # define input string
    # define universe of possible input values
    # alphabet = ['1.1.10','1.1.7']

    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int.get(char, None) for char in y_data]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        if value is not None:
            letter[value] = 1
        onehot_encoded.append(letter)
    # print(onehot_encoded)
    # invert encoding
    inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)
    onehot_encoded = np.array(onehot_encoded)

    return onehot_encoded


def split_train_test_val(x_data, location_id_info, onehot_encoded, loc_per_set):
    X_train, X_test, X_val, y_train, y_test, y_val = [], [], [], [], [], []
    loc_id_train = []
    loc_id_test = []
    loc_id_valid = []

    for sample, y_val_ins, loc_id in zip(x_data, onehot_encoded,
                                         location_id_info):
        if loc_id in loc_per_set[0]:
            X_train.append(sample)
            y_train.append(y_val_ins)
            loc_id_train.append(loc_id)
        elif loc_id in loc_per_set[1]:
            X_test.append(sample)
            y_test.append(y_val_ins)
            loc_id_test.append(loc_id)
        elif loc_id in loc_per_set[2]:
            X_val.append(sample)
            y_val.append(y_val_ins)
            loc_id_valid.append(loc_id)
        else:
            print('error')

    X_train, X_test, X_val = np.array(X_train), np.array(X_test), np.array(
        X_val)
    y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(
        y_val)

    X_train, X_test, X_val = torch.from_numpy(X_train).float(
    ), torch.from_numpy(X_test).float(), torch.from_numpy(X_val).float()
    y_train, y_test, y_val = torch.from_numpy(y_train).float(
    ), torch.from_numpy(y_test).float(), torch.from_numpy(y_val).float()

    return X_train, X_test, X_val, y_train, y_test, y_val


def prepare_run_inputs(config, X_train, X_test, X_val, y_train, y_test, y_val):

    # toTensor = augmentations.ToTensor(maxMelLen,runconfigs.SAMPLING_RATE)
    to_tensor = modelarchs.ToTensor(runconfigs.MAX_MEL_LEN,
                                    runconfigs.SAMPLING_RATE)

    transformCompose = transforms.Compose([
        to_tensor,
    ])

    sound_datasets = {
        phase: runutils.audioDataset(XY[0], XY[1], transform=transformCompose)
        for phase, XY in
        zip(['train', 'val', 'test'],
            [[X_train, y_train], [X_val, y_val], [X_test, y_test]])
    }

    data_loader_params = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': 0
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(sound_datasets[x], **data_loader_params)
        for x in ['train', 'val', 'test']
    }

    # this will change
    h_w = [128, 938]
    # kernel_size = (3, 3)
    output_shape = (runconfigs.CATEGORY_COUNT,)
    model = modelarchs.singleconv1dModel(out_channels=config['CNN_filters_1'],
                                         h_w=(1, h_w[0] * h_w[1]),
                                         fc_1_size=config['fc_1_size'],
                                         kernel_size=config['CNN_kernel_size'],
                                         output_shape=output_shape)

    # device is defined before

    model.float().to(config['device'])  # Move model before creating optimizer
    optimizer = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        #                                 weight_decay=config['weight_decay'],
    )

    criterion = nn.BCEWithLogitsLoss()
    # statHistory={'valLoss':[],'trainLoss':[],'trainAUC':[],'valAUC':[]}

    metrics = {
        'loss':
            Loss(criterion),  # 'accuracy': Accuracy(),
        #     'ROC_AUC': ROC_AUC(runutils.activated_output_transform),
        'ROC_AUC':
            megan.metrics.ROC_AUC_perClass(  # type: ignore
                megan.metrics.activated_output_transform),  # type: ignore
    }

    return model, optimizer, dataloaders, metrics, criterion


def run_exp(wandb_logger_ins):

    config = wandb_logger_ins.config
    loc_per_set = [[
        '12', '14', '27', '49', '31', '39', '44', '45', '48', '19', '16', '21',
        '38', '41', '20', '29', '37', '15'
    ], ['17', '46', '50', '32', '33', '25', '40'],
                   ['11', '18', '34', '24', '13', '22', '36', '47', '30']]

    target_taxo = [
        '1.0.0', '1.1.0', '1.1.10', '1.1.7', '0.0.0', '1.3.0', '1.1.8', '0.2.0',
        '3.0.0'
    ]
    # 0.0.0 anthrophony
    # 0.2.0 aircraft
    # 1.0.0 biophony
    # 1.1.0 bird
    # 1.1.10 songbirds
    # 1.1.7 duck-goose-swan
    # 1.3.0 insect
    # 1.1.8 grouse-ptarmigan
    # 3.0.0 silence



    audio_dataset, _ = prepare_dataset()
    audio_dataset = dataset_generate_samples(audio_dataset,
                                             runconfigs.EXCERPT_LENGTH)

    other_taxo = set()
    for sound_ins in audio_dataset.values():
        if sound_ins.taxo_code not in target_taxo:
            other_taxo.add(sound_ins.taxo_code)
    other_taxo = list(other_taxo)

    x_data, y_data, location_id_info = put_samples_into_array(
        target_taxo, other_taxo, audio_dataset)

    #     onehot_encoded = create_one_hot_vector(target_taxo, y_data)
    multi_label_vector = create_multi_label_vector(target_taxo, y_data)

    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val(
        x_data, location_id_info, multi_label_vector, loc_per_set)

    model, optimizer, dataloaders, metrics, criterion = prepare_run_inputs(
        config, X_train, X_test, X_val, y_train, y_test, y_val)

    print('ready ?')
    checkpoints_dir = runconfigs.EXP_DIR / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    runutils.run(model,
                 dataloaders,
                 optimizer,
                 criterion,
                 metrics,
                 config['device'],
                 config,
                 runconfigs.PROJECT_NAME,
                 checkpoints_dir=checkpoints_dir,
                 wandb_logger_ins=wandb_logger_ins)


def main():
    wandb_project_name = runconfigs.PROJECT_NAME
    default_config = runconfigs.default_config
    config = setup_config(default_config, wandb_project_name)

    wandb_logger_ins = wandb_logger.WandBLogger(
        project=wandb_project_name,
        # name=runconfigs.PROJECT_NAME,
        config=config,
    )

    run_exp(wandb_logger_ins)
    # return audio_dataset, config


if __name__ == '__main__':
    # do not upload model files
    main()
