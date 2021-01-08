from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from sklearn.model_selection import train_test_split

from argparse import ArgumentParser
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC

import wandb

import augmentations
import modelArchs
import runutils
# REPRODUCE

torch.manual_seed(42)
np.random.seed(42)


def main():
    import warnings
    warnings.filterwarnings('ignore')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--device', type=int, default=0, metavar='N',
    #                      help='GPU device ID (default: 0)')
    # args = parser.parse_args()

    #   DEFAULT values
    local_config = {
        'batch_size': 58,
        'epochs': 1000,
        'patience': -1,
        'weight_decay': 0.001,
        'augmentadSize': 200,
        'CNNLayer_count': 1,
        'CNN_filters_1': 45,
        'CNN_filters_2': 64,
        'device': 0,
        # augmentation params
        'pitch_shift_n_steps': [3.5, 2.5, 0, -2.5, -3.5],
        'time_stretch_factor': [0.8, 1.2],
        'noise_factor': 0.001,
        'roll_rate': 1.1,
        # 'aug_ID':3, # ['pitch_shift':0,'time_stretch':1,'noise_factor':2, 'roll_rate':3]
        #     'lr': lr,
        #     'momentum': momentum,
    }

    wandb_project_name = 'pytorch-ignite-integration'
    wandb.init(config=local_config, project=wandb_project_name)
    config = wandb.config
    # wandb.config.update(args) # adds all of the arguments as config variables

    params = {
        'batch_size': config.batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    tagSet = [
        'Songbird', 'Water Bird', 'Insect', 'Running Water', 'Rain', 'Cable',
        'Wind', 'Aircraft'
    ]

    labelsbyhumanpath = Path('/scratch/enis/data/nna/labeling/results/')
    # splits_path= Path('/files/scratch/enis/data/nna/labeling/splits/')
    sourcePath = Path('/scratch/enis/data/nna/labeling/splits/')

    device = torch.device(
        f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')

    # RAW DATA
    # load labels for data
    with open(labelsbyhumanpath / 'np_array_Ymatrix.npy', 'rb') as f:
        y_true = np.load(f)

        # ## load Dataset
        # # X.shape is (1300, 10, 44100)
    with open(sourcePath / 'np_array_Xmatrix_shortby441000.npy', 'rb') as f:
        X = np.load(f)
        #
        # ### split data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_true,
                                                        test_size=0.2,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      random_state=42)

    ##### AUGMENTATIONS #######

    # XArrays=[X_val,X_test,X_train]
    # X_val,X_test,X_train = runUtils.clipped_mel_loop(XArrays,850)
    #

    pitch = augmentations.pitch_shift_n_stepsClass(
        44100, config['pitch_shift_n_steps'])
    noise = augmentations.addNoiseClass(config['noise_factor'])
    strech = augmentations.time_stretchClass(441000,
                                             config['time_stretch_factor'],
                                             isRandom=True)
    shift = augmentations.shiftClass(config['roll_rate'], isRandom=True)
    toTensor = augmentations.ToTensor(850)

    transformCompose = transforms.Compose([
        pitch,
        strech,
        shift,
        noise,
        toTensor,
    ])

    sound_datasets = {
        phase: runutils.audioDataset(XY[0], XY[1], transform=transformCompose)
        for phase, XY in
        zip(['train', 'val', 'test'],
            [[X_train, y_train], [X_val, y_val], [X_test, y_test]])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(sound_datasets[x], **params)
        for x in ['train', 'val', 'test']
    }

    h_w = [128, 850]
    kernel_size = (3, 3)
    if config.CNNLayer_count == 1:
        model = modelArchs.NetCNN1(config.CNN_filters_1, h_w,
                                   kernel_size).float().to(device)

    if config.CNNLayer_count == 2:
        model = modelArchs.NetCNN2(config.CNN_filters_1, config.CNN_filters_2,
                                   h_w, kernel_size,
                                   kernel_size).float().to(device)

    # device is defined before

    model.float().to(device)  # Move model before creating optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  weight_decay=config['weight_decay'])

    criterion = nn.BCEWithLogitsLoss()
    # statHistory={'valLoss':[],'trainLoss':[],'trainAUC':[],'valAUC':[]}

    metrics = {
        'loss': Loss(criterion),  # 'accuracy': Accuracy(),
        'ROC_AUC': ROC_AUC(runutils.activated_output_transform),
    }

    print('ready ?')
    runutils.run(model, dataloaders, optimizer, criterion, metrics, device,
                 config, wandb_project_name)


if __name__ == '__main__':
    main()
