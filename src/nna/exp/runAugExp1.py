from pathlib import Path
import pickle
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import librosa

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
    warnings.filterwarnings("ignore")

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--device', type=int, default=0, metavar="N",
    #                      help='GPU device ID (default: 0)')
    # args = parser.parse_args()

    #   DEFAULT values
    local_config = {
        "batch_size": 58,
        "epochs": 1000,
        "patience": -1,
        "weight_decay": 0.001,
        "augmentadSize": 200,
        "CNNLayer_count": 1,
        "CNN_filters_1": 45,
        "CNN_filters_2": 64,
        "device": 0,
        # augmentation params
        "pitch_shift_n_steps": 2.2,
        "time_stretch_factor": 0.8,
        "noise_factor": 0.001,
        "roll_rate": 1.1,
        # ["pitch_shift":0,"time_stretch":1,"noise_factor":2, "roll_rate":3]
        "aug_ID": 3,
        #     "lr": lr,
        #     "momentum": momentum,
    }

    wandb_project_name = "pytorch-ignite-integration"
    wandb.init(config=local_config, project=wandb_project_name)
    config = wandb.config
    # wandb.config.update(args) # adds all of the arguments as config variables

    params = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": 0
    }

    tagSet = [
        "Songbird", "Water Bird", "Insect", "Running Water", "Rain", "Cable",
        "Wind", "Aircraft"
    ]

    labelsbyhumanpath = Path("/scratch/enis/data/nna/labeling/results/")
    # splits_path= Path('/files/scratch/enis/data/nna/labeling/splits/')
    sourcePath = Path("/scratch/enis/data/nna/labeling/splits/")

    device = torch.device(
        f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")

    # RAW DATA
    # load labels for data
    with open(labelsbyhumanpath / "np_array_Ymatrix.npy", "rb") as f:
        y_true = np.load(f)

        # ## load Dataset
        # # X.shape is (1300, 10, 44100)
    with open(sourcePath / "np_array_Xmatrix_shortby441000.npy", "rb") as f:
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
import torch
import time
b=[]
m=100
while True:
    try:
        sh=(250000,m)
        bb=torch.rand(sh).to('cuda:1')
        b.append(bb)
    except:
        m-=1
        print(bb.shape)
        time.sleep(60)
    ##### AUGMENTATIONS #######
    def augmentData(data, sr, config):

        if config["aug_ID"] == 0:
            dataAug = augmentations.pitch_shift(data, sr,
                                                config["pitch_shift_n_steps"])
        elif config["aug_ID"] == 1:
            output_length = 44100 * 10
            dataAug = augmentations.time_stretch(data, output_length,
                                                 config["time_stretch_factor"])
        elif config["aug_ID"] == 2:
            dataAug = augmentations.addNoise(data, config["noise_factor"])
        elif config["aug_ID"] == 3:
            # Shifting the sound
            shift = int(np.prod(data.shape[1:]) * config["roll_rate"])
            dataAug = np.roll(data.reshape(data.shape[0], -1), shift, axis=1)
            dataAug = dataAug.reshape(dataAug.shape[0], -1, sr)
        else:
            print("Augmentation not found")
            return None
        return dataAug

    # ###### randomADD
    # X_trainAugmented,y_trainAugmented = augmentations.randomAdd(X_train,y_train,config["augmentadSize"],unique=False)
    # X_train = X_trainAugmented
    # y_train = y_trainAugmented
    # # to keep original ones in the same shape
    # # increase size of original ones by merging them with themselves or we could just use augmentad ones
    #
    # ########### calculate mel-spectogram for all data
    #
    print("augmentation start")
    res = []
    for i in range(3):
        local_config["aug_ID"] = i
        X_trainAug = augmentData(X_train, 44100, local_config)
        res.append(X_trainAug)
    X_train = np.concatenate([
        X_train,
    ] + res)
    y_train = np.concatenate([y_train for i in range(4)])

    XArrays = [X_val, X_test, X_train]
    X_val, X_test, X_train = runutils.clipped_mel_loop(XArrays, 850)

    # # add channel dimension and turn data into float32
    XArrays = [X_train, X_test, X_val]
    for i, X_array in enumerate(XArrays):
        XArrays[i] = X_array.reshape(X_array.shape[0], 1, *X_array.shape[1:])

    [X_train, X_test, X_val] = XArrays

    # AUGMENTED ONES
    # with open(sourcePath/"np_array_Xmatrix_addAug_mel.npy", "rb") as f:
    #     XArrays = pickle.load(f)
    #     [X_train,X_test,X_val]=XArrays
    # with open(sourcePath/"np_array_Ymatrix_seperate.npy", "rb") as f:
    #     yArrays = pickle.load(f)
    #     [y_train,y_test,y_val]=yArrays

    # dataloaders
    sound_datasets = {
        phase: runutils.audioDataset(XY[0], XY[1])
        for phase, XY in
        zip(["train", "val", "test"],
            [[X_train, y_train], [X_val, y_val], [X_test, y_test]])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(sound_datasets[x], **params)
        for x in ["train", "val", "test"]
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
                                  weight_decay=config["weight_decay"])

    criterion = nn.BCEWithLogitsLoss()
    # statHistory={"valLoss":[],"trainLoss":[],"trainAUC":[],"valAUC":[]}

    metrics = {
        "loss": Loss(criterion),  # "accuracy": Accuracy(),
        "ROC_AUC": ROC_AUC(runutils.activated_output_transform),
    }

    print("ready ?")
    runutils.run(model, dataloaders, optimizer, criterion, metrics, device,
                 config, wandb_project_name)


if __name__ == "__main__":
    main()
