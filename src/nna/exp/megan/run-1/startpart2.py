# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%



# %%
import os

os.chdir('/home/enis/projects/nna/src/nna/exp/megan/run-1/')


# %%
# import run
# import nna
import torch
from torchvision import transforms
import torch.nn as nn

import numpy as np


# %%
import runconfigs
import wandb
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC


# %%
from nna.exp import augmentations,runutils,modelArchs


# %%
wandb.init(config=runconfigs.default_config, project=runconfigs.PROJECT_NAME)
config = wandb.config
# wandb.config.update(args) # adds all of the arguments as config variables

params = {
    'batch_size': config.batch_size,
    'shuffle': True,
    'num_workers': 0
}


# %%
device = torch.device(
    f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")


# %%
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# labelsbyhumanpath = Path('/scratch/enis/data/nna/labeling/results/')
# sourcePath = Path("/scratch/enis/data/nna/labeling/splits/")


# %%



# %%
# RAW DATA
def load_raw_data(labelsbyhumanpath,sourcePath):
    # load labels for data
    # with open(labelsbyhumanpath/"np_array_Ymatrix.npy", 'rb') as f:
    #     y_true = np.load(f)

    #     # ## load Dataset
    #     # # X.shape is (1300, 10, 44100)
    # with open(sourcePath/"np_array_Xmatrix_shortby441000.npy", 'rb') as f:
    #     X = np.load(f)
    #     #
    #     # ### split data
    sample_count = 120
    X = np.empty((sample_count,480000),dtype=np.float32)
    y_true = np.random.randint(0,10,(sample_count))
    for i,y in enumerate(y_true):
        X[i,:] = y
    n_values = np.max(y_true) + 1
    y_true = np.eye(n_values)[y_true]
    X_train, X_test, y_train, y_test = train_test_split(
                    X, y_true, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.25,random_state=42)
    return X_train,X_test,X_val,y_train,y_test,y_val

   


# %%
X_train,X_test,X_val,y_train,y_test,y_val = load_raw_data('labelsbyhumanpath','sourcePath')


# %%

pitch = augmentations.pitch_shift_n_stepsClass(
    runconfigs.SAMPLING_RATE, config['pitch_shift_n_steps'])
noise = augmentations.addNoiseClass(config['noise_factor'])
strech = augmentations.time_stretchClass(runconfigs.SAMPLING_RATE*runconfigs.EXCERPT_LENGTH,
                                            config['time_stretch_factor'],
                                            isRandom=True)
shift = augmentations.shiftClass(config['roll_rate'], isRandom=True)
maxMelLen = 850 # old 850
toTensor = augmentations.ToTensor(maxMelLen,runconfigs.SAMPLING_RATE)


# %%
# import librosa

# mel = librosa.feature.melspectrogram(y=X_train[0:1,:].reshape(-1),
#                                         sr=runconfigs.SAMPLING_RATE)
# an_x = librosa.power_to_db(mel, ref=np.max)
# # an_x = an_x.astype("float32")
# # an_x = an_x[:, :self.maxMelLen]
# # x = an_x.reshape(1, *an_x.shape[:])


# %%
# mel.shape,an_x.shape,X_train.shape


# %%

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


# %%

print('ready ?')
runutils.run(model, dataloaders, optimizer, criterion, metrics, device,config, runconfigs.PROJECT_NAME)


# %%



