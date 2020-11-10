import numpy as np
import csv
from os import listdir

from ignite.contrib.handlers.wandb_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.utils import setup_logger

import wandb

import torch
from torch.utils.data import Dataset

import librosa

from metrics import ROC_AUC_perClass


def loadLabels(labelsbyhumanpath):
    # filter csv extension also by username
    labelsbyhuman = [i for i in listdir(labelsbyhumanpath) if (".csv" in i)]
    humanresults = {}
    counter = 0
    for apath in labelsbyhuman:
        with open(labelsbyhumanpath / apath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                counter += 1
                humanresults[row[0]] = row[1:]
    # print("unique files:",len(humanresults),"\ntotal files",counter)
    # Join vehicle and Aircraft
    for file_name, tagshere in humanresults.items():

        humanresults[file_name] = list(
            set(["Aircraft" if tag == "Vehicle" else tag for tag in tagshere]))

    return humanresults


# returns a dictionary, keys are tags from tag set and values are binary list
#
def vectorized_y_true(humanresults, tag_set):
    y_true = {tag: np.zeros(len(humanresults)) for tag in tag_set}
    for i, tags in enumerate(humanresults.values()):
        # we  only look for tags in tag_set
        for tag in tag_set:
            if tag in tags:
                y_true[tag][i] = 1
            else:
                y_true[tag][i] = 0
    return y_true


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    return y_pred, y


def run(model, dataloaders, optimizer, criterion, metrics, device, config,
        wandb_project_name):

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        criterion,
                                        device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model,
                                                  metrics=metrics,
                                                  device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")
    validation_evaluator = create_supervised_evaluator(model,
                                                       metrics=metrics,
                                                       device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    best_ROC_AUC = [0]
    best_epoch = [0]

    @trainer.on(Events.EPOCH_COMPLETED, best_ROC_AUC, best_epoch)
    def compute_metrics(engine, best_ROC_AUC, best_epoch):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

        if validation_evaluator.state.metrics["ROC_AUC"] > best_ROC_AUC[0]:
            best_ROC_AUC[0] = validation_evaluator.state.metrics["ROC_AUC"]
            best_epoch[0] = engine.state.epoch
            wandb.log({
                'best_ROC_AUC': best_ROC_AUC[0],
                'best_Epoch': best_epoch[0]
            })
        wandb.log({'epoch': engine.state.epoch})

    wandb_logger = WandBLogger(
        project=wandb_project_name,
        name="ignite-mnist-example",
        config=config,
    )

    # wandb_logger.attach_output_handler(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED(every=100),
    #     tag="training",
    #     output_transform=lambda loss: {"batchloss": loss},
    # )

    for tag, evaluator in [("training", train_evaluator),
                           ("validation", validation_evaluator)]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=[
                "loss",  # "accuracy"
                "ROC_AUC",
            ],
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    wandb_logger.attach_opt_params_handler(trainer,
                                           event_name=Events.EPOCH_COMPLETED,
                                           optimizer=optimizer)
    wandb_logger.watch(model, log="all")

    def score_function(engine):
        return engine.state.metrics["ROC_AUC"]

    def score_function2(engine):
        return engine.state.metrics["loss"]

    model_checkpoint = ModelCheckpoint(
        wandb_logger.run.dir,
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="ROC_AUC",
        # `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the `trainer`
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,
                                           {"model": model})

    if config["patience"] > 1:
        es_handler = EarlyStopping(patience=config["patience"],
                                   score_function=score_function2,
                                   trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, es_handler)


#     kick everything off
    trainer.run(train_loader, max_epochs=config["epochs"])

    wandb_logger.close()


def clipped_mel_loop(XArrays, maxMelLen):
    """
  maxMelLen: will clip arrays after that threshold, 850 for randomAdd
  because it does not change original size of 10second audio files

  ! samping rate is hard coded
  """
    results = []
    for X_array_i in range(len(XArrays)):
        X_array = XArrays[X_array_i]
        for index, y in enumerate(X_array):

            mel = librosa.feature.melspectrogram(y=y.reshape(-1), sr=44100)
            an_x = librosa.power_to_db(mel, ref=np.max)
            an_x = an_x.astype("float32")
            if index == 0:
                XMel = np.empty((X_array.shape[0], 128, maxMelLen),
                                dtype=np.float32)
            XMel[index, :, :] = an_x[:, :maxMelLen]
            # if index%100==0:
            #     print(index)
    #     X_array = XMel[:]
        results.append(XMel)
    #     print(X_array.shape)
    return results


class audioDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
    Args:

    """
        self.X = X
        self.y = y
        #         self.landmarks_frame = pd.read_csv(csv_file)
        #         self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        sample = self.X[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
