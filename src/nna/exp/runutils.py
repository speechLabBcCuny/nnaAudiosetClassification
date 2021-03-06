import numpy as np
import csv
from os import listdir

from ignite.contrib.handlers import wandb_logger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Accuracy, Loss
# from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.utils import setup_logger

# import wandb

import torch
from torch.utils.data import Dataset

# import librosa

# from nna.exp.metrics import ROC_AUC_perClass


def loadLabels(labelsbyhumanpath):
    # filter csv extension also by username
    labelsbyhuman = [i for i in listdir(labelsbyhumanpath) if '.csv' in i]
    humanresults = {}
    counter = 0
    for apath in labelsbyhuman:
        with open(labelsbyhumanpath / apath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                counter += 1
                humanresults[row[0]] = row[1:]
    # print('unique files:',len(humanresults),'\ntotal files',counter)
    # Join vehicle and Aircraft
    for file_name, tagshere in humanresults.items():

        humanresults[file_name] = list(
            set(['Aircraft' if tag == 'Vehicle' else tag for tag in tagshere]))

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


def run(model,
        dataloaders,
        optimizer,
        criterion,
        metrics,
        device,
        config,
        wandb_project_name,
        run_name=None,
        checkpoints_dir=None,
        wandb_logger_ins=None,
        taxo_names=None):

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        criterion,
                                        device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model,
                                                  metrics=metrics,
                                                  device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model,
                                                       metrics=metrics,
                                                       device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    # best_ROC_AUC -> [mean,min]
    best_ROC_AUC = [0,0]
    best_epoch = [0,0]

    @trainer.on(Events.EPOCH_COMPLETED, best_ROC_AUC, best_epoch,taxo_names)
    def compute_metrics(engine, best_ROC_AUC, best_epoch,taxo_names=None):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)
        roc_auc_array_val = validation_evaluator.state.metrics['ROC_AUC']
        roc_auc_array_train = train_evaluator.state.metrics['ROC_AUC']

        print('train loss', train_evaluator.state.metrics['loss'])
        print('val loss', validation_evaluator.state.metrics['loss'])
        print('validation roc auc', roc_auc_array_val, engine.state.epoch)
        print('train roc auc', roc_auc_array_train, engine.state.epoch)

        current_train_roc_auc_mean = np.mean(roc_auc_array_train)
        current_train_roc_auc_mean = current_train_roc_auc_mean.item()

        current_val_roc_auc_mean = np.mean(roc_auc_array_val)
        current_val_roc_auc_mean = current_val_roc_auc_mean.item()

        current_val_roc_auc_min = np.min(roc_auc_array_val)
        current_val_roc_auc_min = current_val_roc_auc_min.item()

        if current_val_roc_auc_mean > best_ROC_AUC[0]:
            best_ROC_AUC[0] = current_val_roc_auc_mean
            best_epoch[0] = engine.state.epoch
            wandb_logger_ins.log(
                {
                    'best_mean_ROC_AUC': best_ROC_AUC[0],
                    'best_mean_Epoch': best_epoch[0]
                },
                step=trainer.state.iteration)
    
        if current_val_roc_auc_min > best_ROC_AUC[1]:
            best_ROC_AUC[1] = current_val_roc_auc_min
            best_epoch[1] = engine.state.epoch
            wandb_logger_ins.log(
                {
                    'best_min_ROC_AUC': best_ROC_AUC[1],
                    'best_min_Epoch': best_epoch[1]
                },
                step=trainer.state.iteration)
        #log epochs seperatly to use in X axis 
        wandb_logger_ins.log({'epoch': engine.state.epoch},
                             step=trainer.state.iteration)
        wandb_logger_ins.log({'val_roc_auc_mean': current_val_roc_auc_mean},
                             step=trainer.state.iteration)
        wandb_logger_ins.log({'train_roc_auc_mean': current_train_roc_auc_mean},
                             step=trainer.state.iteration)

        if taxo_names is None:
            taxo_names = [f'class_{k}' for k in range(len(roc_auc_array_val))]

        for i,taxo_name in enumerate(taxo_names): # type: ignore
            wandb_logger_ins.log({f'val_roc_auc_{taxo_name}': roc_auc_array_val[i]},
                             step=trainer.state.iteration)

    
    if wandb_logger_ins is None:
        wandb_logger_ins = wandb_logger.WandBLogger(
            project=wandb_project_name,
            # name=run_name,
            config=config,
        )

    wandb_logger_ins.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,  # could add (every=100),
        tag='training',  # type: ignore
        output_transform=lambda loss: {'batchloss': loss}  # type: ignore
    )

    for tag, evaluator in [('training', train_evaluator),
                           ('validation', validation_evaluator)]:
        # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
        # We setup `global_step_transform=lambda *_: trainer.state.iteration` to take iteration value
        # of the `trainer`:
        wandb_logger_ins.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=['loss', 'ROC_AUC'],  # type: ignore
            global_step_transform=lambda *_: trainer.state.
            iteration,  # type: ignore
        )

    wandb_logger_ins.attach_opt_params_handler(
        trainer, event_name=Events.EPOCH_COMPLETED, optimizer=optimizer)
    wandb_logger_ins.watch(model, log='all')

    def score_function(engine):
        return np.mean(engine.state.metrics['ROC_AUC']).item()

    def score_function2(engine):
        print('loss', engine.state.metrics['loss'])
        return engine.state.metrics['loss']
    
    def score_funtion_min(engine):
        return np.min(engine.state.metrics['ROC_AUC']).item()


    if checkpoints_dir is None:
        checkpoint_dir = wandb_logger_ins.run.dir
    else:
        checkpoint_dir = checkpoints_dir / wandb_logger_ins.run.name

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=2,
        filename_prefix='best',
        score_function=score_funtion_min,
        score_name='min_ROC_AUC',
        # to take the epoch of the `trainer`L
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,
                                           {'model': model})

    if config['patience'] > 1:
        es_handler = EarlyStopping(patience=config['patience'],
                                   score_function=score_function2,
                                   trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, es_handler)

#     kick everything off
    trainer.run(train_loader, max_epochs=config['epochs'])

    wandb_logger_ins.close()


# def clipped_mel_loop(XArrays, maxMelLen):
#     '''
#   maxMelLen: will clip arrays after that threshold, 850 for randomAdd
#   because it does not change original size of 10second audio files

#   ! samping rate is hard coded
#   '''
#     results = []
#     for X_array_i in range(len(XArrays)):
#         X_array = XArrays[X_array_i]
#         for index, y in enumerate(X_array):

#             mel = librosa.feature.melspectrogram(y=y.reshape(-1), sr=44100)
#             an_x = librosa.power_to_db(mel, ref=np.max)
#             an_x = an_x.astype('float32')
#             if index == 0:
#                 XMel = np.empty((X_array.shape[0], 128, maxMelLen),
#                                 dtype=np.float32)
#             XMel[index, :, :] = an_x[:, :maxMelLen]
#             # if index%100==0:
#             #     print(index)
#     #     X_array = XMel[:]
#         results.append(XMel)
#     #     print(X_array.shape)
#     return results


class audioDataset(Dataset):

    def __init__(self, X, y=None, transform=None):
        '''
    Args:

    '''
        self.X = X
        self.y = y
        #         self.landmarks_frame = pd.read_csv(csv_file)
        #         self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            sample =  self.X[idx], torch.zeros((2))
        else:
            sample = self.X[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
