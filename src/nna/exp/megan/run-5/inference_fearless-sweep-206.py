# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchaudio
torchaudio.set_audio_backend('sox_io')

import numpy as np
import pandas as pd

from nna.exp import runutils

# %%
import os

# %%

from nna import dataimport
from nna import fileUtils
from nna.exp import runutils

import modelarchs  # type: ignore

# %%


def repeat_data(data, expected_len) -> np.ndarray:
    '''pad by zeros if it is not divisible expected_len seconds.
    '''
    sr = 48000
    left_over = (data.shape[0]) % (expected_len * sr)

    if left_over != 0:
        missing_element_count = (expected_len * sr) - left_over
        padded_data = np.pad(data[-left_over:], (0, missing_element_count),
                             'constant',
                             constant_values=(0, 0))
        return np.concatenate([data[:-left_over], padded_data])  # type: ignore
    else:
        return data


# %%
class pathMap():

    def __init__(self) -> None:
        scratch = '/scratch/enis/data/nna/'
        home = '/home/enis/projects/nna/'
        self.exp_dir = '/home/enis/projects/nna/src/nna/exp/megan/run-5/'
        self.clipping_results_path = (scratch +
                                      'clipping_info/all-merged_2021-02-10/')
        self.output_dir = scratch + 'real/'
        self.file_properties_df_path = scratch + 'database/allFields_dataV5.pkl'
        # model_path= ('/home/enis/projects/nna/src/nna/exp/megan/run-3/'+
        # 'checkpoints_keep/glorious-sweep-57/best_model_56_ROC_AUC=0.8690.pt')
        checkpoints_dir = scratch + 'runs_models/megan/run-5/checkpoints/'
        self.model_path = (checkpoints_dir +
                           'fearless-sweep-206/best_model_64_min_ROC_AUC=0.6955.pt')


def setup_inputs(args):
    index, count = int(args.index), int(args.count)

    region_location = [['anwr', '31'], ['anwr', '32'], ['anwr', '33'],
                       ['anwr', '34'], ['anwr', '35'], ['anwr', '36'],
                       ['anwr', '37'], ['anwr', '38'], ['anwr', '39'],
                       ['anwr', '40'], ['anwr', '41'], ['anwr', '42'],
                       ['anwr', '43'], ['anwr', '44'], ['anwr', '45'],
                       ['anwr', '46'], ['anwr', '47'], ['anwr', '48'],
                       ['anwr', '49'], ['anwr', '50'], ['dalton', '01'],
                       ['dalton', '02'], ['dalton', '03'], ['dalton', '04'],
                       ['dalton', '05'], ['dalton', '06'], ['dalton', '07'],
                       ['dalton', '08'], ['dalton', '09'], ['dalton', '10'],
                       ['dempster', '11'], ['dempster',
                                            '12'], ['dempster', '13'],
                       ['dempster', '14'], ['dempster',
                                            '16'], ['dempster', '17'],
                       ['dempster', '19'], ['dempster',
                                            '20'], ['dempster', '21'],
                       ['dempster', '22'], ['dempster', '23'],
                       ['dempster', '24'], ['dempster', '25'],
                       ['ivvavik', 'AR01'], ['ivvavik', 'AR02'],
                       ['ivvavik', 'AR03'], ['ivvavik', 'AR04'],
                       ['ivvavik', 'AR05'], ['ivvavik', 'AR06'],
                       ['ivvavik', 'AR07'], ['ivvavik', 'AR08'],
                       ['ivvavik', 'AR09'], ['ivvavik', 'AR10'],
                       ['ivvavik', 'SINP01'], ['ivvavik', 'SINP02'],
                       ['ivvavik', 'SINP03'], ['ivvavik', 'SINP04'],
                       ['ivvavik', 'SINP05'], ['ivvavik', 'SINP06'],
                       ['ivvavik', 'SINP07'], ['ivvavik', 'SINP08'],
                       ['ivvavik', 'SINP09'], ['ivvavik', 'SINP10'],
                       ['prudhoe', '11'], ['prudhoe', '12'], ['prudhoe', '13'],
                       ['prudhoe', '14'], ['prudhoe', '15'], ['prudhoe', '16'],
                       ['prudhoe', '17'], ['prudhoe', '18'], ['prudhoe', '19'],
                       ['prudhoe', '20'], ['prudhoe', '21'], ['prudhoe', '22'],
                       ['prudhoe', '23'], ['prudhoe', '24'], ['prudhoe', '25'],
                       ['prudhoe', '26'], ['prudhoe', '27'], ['prudhoe', '28'],
                       ['prudhoe', '29'], ['prudhoe', '30']]

    return region_location[index:index + count]


def setup(args):

    pathmap = pathMap()

    os.chdir(pathmap.exp_dir)

    file_properties_df = pd.read_pickle(pathmap.file_properties_df_path)

    device = 'cuda:' + str(args.gpu)
    device = torch.device(device)

    CATEGORY_COUNT = 9
    # '1.1.10','1.1.7'
    maxMelLen = 938
    ToTensor_ins = modelarchs.ToTensor(maxMelLen, 48000, device)
    transformCompose = transforms.Compose([
        ToTensor_ins,
    ])

    h_w = [128, 938]

    config = {}
    config['label_names'] = [
        '1-0-0',
        '1-1-0',
        '1-1-10',
        '1-1-7',
        '0-0-0',
        '1-3-0',
        '1-1-8',
        '0-2-0',
        '3-0-0',
    ]
    config['v_str'] = 'multi9-V1-fearless-206'
    config['CNN_filters_1'] = 5
    config['CNN_kernel_size'] = 12
    config['fc_1_size'] = 115

    config['expected_len'] = 10
    config['device'] = device

    output_shape = (CATEGORY_COUNT,)

    model_saved = modelarchs.singleconv1dModel(
        out_channels=config['CNN_filters_1'],
        h_w=(1, h_w[0] * h_w[1]),
        fc_1_size=config['fc_1_size'],
        kernel_size=config['CNN_kernel_size'],
        output_shape=output_shape)

    model_saved.load_state_dict(
        torch.load(pathmap.model_path, map_location=config['device']))
    model_saved.eval().to(config['device'])

    return model_saved, transformCompose, config, file_properties_df, pathmap


def load_audio_files(region_location, file_properties_df):
    '''Load audio files from given regions and locations as Audio dataset
    '''
    region_location_datasets = {}
    for region, location in region_location:
        filtered_files = file_properties_df[file_properties_df.region == region]
        filtered_files = filtered_files[filtered_files.locationId == location]
        filtered_files = filtered_files[filtered_files.durationSec > 0]
        dataset_name_v = '-'.join([region, location])
        audio_dataset = dataimport.Dataset(dataset_name_v=dataset_name_v)
        for i in filtered_files.iterrows():
            audio_dataset[i[0]] = dataimport.Audio(i[1].name,
                                                   float(i[1].durationSec))
        region_location_datasets[(region, location)] = audio_dataset
    return region_location_datasets


# %%
def prepare_dataloader_from_audio_ins(audio_ins, config, transformCompose):
    audio_ins.load_data()
    audio_ins.pick_channel_by_clipping(config['expected_len'])
    input_file_data = repeat_data(audio_ins.data, config['expected_len'])

    # divide to 10 second excerpts
    input_file_data = input_file_data.reshape(-1, 480000)
    input_file_data = torch.from_numpy(input_file_data).float()
    dataset = {
        'predict':
            runutils.audioDataset(input_file_data,
                                  None,
                                  transform=transformCompose)
    }
    dataloader = {
        'predict':
            torch.utils.data.DataLoader(dataset['predict'],
                                        shuffle=False,
                                        batch_size=128)
    }
    return dataloader


def single_file_inference(dataloader, config, model_saved):
    outputs = []
    for inputs, labels in dataloader['predict']:
        del labels
        inputs = inputs.float().to(config['device'])
        output = model_saved(inputs)
        output = output.to('cpu')
        index = output.data.numpy()
        outputs.append(index)
    outputs = np.concatenate(outputs)
    return outputs


def save_results_disk(outputs, audio_ins, label_names, v_str,
                      file_properties_df, pathmap):
    # label_names = ['1-1-10', '1-1-7']
    # v_str = 'V3'
    file_names = output_file_names(audio_ins, label_names, v_str,
                                   file_properties_df, pathmap)
    for i, file_name in enumerate(file_names):
        file_name.parent.mkdir(parents=True, exist_ok=True)
        file_name = file_name.with_suffix('.npy')
        np.save(str(file_name), outputs[:, i])
    audio_ins.data = None


def output_file_names(audio_ins, label_names, v_str, file_properties_df,
                      pathmap):
    row = file_properties_df.loc[audio_ins.path]
    file_names = []
    for _, label_name in enumerate(label_names):
        sub_directory_addon = v_str + '-' + label_name
        file_name_addon = sub_directory_addon
        file_name = fileUtils.standard_path_style(
            pathmap.output_dir,
            row,
            sub_directory_addon=sub_directory_addon,
            file_name_addon=file_name_addon)
        #             print(file_name)
        file_names.append(file_name)
    return file_names


def is_result_exist(audio_ins, label_names, v_str, file_properties_df, pathmap):
    file_names = output_file_names(audio_ins, label_names, v_str,
                                   file_properties_df, pathmap)
    for file_name in file_names:
        file_name = file_name.with_suffix('.npy')
        if not file_name.exists():
            return False
    return True


def main(args):
    region_location = setup_inputs(args)
    model_saved, transformCompose, config, file_properties_df, pathmap = setup(
        args)
    region_location_datasets = load_audio_files(region_location,
                                                file_properties_df)
    label_names = config['label_names']
    v_str = config['v_str']
    for region, location in region_location:
        print(region, location)
        region_location_ins = region_location_datasets[(region, location)]
        region_location_ins.update_samples_w_clipping_info(
            output_folder=pathmap.clipping_results_path)
        # print('inference part')
        for audio_ins in region_location_ins.values():
            if is_result_exist(audio_ins, label_names, v_str,
                               file_properties_df, pathmap):
                continue
            dataloader = prepare_dataloader_from_audio_ins(
                audio_ins, config, transformCompose)
            outputs = single_file_inference(dataloader, config, model_saved)
            save_results_disk(outputs, audio_ins, label_names, v_str,
                              file_properties_df, pathmap)


# %%

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--index', help='index of array', required=True)
    parser.add_argument('--count', help='count of items', required=True)
    parser.add_argument('-g', '--gpu', help='gpu index', type=int, default=1)
    args = parser.parse_args()

    main(args)

# %%
# class Arguments():
#     def __init__(self,index,count,gpu):
#         self.index=index
#         self.count=count
#         self.gpu=gpu

# args = Arguments(0,1,1)

# main(args)

# print('done')
