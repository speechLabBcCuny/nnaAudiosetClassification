#### when this code is run, it runns VGG on files from PRE_PROCESSED_queue

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from models_api import FC_embed
import torch

from params import VGGISH_EMBEDDINGS_queue,Audioset_processing_queue,Audioset_output_queue

Audioset_processing_queue='./job_logs/FCmodel_processing_queue.csv'
Audioset_output_queue='./job_logs/FCmodel_output_queue.csv'

import pre_process_func
from pre_process_func import read_queue

import random
import csv
import time
from pathlib import Path
#PARAMETERS
# there might be multiple GPUs running so we batch input
file_batch_size=10

# # GPU_INDEX=1
# # audioset classificaton batch size
# audioset_batch_size=500


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g","--gpu", help="gpu index",default=0)

    args = parser.parse_args()

    # bot_client = slack.WebClient(token=slack_bot_token)
    # send_logs(args.title,bot_client)
    GPU_INDEX=args.gpu
    GPU_INDEX=str(GPU_INDEX)

    device = torch.device("cuda:"+GPU_INDEX if torch.cuda.is_available() else "cpu")

    classifier=FC_embed(device=device,classifier_model_path="assets/mlp_models/bird_FC_089valid_20191122-131739.pth")

    while True:
        # files_in_pre_queu=read_queue(PRE_PROCESSED_queue)
        # files_in_processing=read_queue(VGGISH_processing_queue)
        vgg_embedding_files=set(read_queue(VGGISH_EMBEDDINGS_queue))
        files_in_processing=set(read_queue(Audioset_processing_queue))
        files_done=set(read_queue(Audioset_output_queue))
        files_to_do = set(vgg_embedding_files).difference(set(files_in_processing),set(files_done))
        files_to_do=list(files_to_do)
        if files_to_do:
            if len(files_to_do)>file_batch_size:
                files_to_do = random.sample(files_to_do, k=file_batch_size)
            # save to processing queue
            pre_process_func.save_to_csv(Audioset_processing_queue,[[str(afile)] for afile in files_to_do])
            for vgg_npy_file in files_to_do:
                classifier.classify_file(vgg_npy_file)
                pre_process_func.save_to_csv(Audioset_output_queue,[[str(vgg_npy_file)]])

            time.sleep(random.randint(0,5))
        else:
            time.sleep(random.randint(60,300))
