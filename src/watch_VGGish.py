#### when this code is run, it runns VGG on files from PRE_PROCESSED_queue

from models_api import VggishModelWrapper
import tensorflow as tf

from params import PRE_PROCESSED_queue,VGGISH_processing_queue,VGGISH_EMBEDDINGS_queue

import pre_process_func

import random
import csv
import time
from pathlib import Path
#PARAMETERS
# there might be multiple GPUs running so we batch input
file_batch_size=10
GPU_INDEX=1

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(GPU_INDEX)
config.gpu_options.allow_growth = True

vgg = VggishModelWrapper(sess_config=config,model_loaded=True)

def read_queue(queue_csv):
    files_in_queue=[]
    if Path(queue_csv).exists():
        with open(queue_csv, newline='') as f:
            reader=csv.reader(f)
            for row in reader:
                files_in_queue.append(row[0])
    return files_in_queue


while True:
    pre_processed_npy_files=[]
    files_in_pre_queu=read_queue(PRE_PROCESSED_queue)
    files_in_processing=read_queue(VGGISH_processing_queue)
    files_done=read_queue(VGGISH_EMBEDDINGS_queue)
    files_to_do = set(files_in_pre_queu).difference(set(files_in_processing)).difference(set(files_done))
    files_to_do = list(files_to_do)[:file_batch_size]
    if files_to_do:
        # save to processing queue
        pre_process_func.save_to_csv(VGGISH_processing_queue,[[str(afile)] for afile in files_to_do])
        for npy_file in files_to_do:
            embeddings_file_path = vgg.inference_file(npy_file)
            pre_process_func.save_to_csv(VGGISH_EMBEDDINGS_queue,[[str(npy_file)]])

        time.sleep(random.randint(0,5))
    else:
        time.sleep(random.randint(60,300))
