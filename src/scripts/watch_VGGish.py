#### when this code is run, it runns VGG on files from PRE_PROCESSED_queue

from nna.models_api import VggishModelWrapper
import tensorflow as tf

from nna.params import PRE_PROCESSED_queue,VGGISH_processing_queue,VGGISH_EMBEDDINGS_queue

from nna import pre_process_func
from nna.pre_process_func import read_queue

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
VGGish_embedding_checkpoint="/home/enis/projects/nna/src/nna/assets/vggish_model.ckpt"
pca_params="/home/enis/projects/nna/src/nna/assets/vggish_pca_params.npz"

vgg = VggishModelWrapper(embedding_checkpoint=VGGish_embedding_checkpoint, #MODEL
                            pca_params= pca_params,
                            sess_config=config,model_loaded=True)


while True:
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
