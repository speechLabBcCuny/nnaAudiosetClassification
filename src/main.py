from subprocess import Popen, PIPE
from pre_process_func import *

import os, shutil
import tensorflow as tf

from time import time
import argparse
import sys
import numpy as np

sys.path.insert(0, './audioset')
import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess
# sp = subprocess.run(["conda","run","-n","speechEnv","python", "test_env.py"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# print(sp.stdout)

#async
from params import INPUT_DIR_PARENT,OUTPUT_DIR
from models_api import VggishModelWrapper



logs_folder="./job_logs/"

ram_memory=100 #GB
segment_length=1 #hour
cpu_count=50
file_per_epoch=70

#1 hour is 0.04
memory_usage=(0.04*30*segment_length*cpu_count)
disk_space=500 #gb
disk_usage= file_per_epoch*2 #gb
assert disk_usage<=disk_space, "not enough disk space"
assert memory_usage<=ram_memory, "not enough ram memory"

abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
output_folder="/scratch/enis/data/nna/NUI_DATA/"

if not os.path.exists(output_folder):
    SRC=abs_input_path
    DEST=output_folder
    shutil.copytree(SRC, DEST, ignore=ig_f)


def inference(pre_processed_npy_files,vgg,embeddings_file_name,batch_size=256):
    # print("len sounds",len(sounds))
    embeddings = np.array([], dtype=np.int16).reshape(0,128)
    postprocessed = np.array([], dtype=np.uint8).reshape(0,128)
    for npy_file in pre_processed_npy_files:
        sound=np.load(npy_file)

        raw_embeddings_file,post_processed_embed_file = vgg.generate_embeddings(sound)

        embeddings = np.concatenate((embeddings,raw_embeddings_file))
        postprocessed = np.concatenate((postprocessed,post_processed_embed_file))

    raw_embeddings_file_name=embeddings_file_name[:-15]+"_rawembeddings.npy"
    np.save(raw_embeddings_file_name,embeddings)
    # postprocessed_batch = pproc.postprocess(embeddings)
    # del embeddings
    # print("saving")
    np.save(embeddings_file_name,postprocessed)
    # del postprocessed_batch
    return embeddings_file_name


########################main.py
my_file=open("/home/enis/projects/nna/mp3files.txt","r+")
temp = my_file.read().splitlines()
my_file.close()

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(1)
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config,)
# vgg,pproc = CreateVGGishNetwork()
vgg = VggishModelWrapper(config=config,model_loaded=True)

for i in range(0,len(temp),file_per_epoch):
    epoch_files=temp[i:i+file_per_epoch]
    parallel_pre_process(epoch_files,output_dir=OUTPUT_DIR,
                            cpu_count=50,segment_len="01:00:00",
                            logs_file_path="logs.txt")

    ##### step - inference
    for input_file in epoch_files:
        mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder=preb_names(input_file,output_folder,abs_input_path)
        if os.path.exists(embeddings_file_name):
            continue
        pre_processed_npy_files=[pre_processed_folder+file for file in os.listdir(pre_processed_folder)]
        # print(pre_processed_npy_files)
        embeddings_file_name=inference(pre_processed_npy_files,vgg,embeddings_file_name,batch_size=128)
        rmv_folder(pre_processed_folder)

# this one does VGGish inference
# echo "10:00/09/July\n" &>> logs_run0.96.txt &&  python main.py &>> logs_run0.96.txt &
#cat "/home/enis/projects/nna/mp3files.txt" | parallel --xargs CUDA_VISIBLE_DEVICES=1 python  pipe4.py --input_files {} &>> logs_run_last.txt &
