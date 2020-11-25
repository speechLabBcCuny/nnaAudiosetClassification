from nna.params import disk_usage, disk_space, memory_usage, ram_memory
from nna.params import INPUT_DIR_PARENT, OUTPUT_DIR
from nna.models_api import VggishModelWrapper
# from nna import vggish_postprocess
# from nna import vggish_input
# from nna import vggish_params
# from nna import vggish_slim
# from subprocess import Popen, PIPE
from nna import pre_process_func

import os
import shutil
import tensorflow as tf

# from time import time
# import argparse
import sys
# import numpy as np

sys.path.insert(0, "./audioset")
# sp = subprocess.run(["conda","run","-n","speechEnv","python", "test_env.py"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# print(sp.stdout)

# async

assert disk_usage <= disk_space, "not enough disk space"
assert memory_usage <= ram_memory, "not enough ram memory"

# BUG
ig_f = "I do not know what is if_f"
if not os.path.exists(OUTPUT_DIR):
    SRC = INPUT_DIR_PARENT
    DEST = OUTPUT_DIR
    shutil.copytree(SRC, DEST, ignore=ig_f)

# main.py
my_file = open("/home/enis/projects/nna/mp3files.txt", "r+")
temp = my_file.read().splitlines()
my_file.close()

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(1)
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config,)
# vgg,pproc = CreateVGGishNetwork()
vgg = VggishModelWrapper(sess_config=config, model_loaded=True)
# BUG
file_per_epoch = " I do not know where this is from :("
for i in range(0, len(temp), file_per_epoch):
    epoch_files = temp[i:i + file_per_epoch]
    pre_process_func.parallel_pre_process(epoch_files,
                                          output_dir=OUTPUT_DIR,
                                          cpu_count=50,
                                          segment_len="01:00:00",
                                          logs_file_path="logs.txt")

    ##### step - inference
    for input_file in epoch_files:
        segments_folder, embeddings_file_name, pre_processed_folder = pre_process_func.preb_names(
            input_file, OUTPUT_DIR, INPUT_DIR_PARENT)
        if os.path.exists(embeddings_file_name):
            continue
        pre_processed_npy_files = [
            pre_processed_folder + file
            for file in os.listdir(pre_processed_folder)
        ]
        # print(pre_processed_npy_files)
        embeddings_file_name = pre_process_func.inference(
            pre_processed_npy_files, vgg, embeddings_file_name, batch_size=128)
        pre_process_func.rmv_folder(pre_processed_folder)

# this one does VGGish inference
# echo "10:00/09/July\n" &>> logs_run0.96.txt &&  python main.py &>> logs_run0.96.txt &
# cat "/home/enis/projects/nna/mp3files.txt" | parallel --xargs CUDA_VISIBLE_DEVICES=1 python  pipe4.py --input_files {} &>> logs_run_last.txt &
