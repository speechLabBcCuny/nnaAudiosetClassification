#### when this code is run, it runns VGG on files from PRE_PROCESSED_queue

from models_api import classicML

from params import VGGISH_EMBEDDINGS_queue,Audioset_processing_queue,Audioset_output_queue



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

    parser.add_argument("-m","--modelPath", help="classifier model path")
    parser.add_argument("-n","--modelName", help="ModelName to use with file and folder names")
    parser.add_argument("--many2one", help="set True if model producing 1 predict per 10 second",action="store_true")

    args = parser.parse_args()


    if args.modelPath==None or args.modelName==None:
        print("--modelPath and --modelName is required")
        return -1

    Audioset_processing_queue='./job_logs/'+args.modelPath+'_processing_queue.csv'
    Audioset_output_queue='./job_logs/'+args.modelPath+'_output_queue.csv'
    
    classifier=classicML(classifier_model_path=args.modelPath)

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
                classifier.classify_file(vgg_npy_file,ModelName=args.modelName,many2one=args.many2one)
                pre_process_func.save_to_csv(Audioset_output_queue,[[str(vgg_npy_file)]])

            time.sleep(random.randint(0,5))
        else:
            time.sleep(random.randint(60,300))
