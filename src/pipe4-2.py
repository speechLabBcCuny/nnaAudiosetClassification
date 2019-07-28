#this one does audioset inference
# version that uses logic from: https://github.com/IBM/MAX-Audio-Classifier
#reqiures:
# curl https://storage.googleapis.com/deep_learning_enis/speech_audio_understanding/Eusipco2018_Google_AudioSet/md_50000_iters.tar --output ./md_50000_iters.tar
# curl https://gist.githubusercontent.com/EnisBerk/19d817e9a6a9c060465a5b95c8d54d97/raw/8e0c24e92853e779e222bc3a152d7d65351cfe52/model.py --output ./model.py
#
import torch
from pre_process_func import *

from time import time
import argparse
import sys
import numpy as np
# from model import *

sys.path.insert(0, './audioset')

import tensorflow as tf

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess



def uint8_to_float32(x):
  return (np.float32(x) - 128.) / 128.

def bool_to_float32(y):
  return np.float32(y)



# makes audioset inference, prediction is over 10seconds
def inference(input_x,batch_size=500,first_k=1):
  predicted_labels=[]
  y_preds=np.empty((0,first_k),dtype=np.float32)
  y_pred_indexs=np.empty((0,first_k),dtype=np.long)
  start=0
  end=input_x.shape[0]-batch_size
  left=input_x.shape[0]%batch_size
  loop_count=10 # number of inputs for single output
  # print(input_x.shape)
  # print(left)
  i=0
  for i in range(0,end,batch_size):
    # y_pred=model(input_x[i:i+batch_size].reshape(int(batch_size/loop_count),loop_count,128))
    processed_embeddings = input_x[i:i+batch_size].reshape(int(batch_size/loop_count),loop_count,128)
    y_pred = output_tensor.eval(feed_dict={input_tensor: processed_embeddings}, session=session_classify)
    y_pred=torch.tensor(y_pred)
    y_pred, y_pred_index = torch.topk(y_pred, first_k, dim=1, largest=True, sorted=True)
    y_preds=np.concatenate((y_preds,y_pred.detach().numpy()),0)
    y_pred_indexs=np.concatenate((y_pred_indexs,y_pred_index.detach().numpy()),0)
  left_loop_count=left%loop_count
  left-=left_loop_count
  if left>0:
#   for i in range(input_x.shape[0]-left,input_x.shape[0]):
#     print(i)
    processed_embeddings = input_x[i:i+left].reshape(int(left/loop_count),loop_count,128)
    y_pred = output_tensor.eval(feed_dict={input_tensor: processed_embeddings}, session=session_classify)
    # y_pred=model(input_x[i:i+left].reshape(int(left/loop_count),loop_count,128))
#     y_pred=model(input_x[i:i+1].reshape(1,10,128))
    y_pred=torch.tensor(y_pred)
    y_pred, y_pred_index = torch.topk(y_pred, first_k, dim=1, largest=True, sorted=True)
    y_preds=np.concatenate((y_preds,y_pred.detach().numpy()),0)
    y_pred_indexs=np.concatenate((y_pred_indexs,y_pred_index.detach().numpy()),0)
  if left_loop_count>0:
    processed_embeddings = input_x[-left_loop_count:].reshape(1,left_loop_count,128)
    y_pred = output_tensor.eval(feed_dict={input_tensor: processed_embeddings}, session=session_classify)
    y_pred=model(input_x[-left_loop_count:].reshape(1,left_loop_count,128))
#     y_pred=model(input_x[i:i+1].reshape(1,10,128))
    y_pred=torch.tensor(y_pred)
    y_pred, y_pred_index = torch.topk(y_pred, first_k, dim=1, largest=True, sorted=True)
    y_preds=np.concatenate((y_preds,y_pred.detach().numpy()),0)
    y_pred_indexs=np.concatenate((y_pred_indexs,y_pred_index.detach().numpy()),0)
  return y_preds,y_pred_indexs

if __name__ == "__main__":

    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/wav_files/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
    parser = argparse.ArgumentParser(description='mp3 to embeddings')
    parser.add_argument('--output_folder', type=str,
       help='output folder',default="/scratch/enis/data/nna/NUI_DATA/")
    parser.add_argument('--abs_input_path', type=str,
       help='absoulute input folder such as',default="/home/data/nna/stinchcomb/NUI_DATA/")
    parser.add_argument('--segment_len', type=str,
       help='length of segments',default="01:00:00")
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()

    # args.output_folder= "/scratch/enis/data/nna/wav_files/"
    # args.abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
    # args.input_files=[" "," "," "]
    # create directory tree from original folder

    # input_files=[" "," "," "]
    #find all mp3 files
    # input_files=!find '{abs_input_path}' -name "*.*3"
    input_files=args.input_files
    output_folder=args.output_folder
    abs_input_path=args.abs_input_path
    # if not os.path.exists(output_folder):
    #     SRC=abs_input_path
    #     DEST=output_folder
    #     shutil.copytree(SRC, DEST, ignore=ig_f)

    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units = 1024
    drop_rate = 0.5
# initilize model
    # model = FeatureLevelSingleAttention(
    #             freq_bins, classes_num, hidden_units, drop_rate)
    # checkpoint = torch.load("md_50000_iters.tar")
    # model.load_state_dict(checkpoint['state_dict'])
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    # iteration = checkpoint['iteration']
    #
    # model.eval()
    session_classify = tf.keras.backend.get_session()
    classify_model = tf.keras.models.load_model("classifier_model.h5", compile=False)

    output_tensor = classify_model.output
    input_tensor = classify_model.input

    for i,input_file in enumerate(input_files):
        start=time()
        print("{} - file: {}".format(i,input_file))
        sys.stdout.flush()
        ##### step - 0 get prepare names
        mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder=preb_names(input_file,
          output_folder,abs_input_path)
        predictions_file_name=embeddings_file_name[:-15]+"_preds.npy"
        prediction_index_file_name=embeddings_file_name[:-15]+"_preds_index.npy"
        if os.path.exists(predictions_file_name):
            continue
        preprocessed=np.load(embeddings_file_name)
        preprocessed=uint8_to_float32(preprocessed)
        preprocessed_tensor = torch.tensor(preprocessed,dtype=torch.float32)
        y_preds,y_preds_index=inference(preprocessed_tensor,batch_size=500,first_k=527)
        np.save(predictions_file_name,y_preds)
        np.save(prediction_index_file_name,y_preds_index)
        end=time()
        print("It took {} seconds".format(end-start))
        sys.stdout.flush()