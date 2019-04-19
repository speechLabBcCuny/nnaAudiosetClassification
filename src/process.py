import subprocess
import os
from io import BytesIO
import copy
import soundfile as sf
from pydub import AudioSegment
from time import time

import tensorflow as tf
import argparse

import sys
sys.path.insert(0, './audioset')

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess
import numpy as np


# def mp3toEmbed(mp3_file_path,output_dicretory,abs_input_path):

pca_params_path="./vggish_pca_params.npz"
mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
output_dicretory="/scratch/enis/data/nna/wav_files/"
abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
abs_len=len(abs_input_path)
relative_path=mp3_file_path[abs_len:]
mp3_file_name=os.path.basename(mp3_file_path)
path_to_file=relative_path[:-len(mp3_file_name)]
mp3_file_name=mp3_file_name.split(".")[0]
wav_file_name=mp3_file_name+".wav"
wav_file_path=os.path.join(output_dicretory,path_to_file,wav_file_name)

segment_folder=os.path.join(output_dicretory,path_to_file,mp3_file_name+"_segments")
embeddings_file_name=os.path.join(output_dicretory,path_to_file,mp3_file_name)+"_embeddings.npy"
segment_folder+="/"
try:
    os.mkdir(segment_folder)
except:
    print("folder exists")

sp = subprocess.Popen(['ffmpeg','-i',mp3_file_path,"-c","copy","-map","0",
                                "-segment_time", "01:00:00", "-f", "segment",
                                segment_folder+"output%03d.mp3"],shell=False,stderr=subprocess.DEVNULL)
sp.wait()
mp3_segments=os.listdir(segment_folder)
mp3_segments=mp3_segments[:2]
batch_size=128
sr=44100

#wav file: indexes of signal that it generated (10,22) means embeddings[10,22] belong to that wav file
# data_file_indexes={}
def run():
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        print("Vggish")
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        checkpoint_path = 'vggish_model.ckpt'
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        for mp3_segment in mp3_segments:
            input_file_path=segment_folder+mp3_segment
            print("ffmpeg")
            sp = subprocess.Popen(
                ['ffmpeg', '-i', input_file_path, '-f', 'wav', '-'],shell=False,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            # sp.wait()
            bio = BytesIO(sp.stdout.read())
            wav_data, sr = sf.read(bio, dtype='int16')
            del bio
            assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
            wav_data = wav_data/  32768.0  # Convert to [-1.0, +1.0]
            print("waveform")
            sound= vggish_input.waveform_to_examples(wav_data, sr)
            del wav_data
            print("INF")
            numberoffiles=sound.shape[0]
            embeddings = np.array([], dtype=np.int16).reshape(0,128)
            batch_size=256
            for batch_index in range(0,numberoffiles,batch_size):
                print("inference")
                if (batch_index+batch_size) <numberoffiles:
                    a_batch=sound[batch_index:batch_index+batch_size]
                else:
                    a_batch=sound[batch_index:]
                [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: a_batch})
                embeddings = np.concatenate((embeddings,embedding_batch))
            print("postprocess")
            postprocessed_batch = pproc.postprocess(embeddings)
            del embeddings
        print("saving")
        np.save(embeddings_file_name,postprocessed_batch)
        del postprocessed_batch
        for mp3_segment in mp3_segments:
            file2del=segment_folder+mp3_segment
            os.remove(file2del)

run()

if __name__ == "__main__":
    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/wav_files/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"

    parser = argparse.ArgumentParser(description='mp3 to embeddings')
    parser.add_argument('--output_folder', type=str,
                       help='output folder')
    parser.add_argument('--abs_input_path', type=str,
                       help='absoulute input folder such as /home/data/nna/stinchcomb/NUI_DATA/')
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    # args.output_folder= "/scratch/enis/data/nna/wav_files/"
    # args.abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
    # args.input_files=[" "," "," "]
    for input_file in args.input_files:
        mp3toEmbed(input_file,args.output_folder,args.abs_input_path)
    # sounds.append(sound)
