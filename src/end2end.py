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
import shutil

from multiprocessing import Pool

def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """

  vggish_slim.define_vggish_slim(training=False)
  checkpoint_path = 'vggish_model.ckpt'
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size

  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)

  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
  pca_params_path="./vggish_pca_params.npz"
  pproc = vggish_postprocess.Postprocessor(pca_params_path)

  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         },pproc


###################input_file############
def preb_names(mp3_file_path,output_dicretory,abs_input_path):

# def mp3toEmbed(mp3_file_path,output_dicretory,abs_input_path):
    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/embeddings/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"

    abs_len=len(abs_input_path)
    relative_path=mp3_file_path[abs_len:]
    mp3_file_name=os.path.basename(mp3_file_path)
    path_to_file=relative_path[:-len(mp3_file_name)]
    mp3_file_name=mp3_file_name.split(".")[0]
    # wav_file_name=mp3_file_name+".wav"
    # wav_file_path=os.path.join(output_dicretory,path_to_file,wav_file_name)

    embeddings_file_name=os.path.join(output_dicretory,path_to_file,mp3_file_name)+"_embeddings.npy"

    segments_folder=os.path.join(output_dicretory,path_to_file,mp3_file_name+"_segments/")
    try:
        os.mkdir(segments_folder)
    except:
        print("folder exists")
    return mp3_file_path,segments_folder,embeddings_file_name
###################DIVIDE MP3#############
def divide_mp3(mp3_file_path,segments_folder,segment_len="01:00:00"):
    sp = subprocess.Popen(['ffmpeg','-i',mp3_file_path,"-c","copy","-map","0",
                                    "-segment_time", "01:00:00", "-f", "segment",
                                    segments_folder+"output%03d.mp3"],shell=False,stderr=subprocess.DEVNULL)
    sp.wait()
    mp3_segments=os.listdir(segments_folder)
    return mp3_segments


def pre_process(mp3_segment,segments_folder):
    # for mp3_segment in mp3_segments:
    input_file_path=segments_folder+mp3_segment
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
    return sound

def inference(sounds,vgg,sess,embeddings_file_name,batch_size=256):
    embeddings = np.array([], dtype=np.int16).reshape(0,128)
    for sound in sounds:
        numberoffiles=sound.shape[0]
        for batch_index in range(0,numberoffiles,batch_size):
            # print("inference")
            if (batch_index+batch_size) <numberoffiles:
                a_batch=sound[batch_index:batch_index+batch_size]
            else:
                a_batch=sound[batch_index:]
            [embedding_batch] = sess.run([vgg['embedding']],
                                     feed_dict={vgg['features']: a_batch})
            embeddings = np.concatenate((embeddings,embedding_batch))
    # print("postprocess")
    del sounds
    postprocessed_batch = pproc.postprocess(embeddings)
    del embeddings
    # print("saving")
    np.save(embeddings_file_name,postprocessed_batch)
    del postprocessed_batch
    return embeddings_file_name

def rmv_segmets(segments_folder):
    try:
        shutil.rmtree(segments_folder)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))


# python end2end.py --input_files `find /home/data/nna/stinchcomb/ -name "*.*3"`
if __name__ == "__main__":
    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/wav_files/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
    parser = argparse.ArgumentParser(description='mp3 to embeddings')
    parser.add_argument('--output_folder', type=str,
                       help='output folder',default="/scratch/enis/data/nna/embeddings/")
    parser.add_argument('--abs_input_path', type=str,
                       help='absoulute input folder such as',default="/home/data/nna/stinchcomb/NUI_DATA/")
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    # args.output_folder= "/scratch/enis/data/nna/wav_files/"
    # args.abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
    # args.input_files=[" "," "," "]
    tf.reset_default_graph()
    sess = tf.Session()
    vgg,pproc = CreateVGGishNetwork()
    for i,input_file in enumerate(args.input_files):
        print("{} - file: {}".format(i,input_file))
        mp3_file_path,segments_folder,embeddings_file_name=preb_names(input_file,
                                            args.output_folder,args.abs_input_path)
        print("dividing into 1 hour segments")
        mp3_segments=divide_mp3(mp3_file_path,segments_folder,segment_len="01:00:00")
        # parallelize pre_process
        mp3_segments.sort()
        pool = mp.Pool(processes=25)
        sounds = [pool.apply(pre_process, args=(mp3_segment,segments_folder)) for mp3_segment in mp3_segments]
        # sound=pre_process(mp3_segment,segments_folder,input_file_path)
        # single cpu and gpu
        embeddings_file_name=inference(sounds,vgg,sess,embeddings_file_name,segments_folder,batch_size=256)
        rmv_segmets(segments_folder)
    # sounds.append(sound)
