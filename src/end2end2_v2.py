colab=False

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
import distutils.dir_util

# from multiprocessing import Pool
import multiprocessing as mp

from itertools import repeat

import librosa



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
    return mp3_file_path,segments_folder,embeddings_file_name
###################DIVIDE MP3#############
def divide_mp3(mp3_file_path,segments_folder,segment_len="01:00:00"):
    print(segments_folder)
    print("******",os.path.exists(segments_folder))
    sys.stdout.flush()
    if not os.path.exists(segments_folder):
        os.mkdir(segments_folder)
    sp = subprocess.run(['ffmpeg','-i',mp3_file_path,"-c","copy","-map","0",
                                "-segment_time", segment_len, "-f", "segment",
                                segments_folder+"output%03d.mp3"])
    # sp.wait()
    print("processssssseing done")
    sys.stdout.flush()
    mp3_segments=os.listdir(segments_folder)
    return mp3_segments


def pre_process(mp3_segment,segments_folder):
    # for mp3_segment in mp3_segments:
    input_file_path=segments_folder+mp3_segment
# https://stackoverflow.com/posts/1606870/revisions
    # cmd = ['ffmpeg', '-i', input_file_path, '-f', 'wav', '-']
    # sp = subprocess.Popen(cmd,
    #     stdout=subprocess.PIPE,stderr=None,shell=False)
    ##### !NO! #####
    # creates deadlock when stdout used with Popen
    # sp.wait()
    ##### !NO! #####
    # stdout, stderr = p.communicate()
    # bio = BytesIO(out)
    # wav_data, sr = sf.read(bio, dtype='int16')
    # tmp_wav=input_file_path[:-4]+".wav"
#########
    # wav_data = AudioSegment.from_mp3(input_file_path)
    # sr=wav_data.frame_rate
    # wav_data = wav_data.get_array_of_samples()
    # wav_data = np.array(wav_data)
#####
    wav_data, sr = librosa.load(input_file_path,sr=44100,mono=False,)
    wav_data=wav_data.T
    wav_data=wav_data.reshape(-1,2)

    maxv = np.iinfo(np.int16).max+1

    wav_data=(wav_data * maxv).astype(np.int16)
    ########
    # del bio
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data/  32768.0  # Convert to [-1.0, +1.0]
    print("ready to waveform")
    sys.stdout.flush()
    print(wav_data.size)
    sound= vggish_input.waveform_to_examples(wav_data, sr)
    print(sound.size)
    sound=sound.astype(np.float32)
    print("ready to return")
    sys.stdout.flush()
    tmp_npy=input_file_path[:-4]+".npy"
    np.save(tmp_npy,sound)
    # return sound

    # return sound

def inference(sounds,vgg,sess,embeddings_file_name,batch_size=256):
    # print("len sounds",len(sounds))
    embeddings = np.array([], dtype=np.int16).reshape(0,128)
    for sound in sounds:
        print(len(sound))
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
    # del sounds
    raw_embeddings_file_name=embeddings_file_name[:-15]+"rawembeddings.npy"
    np.save(raw_embeddings_file_name,embeddings)
    postprocessed_batch = pproc.postprocess(embeddings)
    # del embeddings
    # print("saving")
    np.save(embeddings_file_name,postprocessed_batch)
    # del postprocessed_batch
    return embeddings_file_name

def rmv_segmets(segments_folder):
    try:
        shutil.rmtree(segments_folder)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

def end2end(input_file,output_folder,abs_input_path,sess,vgg,pproc,cpu_count=2,segment_len="01:00:00"):

  mp3_file_path,segments_folder,embeddings_file_name=preb_names(input_file,
                                      output_folder,abs_input_path)
  if os.path.exists(embeddings_file_name):
    return None
#   print("dividing into 1 hour segments")
  # print(mp3_file_path,segments_folder,embeddings_file_name)
  # print("divide mp3")
  mp3_segments=divide_mp3(mp3_file_path,segments_folder,segment_len=segment_len)
  mp3_segments.sort()

#   print("pre_process")
  print("pre_process")
  sys.stdout.flush()
  with mp.Pool(processes=cpu_count) as pool:
    sounds = pool.starmap(pre_process, zip(mp3_segments,repeat(segments_folder)))

  # pool = Pool(processes=cpu_count)
  # sounds = [pool.apply(pre_process, args=(mp3_segment,segments_folder)) for mp3_segment in mp3_segments]
  # printm()
  # single cpu and gpu
  print("inference")
  sys.stdout.flush()
  embeddings_file_name=inference(sounds,vgg,sess,embeddings_file_name,batch_size=256)
#   printm()
  rmv_segmets(segments_folder)
  # if colab==True:
  #     src=os.path.dirname(embeddings_file_name)
  #     dest=os.path.join( "gs://deep_learning_enis/speech_audio_understanding/",src[2:])
  #     !gsutil rsync -r '{src}' '{dest}'

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

# find /home/data/nna/stinchcomb/ -name "*.*3" -print0 | xargs -0 python end2end.py --input_files &> endlogs.txt &
if __name__ == "__main__":

    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/wav_files/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
    parser = argparse.ArgumentParser(description='mp3 to embeddings')
    parser.add_argument('--output_folder', type=str,
                       help='output folder',default="/scratch/enis/data/nna/wav_files/")
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
    if not os.path.exists(output_folder):
        SRC=abs_input_path
        DEST=output_folder
        shutil.copytree(SRC, DEST, ignore=ig_f)

    input_files.sort()
    tf.reset_default_graph()
    sess = tf.Session()
    vgg,pproc = CreateVGGishNetwork()
    for i,input_file in enumerate(input_files):
        start=time()
        print("{} - file: {}".format(i,input_file))
        sys.stdout.flush()
        end2end(input_file,output_folder,abs_input_path,sess,vgg,pproc,cpu_count=8,segment_len=args.segment_len)
        end=time()
        print("It took {} seconds".format(end-start))
        sys.stdout.flush()
    sess.close()
