import os
import numpy as np
from scipy.io.wavfile import read
import argparse

files_exists=True

if files_exists==False:
#clone repository
#   !git clone https://github.com/tensorflow/models.git
   # Grab the VGGish model
    os.system("curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt")
    os.system("curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz")
    os.system("svn export https://github.com/tensorflow/models/trunk/research/audioset")

import sys
sys.path.insert(0, './audioset')

# import vggish_slim
import vggish_params
import vggish_input
# import vggish_postprocess


batch_size=128
# sr=44100

#wav file: indexes of signal that it generated (10,22) means embeddings[10,22] belong to that wav file
data_file_indexes={}
counter=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wav files pre-process' )

    parser.add_argument('--output_folder', type=str,
                       help='output folder')
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    parser.add_argument("--job_id",type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    path_to_files=""
#     path_to_files=""
    wav_data_files=args.input_files
    # wav_data_files=os.listdir(path_to_files)


    # embeddings = np.array([], dtype=np.float32).reshape(0,128)
    # postprocessed = np.array([], dtype=np.float32).reshape(0,128)
    pre_processed = np.array([], dtype=np.int16).reshape(0,96,64)
    for batch_index in range(0,len(wav_data_files),batch_size):
        if (batch_index+batch_size) <len(wav_data_files):
            a_batch=wav_data_files[batch_index:batch_index+batch_size]
        else:
            a_batch=wav_data_files[batch_index:]

        # wav to one second signals

        for i,wav_file in enumerate(a_batch):
            path_to_file = os.path.join(path_to_files,wav_file)
            # rate,sound = read(path_to_file)
            # sound=np.array(sound,dtype=float)
            sound=vggish_input.wavfile_to_examples(path_to_file)
            # sound = vggish_input.waveform_to_examples(sound, sr)
            data_file_indexes[wav_file]=(counter,counter+len(sound))
            counter+=len(sound)
            pre_processed=np.concatenate((pre_processed,sound))
    #     signals to tensors

        if batch_index%(batch_size*20)==0:
            print(batch_index/len(wav_data_files))

    np.save(os.path.join(args.output_folder,"data_file_indexes_"+args.job_id+".npy"),data_file_indexes)
    # np.save(os.path.join(args.output_folder,"embeddings_"+args.job_id+".npy"),embeddings)
    np.save(os.path.join(args.output_folder,"postprocessed_"+args.job_id+".npy"),pre_processed)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='mp3 to wav files')
#     parser.add_argument('--output_folder', type=str,
#                        help='output folder')
#     parser.add_argument('--input_files',nargs='+',type=str,default=None)
#     # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
#     args = parser.parse_args()
#     if not os.path.isdir(args.output_folder):
#         os.makedirs(args.output_folder)
#     preprocess_wav(args.input_files,args.output_folder)
