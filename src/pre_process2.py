import os
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import read
import argparse

first_run=False

if first_run==True:
#clone repository
#   !git clone https://github.com/tensorflow/models.git
   # Grab the VGGish model
    os.system("curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt")
    os.system("curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz")
    os.system("svn export https://github.com/tensorflow/models/trunk/research/audioset")

import sys
sys.path.insert(0, './audioset')

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess


batch_size=128
sr=44100

#wav file: indexes of signal that it generated (10,22) means embeddings[10,22] belong to that wav file
data_file_indexes={}


def vggish_inference():

    # path_to_files="/scratch/enis/data/nna/samples_wav/split_02:00/"
    # path_to_files=""
#     wav_data_files=args.input_files
    # wav_data_files=os.listdir(path_to_files)
    data_dir="projects/nna/data/"
    '''- preprocessed_index is a dictionary with keys which are
       full-path to files, values are indexes of
       array called all_data for each files_exists
       - preprocessed_data is an numpy array with shape (total_data_len,96,64)
       total_data_len is equal to total number of seconds in audio samples'''
    preprocessed_index=np.load(os.path.join(data_dir,"data_file_indexes.npy"))
    preprocessed_data=np.load(os.path.join(data_dir,"preprocessed.npy"))

    embeddings = np.array([], dtype=np.float32).reshape(0,128)
    #  TODO
    # postprocessed = np.array([], dtype=np.float32).reshape(0,128)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

  # vggish_params.EXAMPLE_HOP_SECONDS = hop_size

  # vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    embeddings = np.array([], dtype=np.int16).reshape(0,128)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        #TODO make this paramater
        checkpoint_path = 'vggish_model.ckpt'
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        batch_size=256
        numberoffiles=preprocessed_data.shape[0]
        for batch_index in range(0,numberoffiles,batch_size):
            if (batch_index+batch_size) <numberoffiles:
                a_batch=preprocessed_data[batch_index:batch_index+batch_size]
            else:
                a_batch=preprocessed_data[batch_index:]

            # wav to one second signals
            #   batch_processed = np.array([], dtype=np.float64).reshape(0,96,64)

            [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: a_batch})
        # print(embedding_batch)
            embeddings = np.concatenate((embeddings,embedding_batch))
        # TODO we will do post_processing in another file
        # postprocessed_batch = pproc.postprocess(embedding_batch)
        # print(postprocessed_batch)
    np.save(os.path.join(args.output_folder,"embeddings.npy"),embeddings)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wav files pre-process' )

    parser.add_argument('--output_folder', type=str,
                       help='output folder')
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)


    vggish_inference()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # tf.reset_default_graph()
    # sess = tf.Session()
    #
    # vgg = CreateVGGishNetwork()
    #
    # pca_params_path = 'vggish_pca_params.npz'
    # pproc = vggish_postprocess.Postprocessor(pca_params_path)
    #
    #
    #
    # for batch_index in range(0,len(wav_data_files),batch_size):
    #     if (batch_index+batch_size) <len(wav_data_files):
    #         a_batch=wav_data_files[batch_index:batch_index+batch_size]
    #     else:
    #         a_batch=wav_data_files[batch_index:]
    #
    #     # wav to one second signals
    #     #   batch_processed = np.array([], dtype=np.float64).reshape(0,96,64)
    #     for i,wav_file in enumerate(a_batch):
    #         rate,sound = read(os.path.join(path_to_files,wav_file))
    #         sound=np.array(sound,dtype=float)
    #
    #         sound = vggish_input.waveform_to_examples(sound, sr)
    #         data_file_indexes[wav_file]=(counter,counter+len(sound))
    #         counter+=len(sound)
    # #       batch_processed=np.concatenate((batch_processed,sound))
    # #     signals to tensors
    #         [embedding_batch] = sess.run([vgg['embedding']],
    #                                feed_dict={vgg['features']: sound})
    #         # signals to tensors
    #         #   [embedding_batch] = sess.run([vgg['embedding']],
    #         #                            feed_dict={vgg['features']: batch_processed})
    #         embeddings = np.concatenate((embeddings,embedding_batch))
    #         postprocessed_batch = pproc.postprocess(embeddings)
    #         postprocessed = np.concatenate((postprocessed,postprocessed_batch))
    # #   embeddings=np.concatenate((embeddings,embedding_batch))
    # #     postprocessed_batch = pproc.postprocess(embedding_batch)
    # #     postprocessed=np.concatenate((postprocessed,postprocessed_batch))
    #     if batch_index%(batch_size*20)==0:
    #         print(batch_index/len(wav_data_files))
    #
    # np.save(os.path.join(args.output_folder,"data_file_indexes.npy"),data_file_indexes)
    # np.save(os.path.join(args.output_folder,"embeddings.npy"),embeddings)
    # np.save(os.path.join(args.output_folder,"postprocessed.npy"),postprocessed)

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
