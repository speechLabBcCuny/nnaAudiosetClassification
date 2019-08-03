# Api(wrapper) for models

import numpy as np
import tensorflow as tf
import os,requests

from models.audioset import vggish_input
from models.audioset import vggish_params
from models.audioset import vggish_postprocess
from models.audioset import vggish_slim

from params import *

class VggishModelWrapper:
    """
    Contains core functions to generate embeddings and classify them.
    Also contains any helper function required.
    """

    def __init__(self, embedding_checkpoint=VGGish_EMBEDDING_CHECKPOINT,
                pca_params= PCA_PARAMS,
                # classifier_model="assets/classifier_model.h5",
                labels_path=LABELS,
                sess_config=tf.ConfigProto()
                model_loaded=True):

        # # Initialize the classifier model
        # self.session_classify = tf.keras.backend.get_session()
        # self.classify_model = tf.keras.models.load_model(classifier_model, compile=False)
        self.embedding_checkpoint=embedding_checkpoint
        self.pca_params=pca_params
        self.model_loaded = model_loaded
        # Initialize the vgg-ish embedding model and load post-Processsing
        if model_loaded:
            self.load_pre_trained_model(embedding_checkpoint,pca_params)
        # Metadata
        self.labels = self.load_labels(labels_path)

    def load_pre_trained_model(self,
            embedding_checkpoint=self.embedding_checkpoint,
            pca_params=self.pca_params):

        # Initialize the vgg-ish embedding model
        self.graph_embedding = tf.Graph()
        with self.graph_embedding.as_default():
            self.session_embedding = tf.Session(config=self.sess_config)
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.session_embedding,
                                                        embedding_checkpoint)
            self.features_tensor = self.session_embedding.graph.\
                get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.session_embedding.graph.\
                get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Prepare a postprocessor to munge the vgg-ish model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params)
        self.model_loaded=True


    def generate_embeddings(self, sound,batch_size=256):
        """
        Generates embeddings as per the Audioset VGG-ish model.
        Post processes embeddings with PCA Quantization
        Input args:
            sound   = numpy array with samples from mp3file_to_examples
            batch_size
        Returns:
                list of numpy arrays
                [raw_embeddings,post_processed_embeddings]
        """
        if not self.model_loaded:
            self.load_pre_trained_model()

        input_len=sound.shape[0]
        raw_embeddings = np.array([], dtype=np.int16).reshape(0,128)
        for batch_index in range(0,input_len,batch_size):
            a_batch=sound[batch_index:batch_index+batch_size]
            # examples_batch = vggish_input.wavfile_to_examples(wav_file)
            [embedding_batch] = self.session_embedding.\
                run([self.embedding_tensor],
                    feed_dict={self.features_tensor: a_batch})
            raw_embeddings = np.concatenate((raw_embeddings,embedding_batch))
        post_processed_embeddings = self.pproc.postprocess(raw_embeddings)

        return raw_embeddings,post_processed_embeddings

    def pre_process(self,)


















    # returns an array with label strings, index of array corresponding to class index
    def load_labels(self,csv_file="assets/class_labels_indices.csv"):
        if os.path.exists(csv_file):
            csvfile=open(csv_file, newline='')
            csv_lines=csvfile.readlines()
            csvfile.close()
        else:
            url="https://raw.githubusercontent.com/qiuqiangkong/audioset_classification/master/metadata/class_labels_indices.csv"
            with requests.Session() as s:
                download = s.get(url)
                decoded_content = download.content.decode('utf-8')
                csv_lines=decoded_content.splitlines()
        labels=[]
        reader = csv.reader(csv_lines, delimiter=',')
        headers=next(reader)
        for row in reader:
          labels.append(row[2])
        return labels
