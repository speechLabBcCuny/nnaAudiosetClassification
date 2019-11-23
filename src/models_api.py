# Api(wrapper) for models

import numpy as np
import torch

import tensorflow as tf
from tensorflow.keras.backend import set_session

import os

import sys
# sys.path.insert(0, '/Users/berk/Documents/workspace/speech_audio_understanding/src/models/audioset')
# sys.path.insert(0, './models/audioset')

from models.audioset import vggish_input
from models.audioset import vggish_params
from models.audioset import vggish_postprocess
from models.audioset import vggish_slim

from params import *
import pre_process_func

from params import PRE_PROCESSED_queue,VGGISH_EMBEDDINGS_queue
from params import EXCERPT_LENGTH
class VggishModelWrapper:
    """
    Contains core functions to generate embeddings and classify them.
    Also contains any helper function required.
    """

    def __init__(self,
                embedding_checkpoint=VGGish_EMBEDDING_CHECKPOINT, #MODEL
                pca_params= PCA_PARAMS,
                # classifier_model="assets/classifier_model.h5",
                labels_path=LABELS,
                sess_config=tf.ConfigProto(),
                model_loaded=True):

        # # Initialize the classifier model
        # self.session_classify = tf.keras.backend.get_session()
        # self.classify_model = tf.keras.models.load_model(classifier_model, compile=False)
        self.embedding_checkpoint=embedding_checkpoint
        self.pca_params=pca_params
        self.model_loaded = model_loaded
        self.sess_config=sess_config
        # Initialize the vgg-ish embedding model and load post-Processsing
        if model_loaded:
            self.load_pre_trained_model(embedding_checkpoint,pca_params)
        # Metadata

    def load_pre_trained_model(self,
            embedding_checkpoint=None,
            pca_params=None):
        if embedding_checkpoint==None:
            embedding_checkpoint=self.embedding_checkpoint,
        if pca_params==None:
            pca_params=self.pca_params
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
            sound (numpy.ndarray) : samples from pre_process_func.pre_process
            batch_size (int) : batch size of input to vgg inference
        Returns:
                list : list of numpy arrays
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
        #TODO post-processing can be batched as well
        post_processed_embeddings = self.pproc.postprocess(raw_embeddings)

        return raw_embeddings,post_processed_embeddings
    def inference_file(self,pre_processed_npy_file,batch_size=256):
        """
        Calls vgg.generate_embeddings per file from pre_processed_npy_files.
        Saves embeddings and raw_embeddings into two different files.

        Input args:
            pre_processed_npy_file : paths to file storing sound
        Returns:

        """
        # for npy_file in pre_processed_npy_files:
        sound=np.load(pre_processed_npy_file)
        raw_embeddings,postprocessed = self.generate_embeddings(sound,batch_size)

        npy_file=Path(pre_processed_npy_file)
        file_index = npy_file.stem.replace("_preprocessed","").replace("output","")
        original_file_stem = npy_file.parent.stem.replace("_preprocessed","")
        vgg_folder = npy_file.parent.parent / npy_file.parent.stem.replace("_preprocessed","_vgg")
        raw_embeddings_file_path = vgg_folder / (str(original_file_stem) + "_rawembeddings"+file_index+".npy")
        embeddings_file_path =  vgg_folder / (str(original_file_stem) + "_embeddings"+file_index+".npy")

        Path(vgg_folder).mkdir(parents=True, exist_ok=True)
        np.save(raw_embeddings_file_path,raw_embeddings)
        np.save(embeddings_file_path,postprocessed)

        npy_file.unlink()
        return embeddings_file_path


class AudioSet():
    def __init__(self,
                classifier_model_path="assets/classifier_model.h5",
                labels_path=LABELS,
                sess_config=tf.ConfigProto(),
                model_loaded=True,
                vggish_model=None):

        self.session_classification=None
        self.classifier_model_path = classifier_model_path
        self.model_loaded = model_loaded
        self.sess_config = sess_config
        self.vggish_model=vggish_model

        # Initialize the vgg-ish embedding model and load post-Processsing
        if model_loaded:
            self.load_pre_trained_model()
        # Metadata
        self.labels = self.load_labels(labels_path)

    def load_pre_trained_model(self,
                classifier_model_path=None):
        if classifier_model_path==None:
            classifier_model_path=self.classifier_model_path
        # Initialize the Audioset classification model
        sess = tf.Session(config=self.sess_config)
        set_session(sess)
        self.session_classification = tf.keras.backend.get_session()
        self.classifier_model = tf.keras.models.load_model(classifier_model_path,
                                                            compile=False)

        self.model_loaded=True

    def classify_embeddings(self, processed_embeddings):
        """
        Performs classification on PCA Quantized Embeddings.
        Input args:
            processed_embeddings = numpy array of shape (N,10,128), dtype=float32
        Returns:
            class_scores = Output probabilities for the 527 classes - numpy array of shape (N,527).
        """
        if not self.model_loaded:
            self.load_pre_trained_model()
        output_tensor = self.classifier_model.output
        input_tensor = self.classifier_model.input
        class_scores = output_tensor.eval(
                            feed_dict={input_tensor: processed_embeddings},
                                        session=self.session_classification)

        return class_scores

    def classify_embeddings_batch(self, vgg_embeddings,batch_size=500):
        """
        Performs classification on PCA Quantized Embeddings.
        Does batching for memory usage and performance reasons.

        Input args:
            vgg_embeddings = numpy array of shape (N,10,128), dtype=float32
        Returns:
            class_scores = Output probabilities for the 527 classes - numpy array of shape (N,527).
        """
        assert(batch_size%EXCERPT_LENGTH==0)

        left_excerpt_length = vgg_embeddings.shape[0]%EXCERPT_LENGTH
        # if total seconds is not divisible by EXCERPT_LENGTH
        # create a second array with few seconds to be processed
        # ! TF version cannot handle segments smaller than 10 seconds
        if ( left_excerpt_length) != 0:
            vgg_embeddings_left = vgg_embeddings[-left_excerpt_length:]
            vgg_embeddings = vgg_embeddings[:-left_excerpt_length]
            # PyTorch
            # vgg_embeddings_left = vgg_embeddings_left.reshape(1,left_excerpt_length,128)
            # TF
            while left_excerpt_length < 10:
                vgg_embeddings_left = np.stack((vgg_embeddings_left, vgg_embeddings_left))
                left_excerpt_length = vgg_embeddings_left.size / 128
            vgg_embeddings_left = vgg_embeddings_left.reshape((int(left_excerpt_length), 128))
            vgg_embeddings_left = vgg_embeddings_left[0:EXCERPT_LENGTH, :].reshape([1, EXCERPT_LENGTH, 128])


        vgg_embeddings = vgg_embeddings.reshape(-1,EXCERPT_LENGTH,128)


        #batch process embeddings
        input_len = vgg_embeddings.shape[0]
        class_scores = np.array([], dtype=np.float32).reshape(0,527)
        for batch_index in range(0,input_len,batch_size):
            a_batch_embeddings=vgg_embeddings[batch_index:batch_index+batch_size]
            a_batch_class_score=self.classify_embeddings(a_batch_embeddings)
            class_scores = np.concatenate((class_scores,a_batch_class_score))

        # classify left embeddings
        if ( left_excerpt_length) != 0:
            a_batch_class_score=self.classify_embeddings(vgg_embeddings_left)
            class_scores = np.concatenate((class_scores,a_batch_class_score))

        return class_scores

    def classify_file(self,pre_processed_npy_file,batch_size=500):
        """
        Calls audioset.classify_embeddings_batch per file from vgg_npy_files.
        Saves classification scores into a file.

        Input args:
            vgg_npy_file : a path to file storing vgg embeddings
        Returns:

        """
        # pre_processed_npy_file:"/scratch/enis/data/nna/NUI_DATA/12 Anaktuvuk/June 2016/ANKTVK_20160621_051133_preprocessed/output026_preprocessed.npy"
        npy_file=Path(pre_processed_npy_file)
        file_index = npy_file.stem.replace("_preprocessed","").replace("output","")
        original_file_stem = npy_file.parent.stem.replace("_preprocessed","")
        vgg_folder = npy_file.parent.parent / npy_file.parent.stem.replace("_preprocessed","_vgg")
        # raw_embeddings_file_path = vgg_folder / (str(original_file_stem) + "_rawembeddings"+file_index+".npy")
        embeddings_file_path =  vgg_folder / (str(original_file_stem) + "_embeddings"+file_index+".npy")

        audioset_folder = Path(str(vgg_folder).replace("_vgg","_audioset"))
        audioset_file_path =  audioset_folder / (str(original_file_stem) + "_audioset"+file_index+".npy")


        # for npy_file in pre_processed_npy_files:
        vgg_embeddings=np.load(embeddings_file_path)
        vgg_embeddings=self.uint8_to_float32(vgg_embeddings)
        classified=self.classify_embeddings_batch(vgg_embeddings,batch_size=batch_size)

        Path(audioset_folder).mkdir(parents=True, exist_ok=True)
        np.save(audioset_file_path,classified)

        # do not delete original vgg file
        # npy_file.unlink()
        return audioset_file_path






    def classify_sound(self,mp3_file,vggish_model=None,batch_size=256):
        """
        Performs classification on mp3 files.
        Input args:
            mp3_file (Path/str) = path to mp3 files
        Returns:
            class_scores = Output probabilities for the 527 classes -
                            numpy array of shape (N,527).
                            N = (mp3 length in seconds) / 10
        """
        if (vggish_model is None) and (self.vggish_model is None):
            self.vggish_model = VggishModelWrapper(sess_config=self.sess_config )
        elif (vggish_model is not None):
            self.vggish_model = vggish_model
        else:
            pass

        sound = pre_process_func.pre_process(mp3_file)
        embeds = self.vggish_model.generate_embeddings(sound,batch_size=batch_size)
        raw_embeddings,post_processed_embed = embeds
        post_processed_embed=post_processed_embed.reshape([-1,EXCERPT_LENGTH,128])
        post_processed_embed=self.uint8_to_float32(post_processed_embed)
        class_probabilities = self.classify_embeddings(post_processed_embed)
        return class_probabilities

    def prob2labels(self,class_probabilities,first_k=1):
        class_prob, class_index  = tf.nn.top_k(class_probabilities, first_k, sorted=True)
        # Torch version is same, but precision is lower. (4 to 8 after .)
        # class_prob, class_index = torch.topk(class_probabilities, first_k,
        #                                     dim=1, largest=True, sorted=True)
        with self.session_classification.as_default():
            class_index = class_index.eval()
            class_prob = class_prob.eval()
        class_labels = [[self.labels[i] for i in sample] for sample in class_index]
        return class_labels,class_prob


    # returns an array with label strings, index of array corresponding to class index
    def load_labels(self,csv_file="assets/class_labels_indices.csv"):
        import csv
        if os.path.exists(csv_file):
            csvfile=open(csv_file, newline='')
            csv_lines=csvfile.readlines()
            csvfile.close()
        else:
            import requests
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

    def uint8_to_float32(self,x):
        return (np.float32(x) - 128.) / 128.























class FC_embed():
    def __init__(self,
                classifier_model_path="assets/mlp_models/bird_FC_089valid_20191122-131739.pth",
                device=None,
                model_loaded=True,
                vggish_model=None):


        if device!=None:
            self.device=torch.device(device)
        else:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.session_classification=None
        self.classifier_model_path = classifier_model_path
        self.model_loaded = model_loaded
        # self.sess_config = sess_config
        self.vggish_model=vggish_model

        # Initialize the vgg-ish embedding model and load post-Processsing
        if model_loaded:
            self.load_pre_trained_model()
        # Metadata
        # self.labels = self.load_labels(labels_path)

    def load_pre_trained_model(self,
                classifier_model_path=None):
        if classifier_model_path==None:
            classifier_model_path=self.classifier_model_path
        # Initialize the Audioset classification model
        from models.FCmodel.mlp_pytorch import MlpNet

        self.classifier_model = MlpNet()
        self.classifier_model.load_state_dict(torch.load(self.classifier_model_path))
        self.classifier_model.eval()
        self.classifier_model.to(self.device)

        # sess = tf.Session(config=self.sess_config)
        # set_session(sess)
        # self.session_classification = tf.keras.backend.get_session()
        # self.classifier_model = tf.keras.models.load_model(classifier_model_path,
                                                            # compile=False)

        self.model_loaded=True

    def classify_embeddings(self, raw_embeddings):
        # """
        # Performs classification on PCA Quantized Embeddings.
        # Input args:
        #     processed_embeddings = numpy array of shape (N,10,128), dtype=float32
        # Returns:
        #     class_scores = Output probabilities for the 527 classes - numpy array of shape (N,527).
        # """
        if not self.model_loaded:
            self.load_pre_trained_model()

        # X_infer=afile[:]
        with torch.no_grad():
            y_pred = self.classifier_model(raw_embeddings)
            y_pred=torch.exp(y_pred[:,1])

        return y_pred


    def classify_file(self,pre_processed_npy_file,batch_size=500):
        # """
        # Calls audioset.classify_embeddings_batch per file from vgg_npy_files.
        # Saves classification scores into a file.
        #
        # Input args:
        #     vgg_npy_file : a path to file storing vgg embeddings
        # Returns:
        #
        # """
        # pre_processed_npy_file:"/scratch/enis/data/nna/NUI_DATA/12 Anaktuvuk/June 2016/ANKTVK_20160621_051133_preprocessed/output026_preprocessed.npy"
        npy_file=Path(pre_processed_npy_file)
        file_index = npy_file.stem.replace("_preprocessed","").replace("output","")
        original_file_stem = npy_file.parent.stem.replace("_preprocessed","")
        vgg_folder = npy_file.parent.parent / npy_file.parent.stem.replace("_preprocessed","_vgg")
        # raw_embeddings_file_path = vgg_folder / (str(original_file_stem) + "_rawembeddings"+file_index+".npy")
        embeddings_file_path =  vgg_folder / (str(original_file_stem) + "_rawembeddings"+file_index+".npy")

        audioset_folder = Path(str(vgg_folder).replace("_vgg","_FCmodel"))
        audioset_file_path =  audioset_folder / (str(original_file_stem) + "_FCmodel"+file_index+".npy")


        # for npy_file in pre_processed_npy_files:
        vgg_embeddings=np.load(embeddings_file_path)
        # vgg_embeddings=self.uint8_to_float32(vgg_embeddings)
        raw_embeddings=torch.from_numpy(vgg_embeddings).to(self.device)

        classified=self.classify_embeddings(raw_embeddings)
        classified=classified.cpu().numpy()

        Path(audioset_folder).mkdir(parents=True, exist_ok=True)
        np.save(audioset_file_path,classified)

        # do not delete original vgg file
        # npy_file.unlink()
        return audioset_file_path




    def classify_sound(self,mp3_file,vggish_model=None,batch_size=256):
        """
        Performs classification on mp3 files.
        Input args:
            mp3_file (Path/str) = path to mp3 files
        Returns:
            class_scores = Output probabilities for the 527 classes -
                            numpy array of shape (N,527).
                            N = (mp3 length in seconds) / 10
        """
        if (vggish_model is None) and (self.vggish_model is None):
            self.vggish_model = VggishModelWrapper(sess_config=self.sess_config )
        elif (vggish_model is not None):
            self.vggish_model = vggish_model
        else:
            pass

        sound = pre_process_func.pre_process(mp3_file)
        embeds = self.vggish_model.generate_embeddings(sound,batch_size=batch_size)
        raw_embeddings,post_processed_embed = embeds
        # post_processed_embed=post_processed_embed.reshape([-1,EXCERPT_LENGTH,128])
        # post_processed_embed=self.uint8_to_float32(post_processed_embed)
        class_probabilities = self.classify_embeddings(raw_embeddings)
        return class_probabilities
