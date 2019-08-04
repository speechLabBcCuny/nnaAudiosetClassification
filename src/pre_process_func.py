import os
from pydub import AudioSegment
import sys
from pathlib import Path

sys.path.insert(0, './audioset')
# import vggish_slim
import vggish_input
# import vggish_postprocess

import shutil

import numpy as np
import subprocess
from params import EXCERPT_LENGTH

def rmv_segmets(segments_folder):
    try:
        shutil.rmtree(segments_folder)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))


# create directories given set of file paths
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

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
    pre_processed_folder=os.path.join(output_dicretory,path_to_file,mp3_file_name+"_preprocessed/")
    return mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder

# gs://deep_learning_enis/speech_audio_understanding/nna/test/
def divide_mp3(mp3_file_path,segments_folder,segment_len="01:00:00"):
    """Convenience wrapper around ffmpeg segmenting for a common mp3 format.

    Note that splitting may not be accurate. We do not force the reference
    stream key-frames at the given time.
    Creates segments folder if it does not exists already.

    Args:
        mp3_file_path (str/Path): Path to the file is assumed to contain
            mp3 audio data.
        segments_folder (str/Path): folder to save segments of the mp3
        segment_len (str): length of segments to generate, HOURS:MM:SS
    Returns:
        List of file path strings to segments.
    """
    if not os.path.exists(segments_folder):
        os.mkdir(segments_folder)
    #TODO handle stderror
    sp = subprocess.run(['ffmpeg','-y','-i',mp3_file_path,"-c","copy","-map","0",
                                "-segment_time", segment_len, "-f", "segment",
                                segments_folder+"output%03d.mp3"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    mp3_segments=os.listdir(segments_folder)
    return mp3_segments


def load_mp3(input_file_path):
    """ Loads mp3 and returns an array within wav data format

    waveform_to_examples is expecting `np.int16`,
    original code is using soundfile which does support wav files.
    so we load mp3 files as wav
    other ways and performance comparison
    https://colab.research.google.com/drive/1VyZIWY-3_xe8IsqZwlPzaD1nGFyRI6rX

    Args:
        input_file_path (str/Path): Path to the file is assumed to contain
            mp3 audio data.
    Returns:
        A tuple (wav_data, sampling_rate)
    """
    input_file_path=Path(input_file_path)

    wav_data = AudioSegment.from_mp3(input_file_path)
    sr=wav_data.frame_rate
    # waveform_to_examples already takes mean of two channels
    # wav_data=wav_data.set_channels(1)
    wav_data = wav_data.get_array_of_samples()
    wav_data = np.array(wav_data)
    wav_data=wav_data.reshape(-1,2)
    return wav_data,sr

def mp3file_to_examples(mp3_file_path):
    """Wrapper around iterate_for_waveform_to_examples() for a common mp3 format.

    Args:
        mp3_file_path (str/Path): String path to a file. The file is assumed to contain
            mp3 audio data.

    Returns:
        See iterate_for_waveform_to_examples.
    """
    wav_data,sr=load_mp3(mp3_file_path)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    #######iterate over 10 seconds#########
    return iterate_for_waveform_to_examples(samples,sr)


def iterate_for_waveform_to_examples(wav_data,sr):
    """Wrapper for waveform_to_examples from models/audioset/vggish_input.py

        Iterate over data with 10 second batches, so waveform_to_examples produces
        stable results (equal size)
        read **(16/06/2019)** at Project_logs.md for explanations.

        Args:
            wav_data (numpy.array): audio data in wav format
            sr (int): sampling rate of the audio

        Returns:
            See waveform_to_examples.
    """
    offset=sr*EXCERPT_LENGTH
    #EXPECTED sample size, after processing
    sample_size=(len(wav_data)//offset)*20
    remainder_wav_data=len(wav_data)%offset
    if remainder_wav_data>=42998:
        sample_size+=((remainder_wav_data-22712)//20286)
    # in this loop wav_data jumps offset elements and sound jumps EXCERPT_LENGTH*2
    # because offset number of raw data turns into EXCERPT_LENGTH*2 pre-processed
    sound=np.zeros((sample_size,96,64),dtype=np.float32)
    count=0
    for i in range(0,len(wav_data),offset):
    #this is when wav_data%offset!=0
        if i+offset>len(wav_data):
            # left data is smaller than 22712, we cannot pre-process
            # if smaller than 42998, will be 0 anyway
            if remainder_wav_data<42998:
                continue
            a_sound= vggish_input.waveform_to_examples(wav_data[i:i+(offset)], sr)
            sound[count:(count+a_sound.shape[0]),:,:]=a_sound[:,:,:]
            count+=a_sound.shape[0]
        else:
            a_sound= vggish_input.waveform_to_examples(wav_data[i:i+(offset)], sr)
            sound[count:(count+a_sound.shape[0]),:,:]=a_sound[:,:,:]
            count+=a_sound.shape[0]
    return sound

def pre_process(mp3_file_path,output_dir="./", saveAsFile=False):
    """Wrapper for mp3file_to_examples, handles input and output logic

        Saves as a file called mp3_file_name_preprocessed.npy in output_dir
        If output npy file already exists returns None

        Args:
            mp3_file_path (numpy.array): audio data in wav format
            output_dir (str/Path): output directory
            saveAsFile (bool): save as file or not
            sr (int): sampling rate of the audio

        Returns:
            Returns pre_processed sound (numpy.array) if file does not exists
    """
    mp3_file_path = Path(mp3_file_path)
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    npy_file_path = Path(output_dir) / mp3_file_path.stem + "_preprocessed.npy"

    if os.path.exists(npy_file_path):
        return None

    sound = mp3file_to_examples(mp3_file_path)
    sound = sound.astype(np.float32)

    if saveAsFile:
        np.save(npy_file_path,sound)

    return sound

# this function will replace pipe_pre.py
def pre_process_big_file(mp3_file_path,output_dir="./",segment_len="01:00:00"):
    """Divides into segments and calls pre_process on each segment.

    This function divides mp3 files into segments (default: 1 hour)
    Calls pre_process on each segment with saveAsFile=True
    saves each segment file into output_dir/mp3_file_name_preprocessed
    If resulting .npy files exists, does not re-compute them.

    Args:
        mp3_file_path (str/Path): String path to a file.
            The file is assumed to contain mp3 audio data.
        output_folder (str/Path): where resulting .npy files will be saved
        segment_len (str): length of segments to generate, HOURS:MM:SS

    Returns:
        None
    """
    mp3_file_path = Path(mp3_file_path)
    # divide files
    segments_folder = Path(output_dir) / (mp3_file_path.stem + "_segments/")
    mp3_segments=divide_mp3(mp3_file_path,segments_folder,
                            segment_len=segment_len)
    # pre-process each segment
    pre_processed_dir = Path(output_dir) / (mp3_file_path.stem + "_preprocessed")
    if not os.path.exists(pre_processed_folder):
        os.mkdir(pre_processed_folder)

    for mp3_segment in mp3_segments:
        # # if pre-processed
        pre_process(mp3_segment_path,output_dir=pre_processed_folder,
                    saveAsFile=True)
    rmv_segmets(segments_folder)
