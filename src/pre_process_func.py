import os
from pydub import AudioSegment
import sys
from pathlib import Path
from time import gmtime, strftime
from subprocess import Popen, PIPE


sys.path.insert(0, './models/audioset')
from vggish_params import EXAMPLE_HOP_SECONDS
import vggish_input
# import vggish_postprocess

import shutil

import numpy as np
import subprocess
from params import EXCERPT_LENGTH,INPUT_DIR_PARENT,OUTPUT_DIR

def rmv_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

def relative2outputdir(mp3_file_path,output_dir=OUTPUT_DIR,
                                            input_dir_parent=INPUT_DIR_PARENT):
    """Returns output directory for a given mp3 file.

    Given absolute mp3 file path, finds relative path to INPUT_DIR_PARENT
    and creates absolute path of output directory

    Args:
        mp3_file_path (str/Path): absolute path to a file.
        output_folder (str/Path): root of output directory

    Returns:
        Path : where resulting .npy files will be saved
    """
    # INPUT_DIR_PARENT= /foo/foo/
    # mp3_file_path ="/foo/foo/18 Fish Creek 4/July 2016/my_music.mp3"
    # relative_path='18 Fish Creek 4/July 2016'
    mp3_file_path=Path(mp3_file_path)
    relative_path =   mp3_file_path.relative_to(input_dir_parent).parent
    # absolute_output_dir= '/output/directory/18 Fish Creek 4/July 2016/
    absolute_output_dir = output_dir / relative_path
    return (absolute_output_dir)

# create directories given set of file paths
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def preb_names(mp3_file_path,output_dicretory=OUTPUT_DIR,abs_input_path=INPUT_DIR_PARENT):
    mp3_file_path=Path(mp3_file_path)
    absolute_output_dir=relative2outputdir(mp3_file_path,output_dicretory,
                                                abs_input_path)
    segments_folder = absolute_output_dir.joinpath(mp3_file_path.stem+"_segments/")
    embeddings_file_name = absolute_output_dir.joinpath(mp3_file_path.stem+"_embeddings.npy")
    pre_processed_folder= absolute_output_dir.joinpath(mp3_file_path.stem+"_preprocessed/")

    return segments_folder,embeddings_file_name,pre_processed_folder

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
        # os.mkdir(segments_folder)
        Path(segments_folder).mkdir(parents=True, exist_ok=True)
    #TODO handle stderror
    sp = subprocess.run(['ffmpeg','-y','-i',mp3_file_path,"-c","copy","-map","0",
                                "-segment_time", segment_len, "-f", "segment",
                                str(segments_folder)+"/output%03d.mp3"],
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

def cal_sample_size(wav_data,sr):
    """Cal. sample size from log mel spectogram for a given wav_file
        read **(16/06/2019)** at Project_logs.md for explanations.

        Args:
            wav_data (numpy.array): audio data in wav format
            sr (int): sampling rate of the audio

        Returns:
            [int,int,int]
    """
    assert (EXAMPLE_HOP_SECONDS==0.96)
    offset=sr*EXCERPT_LENGTH
    #EXPECTED sample size, after processing
    sample_size=(len(wav_data)//offset)*EXCERPT_LENGTH
    remainder_wav_data=len(wav_data)%offset
    # if remainder will generate more than any samples (requires 42998 numbers)
    if remainder_wav_data<42998:
        pass
    elif remainder_wav_data>=42998 and remainder_wav_data<85334:
        sample_size+=1
    # elif remainder_wav_data>=85334:
    else:
        # sample size increases for each:
            #20286 is for 0.46 hop_size, 42336 is for hop_size 0.96
        # sample_size+=((remainder_wav_data-22712)//42336)
        sample_size+=((remainder_wav_data - 85334)//42336) + 2

    return sample_size,offset,remainder_wav_data

def iterate_for_waveform_to_examples(wav_data,sr):
    """Wrapper for waveform_to_examples from models/audioset/vggish_input.py

        Iterate over data with 10 seconds batches, so waveform_to_examples produces
        stable results (equal size)
        read **(16/06/2019)** at Project_logs.md for explanations.

        Args:
            wav_data (numpy.array): audio data in wav format
            sr (int): sampling rate of the audio

        Returns:
            See waveform_to_examples.
    """
    sample_size,offset,remainder_wav_data=cal_sample_size(wav_data,sr)
    # in this loop wav_data jumps offset elements and sound jumps EXCERPT_LENGTH*2
    # because offset number of raw data turns into EXCERPT_LENGTH*2 pre-processed
    sound=np.zeros((sample_size,96,64),dtype=np.float32)
    count=0
    for i in range(0,len(wav_data),offset):
    #this is when wav_data%offset!=0
        if i+offset>len(wav_data) and remainder_wav_data<42998:
            continue
        # left data is smaller than 22712, we cannot pre-process
        # if smaller than 42998, will be 0 anyway
        a_sound= vggish_input.waveform_to_examples(wav_data[i:i+(offset)], sr)
        sound[count:(count+a_sound.shape[0]),:,:]=a_sound[:,:,:]
        count+=a_sound.shape[0]
    return sound

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
    sound = iterate_for_waveform_to_examples(samples,sr)
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

    npy_file_path = Path(output_dir) / (str(mp3_file_path.stem) + "_preprocessed.npy")

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
    # divide files ! should end with "/"
    segments_folder = Path(output_dir) / (mp3_file_path.stem + "_segments/")
    mp3_segments=divide_mp3(mp3_file_path,segments_folder,
                            segment_len=segment_len)
    # pre-process each segment
    pre_processed_dir = Path(output_dir) / (mp3_file_path.stem + "_preprocessed")
    if not os.path.exists(pre_processed_dir):
        pre_processed_dir.mkdir(parents=True, exist_ok=True)

    for mp3_segment in mp3_segments:
        # # if pre-processed
        mp3_segment_path=segments_folder / mp3_segment
        pre_process(mp3_segment_path,output_dir=pre_processed_dir,
                    saveAsFile=True)
    rmv_folder(segments_folder)


def parallel_pre_process(input_path_list,output_dir=OUTPUT_DIR,
                        input_dir_parent=INPUT_DIR_PARENT,
                        cpu_count=50,segment_len="01:00:00",
                        logs_file_path="logs.txt"):
    """Call pre_process with a seperate cpu process per file

    This function calls pre_process_big_file for each file within a new process
    takes advantage of gnu Parallel keeping Cpus occupied

    Args:
        input_path_list (List[str]): List of absolute paths to mp3 files
        output_dir (str/Path): where resulting .npy files will be saved
        segment_len (str): length of segments to generate, HOURS:MM:SS

    Returns:
        None
    """
    conda_env_name="speechEnv"

    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uuid_time=strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    uuid_time="test"
    tmp_input_file = output_dir / ("input_" + uuid_time + ".txt")
    tmp_input_file=Path(tmp_input_file)
    tmp_input_file.touch(exist_ok=True)
    # write mp3 file path  and relative output path to a file
    with open(tmp_input_file, 'w') as f:
        for mp3_file_path in input_path_list:
            relative2output_dir = relative2outputdir(mp3_file_path,
                                                    output_dir,input_dir_parent)
            line="{}\t{}\n".format(mp3_file_path,relative2output_dir)
            k=f.write(line)

    # # #DO NOT put new line python code
    python_code=("from pre_process_func import pre_process_big_file;"
                + "pre_process_big_file('{}'.split('\t')[0],"#{} for GNU parallel
                + "output_dir='{}'.split('\t')[1],"
                + "segment_len='{}');print('test')".format(segment_len))

    command_list=[]
    # command_text=""
    command_list.extend(["conda","run","-n",conda_env_name])
    command_list.extend(["cat",str(tmp_input_file),"|"])
    command_list.extend(["parallel","-P",str(cpu_count),"-n","1","-q"])

    # command_list.extend(["echo","{}"])
    # command_list.extend(["python", "call2func.py", "{1}", "{2}","2>>","logs.txt"])
    command_list.extend(["python", "-c", python_code])

    process = Popen(command_list, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout, stderr)
    sys.stdout.flush()
    sys.stderr.flush()

    tmp_input_file.unlink()
