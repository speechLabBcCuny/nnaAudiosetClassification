import os
txt=open("mp3files.txt")
mp3_data_files=txt.readlines()
mp3_data_files=[line.strip() for line in mp3_data_files]
abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
output_folder= "/scratch/enis/data/nna/wav_files/"
abs_len=len(abs_input_path)
out=open("mp3paths.txt","w")
for i,mp3_file_path in enumerate(mp3_data_files):
    if not mp3_file_path[:abs_len]==abs_input_path:
        print("Error with abs_input_path {}, got {}".format(abs_input_path,
                                                            mp3_file_path[:abs_len]))
    # relative_path='03 Ocean Pt/June 2016/OCNPT3_20160619_041316.MP3'
    relative_path=mp3_file_path[abs_len:]
    mp3_file_name=os.path.basename(mp3_file_path)
    path_to_file=relative_path[:-len(mp3_file_name)]
    mp3_file_name=mp3_file_name.split(".")[0]
    wav_file_name=mp3_file_name+".wav"
    wav_file_path=os.path.join(output_folder,path_to_file,wav_file_name)
    line=mp3_file_path +"\t"+wav_file_path+"\n"
    a=out.write(line)

mp3file="/scratch/mim/nna/samples/split_60:00/CLVL5_20160621_074419_0000m_00s__0060m_00s.mp3"
mp3_long="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160627_180821.MP3"
import soundfile as sf
from pydub import AudioSegment
from time import time

# def ops(mp3file):
#     time1=time()
#     sound = AudioSegment.from_mp3(mp3file)
#     time2=time()
#     print(time2-time1)
#
# ops(mp3_long)
my_file=open("tmp.wav","w")
sound = AudioSegment.from_mp3(mp3file)
sound.export(my_file, format="wav")
wav_data, sr = sf.read(samples, dtype='int16')
wav_data, sr = sf.read(mp3file)

ffmpeg -i /scratch/mim/nna/samples/split_60:00/CLVL5_20160621_074419_0000m_00s__0060m_00s.mp3 -c copy -map 0 -segment_time 00:30:00 -f segment ./new/output%03d.mp3

import subprocess
from io import BytesIO
import copy
import soundfile as sf
from pydub import AudioSegment
from time import time

import sys
sys.path.insert(0, '../audioset')

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess
import numpy as np

mp3file="/scratch/mim/nna/samples/split_60:00/CLVL5_20160621_074419_0000m_00s__0060m_00s.mp3"
output_dicretory="/scratch/enis/data/nna/wav_files/"

# mkdir new here
sp = subprocess.Popen(['ffmpeg','-i',mp3file,"-c","copy","-map","0",
                                "-segment_time", "00:30:00", "-f", "segment",
                                "new/output%03d.mp3"])


sounds=[]
def run():
    for i in range(3):
        mp3file="new/output00{}.mp3".format(i)
        sp = subprocess.Popen(
            ['ffmpeg', '-i', mp3file, '-f', 'wav', '-'],shell=False,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        bio = BytesIO(sp.stdout.read())
        wav_data, sr = sf.read(bio, dtype='int16')
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        sound= vggish_input.waveform_to_examples(samples, sr)
        sounds.append(sound)
