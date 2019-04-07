import os
import argparse
from pydub import AudioSegment

# import tensorflow as tf
# import numpy as np
# from scipy.io.wavfile import read

def mp3towav(mp3_data_files,output_folder):
    for i,mp3_file_path in enumerate(mp3_data_files):
        sound = AudioSegment.from_mp3(mp3_file_path)
        mp3_file_name=os.path.basename(mp3_file_path)
        mp3_file_name=mp3_file_name.split(".")[0]
        wav_file_name=mp3_file_name+".wav"
        wav_file_path=os.path.join(output_folder,wav_file_name)
        sound.export(wav_file_path, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mp3 to wav files')
    parser.add_argument('--output_folder', type=str,
                       help='output folder')
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    mp3towav(args.input_files,args.output_folder)
