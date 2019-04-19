import os
import argparse
from pydub import AudioSegment

# import tensorflow as tf
# import numpy as np
# from scipy.io.wavfile import read

def mp3towav(mp3_data_files,output_folder,abs_input_path):
    abs_len=len(abs_input_path)
    for i,mp3_file_path in enumerate(mp3_data_files):
        if not mp3_file_path[:abs_len]==abs_input_path:
            print("Error with abs_input_path {}, got {}".format(abs_input_path,
                                                                mp3_file_path[:abs_len]))
            return False
        # relative_path='03 Ocean Pt/June 2016/OCNPT3_20160619_041316.MP3'
        relative_path=mp3_file_path[abs_len:]
        sound = AudioSegment.from_mp3(mp3_file_path)
        mp3_file_name=os.path.basename(mp3_file_path)
        path_to_file=relative_path[:-len(mp3_file_name)]
        mp3_file_name=mp3_file_name.split(".")[0]
        wav_file_name=mp3_file_name+".wav"
        wav_file_path=os.path.join(output_folder,path_to_file,wav_file_name)
        sound.export(wav_file_path, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mp3 to wav files')
    parser.add_argument('--output_folder', type=str,
                       help='output folder')
    parser.add_argument('--abs_input_path', type=str,
                       help='absoulute input folder such as /home/data/nna/stinchcomb/NUI_DATA/')
    parser.add_argument('--input_files',nargs='+',type=str,default=None)
    # parser.add_argument('--input_folder',nargs=1,type=str,default=None)
    args = parser.parse_args()
    # args.output_folder= "/scratch/enis/data/nna/wav_files/"
    # args.abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
    # args.input_files=[" "," "," "]
    mp3towav(args.input_files,args.output_folder,args.abs_input_path)
