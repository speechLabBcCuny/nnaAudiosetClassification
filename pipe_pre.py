from pre_process_func import *
from time import time
import argparse
import sys
import numpy as np

# This one splits mp3 files into smaller chunks, wrapper for ffmpeg
# find /home/data/nna/stinchcomb/ -name "*.*3" -print0 | xargs -0 python end2end.py --input_files &> endlogs.txt &
if __name__ == "__main__":

    # mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
    # output_dicretory="/scratch/enis/data/nna/wav_files/"
    # abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
    parser = argparse.ArgumentParser(description='mp3 to embeddings')
    parser.add_argument('--output_folder', type=str,
                       help='output folder',default="/scratch/enis/data/nna/NUI_DATA/")
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

    input_files.sort()
    # print(input_files)
    for i,input_file in enumerate(input_files):
        start=time()
        print("{} - file: {}".format(i,input_file))
        sys.stdout.flush()
        ##### step - 0 get prepare names
        mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder=preb_names(input_file,
                                              output_folder,abs_input_path)
        # #if inference is done
        if os.path.exists(embeddings_file_name):
            continue

        ##### step 1 -  divide files into parts
        mp3_segments=divide_mp3(mp3_file_path,segments_folder,segment_len=args.segment_len)
        # #### step 2 - pre-process
        mp3_segments=os.listdir(segments_folder)
        if not os.path.exists(pre_processed_folder):
            os.mkdir(pre_processed_folder)
        for mp3_segment in mp3_segments:
            # # if pre-processed
            if os.path.exists(pre_processed_folder+mp3_segment[:-4]+".npy"):
                continue
            pre_process(mp3_segment,segments_folder,pre_processed_folder,saveNoReturn=True)
        rmv_segmets(segments_folder)
        # #### step 3 - inference
        # tmp_npy=mp3_file_path[:-4]+".npy"
        # sounds=np.load(tmp_npy)
        # embeddings_file_name=inference(sounds,vgg,sess,embeddings_file_name,batch_size=256)
        end=time()
        print("It took {} seconds".format(end-start))
        sys.stdout.flush()
