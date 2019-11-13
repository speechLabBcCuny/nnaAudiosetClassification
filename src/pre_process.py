from params import INPUT_DIR_PARENT,OUTPUT_DIR
from params import cpu_count,LOGS_FOLDER,LOGS_FILE
from params import PRE_PROCESSED_queue,VGGISH_EMBEDDINGS_queue
from pathlib import Path

import pre_process_func


# create a list of paths for input
with open("/home/enis/projects/nna/data/flacfiles_left.txt","r+") as my_file:
    input_path_list = my_file.read().splitlines()

    # process each mp3 file to generate input for VVGish and save them locally
pre_process_func.parallel_pre_process(input_path_list,
                                        output_dir=OUTPUT_DIR,
                                        input_dir_parent=INPUT_DIR_PARENT,
                                        cpu_count=cpu_count,
                                        segment_len="03:00:00",
                                        logs_file_path=LOGS_FILE)
