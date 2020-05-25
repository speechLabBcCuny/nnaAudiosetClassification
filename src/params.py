from pathlib import Path

EXCERPT_LENGTH=10 #seconds



VGGish_EMBEDDING_CHECKPOINT="assets/vggish_model.ckpt"
PCA_PARAMS="assets/vggish_pca_params.npz"

LABELS="assets/class_labels_indices.csv"

# this path will be used to get relative path of input
# input_file_path.relativeto(INPUT_DIR_PARENT)
# example :
    # if MAIN_INPUT_DIR="a/b/c" and mp3_path ="a/b/c/d/sound.mp3"
    # output will be saved to  Path(output_dir) / "d/sound.npy"
INPUT_DIR_PARENT = "/tank/data/nna/real/"
OUTPUT_DIR = "/scratch/enis/data/nna/real/"
INPUT_LIST = "/home/enis/projects/nna/data/ExperimentRunV6.txt"


LOGS_FOLDER="./job_logs/"
LOGS_FILE = LOGS_FOLDER + "logs.txt"
if not Path(LOGS_FOLDER).exists():
    # os.mkdir(segments_folder)
    Path(LOGS_FOLDER).mkdir(parents=True, exist_ok=True)

# currently being run on a pre-process, (original big file)
PRE_PROCESSING_queue = LOGS_FOLDER + "pre_processing_queue.csv"
# output of Pre process, input for VGG (segmented small file)
PRE_PROCESSED_queue = LOGS_FOLDER + "pre_processed_queue.csv"
# currently being run on a VGG (segmented small file)
VGGISH_processing_queue = LOGS_FOLDER + "VGGISH_processing_queue.csv"
# output of VGGISH, input for Audioset classifier (segmented small file)
VGGISH_EMBEDDINGS_queue = LOGS_FOLDER + "vggish_embeddings_queue.csv"
# currently being run on a audioset (segmented small file)
Audioset_processing_queue = LOGS_FOLDER + "Audioset_processing_queue.csv"
# output of audioset (segmented small file)
Audioset_output_queue = LOGS_FOLDER + "Audioset_output_queue.csv"


#RESOURCES CHECK
# available resources:
ram_memory=100 #GB
disk_space=300 #gb

segment_length=1 #hour
# CPU count determines, how many file is processed in parallel
# each CPU processes as 49 hour file in, 1 hour at a time in memory
# cpu_count=7
cpu_count=5

#1 hour is 0.04
#30x for mp3 to wav
memory_usage=(0.04*30*segment_length*cpu_count)
assert( memory_usage<ram_memory)

#1.62 mb per minute for preprocessed.npy
#(4.74GB for 50 hours of 2gb mp3)
# we delete preprocessed.npy after VGGish,
# we run VGGish after all CPUs done with single 50 hour file
disk_usage = 4.74 * cpu_count
assert( disk_usage<=disk_space)

# after VGGish, raw_embeddings+ embeedings is 28x smaller than original mp3
