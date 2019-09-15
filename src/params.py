
EXCERPT_LENGTH=10 #seconds



VGGish_EMBEDDING_CHECKPOINT="assets/vggish_model.ckpt"
PCA_PARAMS="assets/vggish_pca_params.npz"

LABELS="assets/class_labels_indices.csv"

# this path will be used to get relative path of input
# input_file_path.relativeto(INPUT_DIR_PARENT)
# example :
    # if MAIN_INPUT_DIR="a/b/c" and mp3_path ="a/b/c/d/sound.mp3"
    # output will be saved to  Path(output_dir) / "d/sound.npy"
INPUT_DIR_PARENT = "/home/data/nna/stinchcomb/NUI_DATA/"
OUTPUT_DIR = "/scratch/enis/data/nna/NUI_DATA/"

logs_folder="./job_logs/"

#RESOURCES CHECK
ram_memory=100 #GB
segment_length=1 #hour
cpu_count=50
file_per_epoch=70

#1 hour is 0.04
memory_usage=(0.04*30*segment_length*cpu_count)
disk_space=500 #gb
disk_usage= file_per_epoch*2 #gb


### tests
