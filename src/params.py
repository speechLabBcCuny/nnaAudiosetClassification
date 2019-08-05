
EXCERPT_LENGTH=10 #seconds


embedding_checkpoint
VGGish_EMBEDDING_CHECKPOINT="assets/vggish_model.ckpt"
PCA_PARAMS="assets/vggish_pca_params.npz"

LABELS="assets/class_labels_indices.csv"

# this path will be used to get relative path of input
# input_file_path.relativeto(INPUT_DIR_PARENT)
# example :
    # if MAIN_INPUT_DIR="a/b/c" and mp3_path ="a/b/c/d/sound.mp3"
    # output will be saved to  Path(output_dir) / "d/sound.npy"
INPUT_DIR_PARENT="/home/data/nna/stinchcomb/NUI_DATA/"
