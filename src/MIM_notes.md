# MIM's notes on getting this running

```bash
# First: Need to download VGGish files
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt

# Then download audioset classifier files
curl https://storage.googleapis.com/deep_learning_enis/speech_audio_understanding/Eusipco2018_Google_AudioSet/md_50000_iters.tar --output ./md_50000_iters.tar
curl https://gist.githubusercontent.com/EnisBerk/19d817e9a6a9c060465a5b95c8d54d97/raw/8e0c24e92853e779e222bc3a152d7d65351cfe52/model.py --output ./model.py

# Need these packages
pip install -r Requirements.txt  # only contains pydub at the moment
pip install torch
pip install tensorflow-gpu==1.12  # Last version supporting CUDA 9.0
pip install soundfile

# Run from ./src/ directory
# Edit paths in pipe?.py files to correspond to this

# Run processing steps.  
# Note: need full path to input file(s) even though directory is also 
# provided as another argument

# Step 1: chop long mp3 into segments using ffmpeg
python pipe1.py \
    --abs_input_path /scratch/mim/nna/examples/ \
    --output_folder /scratch/mim/nna/examples_out/ \
    --input_files /scratch/mim/nna/examples/NIGLIQ1_20160618_061015.mp3

# Step 2: extract mel spectrograms
python pipe2.py \
    --abs_input_path /scratch/mim/nna/examples/ \
    --output_folder /scratch/mim/nna/examples_out/ \
    --input_files /scratch/mim/nna/examples/NIGLIQ1_20160618_061015.mp3

# Step 3: run VGGish and post-process
python pipe3.py \
    --abs_input_path /scratch/mim/nna/examples/ \
    --output_folder /scratch/mim/nna/examples_out/ \
    --input_files /scratch/mim/nna/examples/NIGLIQ1_20160618_061015.mp3

# Step 4: run audioset classifier
python pipe4.py \
    --abs_input_path /scratch/mim/nna/examples/ \
    --output_folder /scratch/mim/nna/examples_out/ \
    --input_files /scratch/mim/nna/examples/NIGLIQ1_20160618_061015.mp3
```
