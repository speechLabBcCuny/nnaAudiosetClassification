### test
python process_audio.py --input_files /scratch/mim/nna/samples/split_60:00/CLVL5_20160621_074419_0000m_00s__0060m_00s.mp3 --output_folder $wav_folder

### 1) turn into wav files (24 hour example)
wav_folder=/scratch/enis/data/nna/samples_wav/24_hour/

ls -d -1 /scratch/mim/nna/samples/split_60\:00/* | sort -k9,9 | head --lines 24 | parallel -P 20 -n 1 'python process_audio.py --input_files {} --output_folder scratch/enis/data/nna/samples_wav/24_hour/'

cd ./projects/nna/
rm -rf datas

### test
python pre_process.py --input_files "$wav_folder" USGS_20160613_234225_2820m_00s__2880m_00s_56m_00s__58m_00s.wav --output_folder ./datas --job_id 0

### 2) generate log-mel
ls -d -1 "$wav_folder"*  | parallel -P 24 -n 1 'python pre_process.py --input_files {} --output_folder ./data --job_id {#}'

### 3) run merge_preprocessed.py
python merge_prepropecessed.py

### 4) run pre_process2.py for inference, probably on cloud
rsync -avz --progress Momentsnotice:/home/enis/projects/nna/data/ /Users/berk/Desktop/data/
you can use "nature sounds, inference with VGGish.ipynb"
rsync -avz --progress Momentsnotice:/home/enis/projects/nna/download/ /Users/berk/Desktop/download/

gsutil
gsutil cp "/home/data/nna/stinchcomb/NUI_DATA/07 Ice Rd/August 2016/ICERD_20160808_024504.MP3" gs://deep_learning_enis/speech_audio_understanding/nna/test/
gsutil init
gsutil config

gsutil -m cp -r "/scratch/enis/data/nna/NUI_DATA/" gs://deep_learning_enis/speech_audio_understanding/nna/ &>> upload_logs.txt &
