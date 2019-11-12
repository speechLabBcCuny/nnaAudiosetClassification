```
rsync -a --include='*/' --exclude='*' /home/data/nna/stinchcomb/NUI_DATA/ /scratch/enis/data/nna/wav_files/

find /home/data/nna/stinchcomb/ -name "*.*3" | wc -l
409
```
```
find /home/data/nna/stinchcomb/ -name "*.*3" | parallel -P 20 -n 20 'python process_audio.py --abs_input_path /home/data/nna/stinchcomb/NUI_DATA/ --input_files {} --output_folder /scratch/enis/data/nna/wav_files/'
```

# start with screen
# get GPU Memory so I can use it later
* one file file at a time (~2 gb)
  * I will divide files to 1 hour slices
  * I will use 25 cpus so 25 processes
  * embeddings file name will be filename_embeddings.



1) create a text file with paths to input audio files (one full path per line)
    `find /tank/data/nna/real/ -iname "*flac" > flacfiles.txt`
2) update input and output directory in params
3) sync files `rsync -av --recursive --update ./ enis@crescent:/home/enis/projects/nna/ `
3) run codes
```
python pre_process.py &>> job_logs/logs.txt; python slack_message.py -t cpu_job &
python watch_VGGish.py &>> job_logs/logs.txt; python slack_message.py &
```
4) re-run
`rsync -av --recursive --update /Users/berk/Documents/workspace/speech_audio_understanding/src/ enis@crescent:/home/enis/projects/nna/`
`find /scratch/enis/data/nna/real/ -iname "*flac"  -delete`
5) tracking progress and backup
```
cat job_logs/pre_processing_queue.csv | wc -l; cat job_logs/pre_processed_queue.csv | wc -l; cat job_logs/VGGISH_processing_queue.csv | wc -l; cat job_logs/vggish_embeddings_queue.csv | wc -l; du -hs /scratch/enis/data/
total segment count is 18908
tar cf - /scratch/enis/data/nna/backup/NUI_DATA/ -P | pv -s $(du -sb /scratch/enis/data/nna/backup/NUI_DATA/ | awk '{print $1}') | gzip > embeddings_backup.tar.gz
```

-1) rsync server
```bash
rsync -uzarv  --prune-empty-dirs --include "*/"  \
--include="*.ipynb" --include="*.py" --include="*.md" --exclude="*" \
/Users/berk/Documents/workspace/speech_audio_understanding/ \
enis@crescent:/home/enis/projects/nna/ && \
rsync -uzarv  --prune-empty-dirs --include "*/"  \
--include="*.ipynb" --include="*.py" --include="*.md" --exclude="*" \
enis@crescent:/home/enis/projects/nna/ \
/Users/berk/Documents/workspace/speech_audio_understanding/ \
```
