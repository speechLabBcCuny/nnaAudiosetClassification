
rsync -a --include='*/' --exclude='*' /home/data/nna/stinchcomb/NUI_DATA/ /scratch/enis/data/nna/wav_files/

find /home/data/nna/stinchcomb/ -name "*.*3" | wc -l
409


find /home/data/nna/stinchcomb/ -name "*.*3" | parallel -P 20 -n 20 'python process_audio.py --abs_input_path /home/data/nna/stinchcomb/NUI_DATA/ --input_files {} --output_folder /scratch/enis/data/nna/wav_files/'


# start with screen
# get GPU Memory so I can use it later
* one file file at a time (~2 gb)
  * I will divide files to 1 hour slices
  * I will use 25 cpus so 25 processes
  * embeddings file name will be filename_embeddings.
