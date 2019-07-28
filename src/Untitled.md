while IFS="" read -r p || [ -n "$p" ]
do
  echo '%s\n' "$p"
done < mp3files.txt

head mp3files.txt --lines 1 | while read line
do
   echo $line
   python pipe1.py --input_files $line --abs_input_path "/home/data/nna/stinchcomb/NUI_DATA/" --output_folder "/scratch/enis/test/" --segment_len "01:00:00" &> endlogs.txt &&
   
done

python pipe1.py --input_files "/scratch/ebc327/nna/test/01 Itkillik/August 2016/shortfile.mp3" --abs_input_path "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --output_folder "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --segment_len "00:05:00" &> endlogs.txt &


preds_index=abs_file_path+"_preds_index.npy"
rawembeddings=abs_file_path+"_rawembeddings.npy"
embeddings=abs_file_path+"_embeddings.npy"
preds=abs_file_path+"_preds.npy"
