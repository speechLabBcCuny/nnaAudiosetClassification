# Lab servers:
    dataset @ /home/data/nna/stinchcomb/NUI_DATA/
    Results @ /scratch/enis/data/nna/NUI_DATA/
    Code @ /home/enis/projects/nna
    paths @ /home/enis/projects/nna/mp3paths.txt   [from and to]
    paths @ /home/enis/projects/nna/mp3files.txt   [from]

#Limits:
  500gb
  50 cpu total:56
  120gb ram total:160 ---> 120/(0.04*30)==100
  9gb GPU ram total:12GB (GeForce GTX 1080 Ti)

# code:
  There are 4 pipeline files
  pipe1.py  [divide files into parts] 1 file ---> multiple
  pipe2.py  [pre-process: ] 1 folder of items --->  1 folder of items
  pipe3.py  [VGGish inference] 1 folder of items --->  1 folder of items
  pipe4.py  [Audioset inference] 1 folder of items --->  1 folder of items

# TODO

  Main.py will manage running jobs.


  # find /home/data/nna/stinchcomb/ -name "*.*3" -print0 | xargs -0 python end2end.py --input_files &> endlogs.txt &
