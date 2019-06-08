# Lab servers:
    dataset @ /home/data/nna/stinchcomb/NUI_DATA/
    Results @ /scratch/enis/data/nna/NUI_DATA/
    Code @ /home/enis/projects/nna
    paths @ /home/enis/projects/nna/mp3paths.txt   [from and to]
    paths @ /home/enis/projects/nna/mp3files.txt   [from]


# code:
  There are 4 pipeline files
  pipe1.py  [divide files into parts] 1 file ---> multiple
  pipe2.py  [pre-process: ] 1 folder of items --->  1 folder of items
  pipe3.py  [VGGish inference] 1 folder of items --->  1 folder of items
  pipe4.py  [Audioset inference] 1 folder of items --->  1 folder of items

# TODO
  Add pipe2, pipe3, pipe4 to 1 file to 1 file capability
  Main.py will manage running jobs.
  
