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
  Main.py will manage running jobs.

sample file: /scratch/enis/data/nna/NUI_DATA/02 Colville 2/August 2016/


#run:
 * update paths, resources in main,py
 * clean output folder such as "/scratch/enis/data/nna/NUI_DATA/"
 * copy main.py, pre_process_func.py, vggish_params.py
    #rm main.py && vim  main.py
    #rm pre_process_func.py && vim  pre_process_func.py
    #rm vggish_params.py && vim  vggish_params.py

 * python main.py
