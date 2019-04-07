import os
import numpy as np

data_dir="data/"
files=os.listdir(data_dir)

file_count=len(files)/2
# all_data = np.array([], dtype=np.float64).reshape(0,96,64)

total_data_len=0
for i in range(1,int(file_count)+1):
  indexes=data_dir+"data_file_indexes_"+str(i)+".npy"
  data=data_dir+"postprocessed_"+str(i)+".npy"
  data_file_indexes=np.load(indexes)
  postprocessed=np.load(data)
  #get dict from numpy array
  data_file_indexes=data_file_indexes[()]
  for wav_file in data_file_indexes.keys():
    index_tuple=data_file_indexes[wav_file]
    # all_data=np.concatenate((all_data,postprocessed[index_tuple[0]:index_tuple[1]]))
    length=index_tuple[1]-index_tuple[0]
    # all_index[wav_file]=(counter,counter+length)
    total_data_len+=length

all_index={}
# all_data = np.array([], dtype=np.int16).reshape(total_data_len,96,64)
all_data=np.zeros((total_data_len,96,64), dtype=np.int16)

counter=0
for i in range(1,int(file_count)+1):
  indexes=data_dir+"data_file_indexes_"+str(i)+".npy"
  data=data_dir+"postprocessed_"+str(i)+".npy"
  data_file_indexes=np.load(indexes)
  postprocessed=np.load(data)
  #get dict from numpy array
  data_file_indexes=data_file_indexes[()]
  for wav_file in data_file_indexes.keys():
    index_tuple=data_file_indexes[wav_file]
    # all_data=np.concatenate((all_data,postprocessed[index_tuple[0]:index_tuple[1]]))
    # print(counter,counter+length,index_tuple[0],index_tuple[1])
    length=index_tuple[1]-index_tuple[0]
    all_data[counter:counter+length]=np.copy(postprocessed[index_tuple[0]:index_tuple[1]])
    all_index[wav_file]=(counter,counter+length)
    counter+=length

np.save(os.path.join(data_dir,"data_file_indexes.npy"),all_index)
# np.save(os.path.join(args.output_folder,"embeddings_"+args.job_id+".npy"),embeddings)
np.save(os.path.join(data_dir,"preprocessed.npy"),all_data)


# ('/scratch/enis/data/nna/samples_wav/split_02:00/USGS_20160613_234225_2820m_00s__2880m_00s_56m_00s__58m_00s.wav', (242078, 242202))
# all_index is a dictionary with keys are full-path to files, values are indexes of array called all_data for each files_exists
all_index=np.load(os.path.join(data_dir,"data_file_indexes.npy"))
all_data=np.load(os.path.join(data_dir,"preprocessed.npy"))


# indexes are as long as 124 and 125, mostly 124 ({'124': 1798, '125': 154})

# import os
# import numpy as np
#
#

# file_count=len(files)/2
# all_index={}
# all_data = np.array([], dtype=np.float64).reshape(0,96,64)
#
# counter=0
# asd={"124":0,"125":0}
# for i in range(1,int(file_count)+1):
#   indexes=data_dir+"data_file_indexes_"+str(i)+".npy"
#   data_file_indexes=np.load(indexes)
#   #get dict from numpy array
#   data_file_indexes=data_file_indexes[()]
#   for key,indexes in data_file_indexes.items():
#       if indexes[1]-indexes[0] not in [124,125]:
#           print(indexes,indexes[1]-indexes[0])
#           print(key)
#           break
#       asd[str(indexes[1]-indexes[0])]+=1
#

# print(asd)
