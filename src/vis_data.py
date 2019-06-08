import numpy as np

SAMPLE_PER_HOUR_2CHANNEL=7500

def get_npy_files(folder_path,relative_file_path):
#   folder_path="/scratch/enis/data/nna/NUI_DATA/"
#   relative_file_path="07 Ice Rd/August 2016/ICERD_20160808_024504.mp3"[:-4]
  abs_file_path=folder_path+relative_file_path

  preds_index=abs_file_path+"_preds_index.npy"
  rawembeddings=abs_file_path+"rawembeddings.npy"
  embeddings=abs_file_path+"_embeddings.npy"
  preds=abs_file_path+"_preds.npy"

  preds_index=np.load(preds_index)
  rawembeddings=np.load(rawembeddings)
  embeddings=np.load(embeddings)
  preds=np.load(preds)
  if (rawembeddings.shape[0]//SAMPLE_PER_HOUR_2CHANNEL) < 49:
      print("error smaller than 49 hours")
      return None
  (rawembeddings,embeddings)=map(modify_array_1,(rawembeddings,embeddings))
  (preds_index,preds)=map(modify_array_2,(preds_index,preds))
  return (preds_index,preds,rawembeddings,embeddings)

# for np_array in [preds_index,preds,rawembeddings,embeddings]:
#     print(np_array.shape)


def info4name(file,name):
  name=name.split("/")[-1]
  file_id=name
  name=name.split("_")
#   if len(name)!=18:
#     print(name)
  site_name=name[0]
  site_names.append(site_name)
  date=name[1]
  year=date[0:4]
  month=date[4:6]
  day=date[6:8]
  file["date"]=(day,month,year)
  file["start_min"]=name[3][:-1]
  file["end_minute"]=name[6][:-1]
  file["site_id"]=name[2]
  file["index"]=data_file_indexes[file["path"]]
  files[file["path"]]=file

def modify_array_1(np_array):
        # (372570, 128)
        # (372570, 128)
    an_hour_double=SAMPLE_PER_HOUR_2CHANNEL #number of sampels in an hours actullay 7499
    assert an_hour_double%2==0, "one hour sample rate should be divisible by two"
    total_hours=49
    hours=np_array.shape[0]//an_hour_double

    np_array=np_array[:total_hours*an_hour_double]
    index_list=[]
    number_of_hours=(np_array.shape[0]//(an_hour_double//2))
    for i in range(0,number_of_hours,2):
        start= 0+(i*an_hour_double//2)
        end= (i+1)*(an_hour_double//2)
        # print(start,end)
        index_list.append(np.arange(start,end))
    index_list = np.array(index_list)
    np_array=np_array[index_list,:]
    np_array=np_array.reshape(-1,128)
    return np_array

def modify_array_2(np_array):
        # (37257, 527)
        # (37257, 527)
    an_hour_double=SAMPLE_PER_HOUR_2CHANNEL//10 #number of sampels in an hours actullay 7499
    assert an_hour_double%2==0, "one hour sample rate should be divisible by two"
    total_hours=49
    hours=np_array.shape[0]//an_hour_double
    np_array=np_array[:(total_hours*an_hour_double)]
    index_list=[]
    number_of_hours=(np_array.shape[0]//(an_hour_double//2))
    for i in range(0,number_of_hours,2):
        start= 0+(i*an_hour_double//2)
        end= (i+1)*(an_hour_double//2)
        # print(start,end)
        index_list.append(np.arange(start,end))
    index_list = np.array(index_list)
    np_array=np_array[index_list,:]
    np_array=np_array.reshape(-1,527)
    return np_array
