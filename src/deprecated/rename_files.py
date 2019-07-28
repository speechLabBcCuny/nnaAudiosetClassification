#this script's is written to change file names that have errors because of a typo in file name generation

import os

def preb_names(mp3_file_path,output_dicretory,abs_input_path):
# def mp3toEmbed(mp3_file_path,output_dicretory,abs_input_path):
	# mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
	# output_dicretory="/scratch/enis/data/nna/embeddings/"
	# abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
	abs_len=len(abs_input_path)
	mp3_file_path=mp3_file_path.replace("NUI_DATA_copy","NUI_DATA")
	# relative_path=mp3_file_path[abs_len:]
	# relative_path="/"+"/".join(mp3_file_path.split("/")[-3:])
	mp3_file_name=os.path.basename(mp3_file_path)
	path_to_file="/".join(mp3_file_path.split("/")[-3:-1])
	mp3_file_name=mp3_file_name.split(".")[0]
	# wav_file_name=mp3_file_name+".wav"
	# wav_file_path=os.path.join(output_dicretory,path_to_file,wav_file_name)
	# print(output_dicretory,path_to_file,mp3_file_name)
	embeddings_file_name=os.path.join(output_dicretory,path_to_file,mp3_file_name)+"_embeddings.npy"
	segments_folder=os.path.join(output_dicretory,path_to_file,mp3_file_name+"_segments/")
	pre_processed_folder=os.path.join(output_dicretory,path_to_file,mp3_file_name+"_preprocessed/")
	return mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder


mp3_list_path="/scratch/enis/mp3_files.txt"

with open(mp3_list_path) as f:
  f=[line.strip() for line in f.readlines()]

# here is the problem:
# some files are missing 2 last numbers
# ITKILLIK1_20160813_032044_embeddings.npy
# ITKILLIK1_20160813_032044rawembeddings.npy
# ITKILLIK1_20160813_0320_preds_index.npy
# ITKILLIK1_20160813_0320_preds.npy

abs_input_path="/scratch/enis/data/nna/NUI_DATA/"
output_dicretory="/scratch/enis/data/nna/NUI_DATA/"
abs_input_path="gs://deep_learning_enis/speech_audio_understanding/NUI_DATA"
output_dicretory="gs://deep_learning_enis/speech_audio_understanding/NUI_DATA"

for mp3_file_path in f:
	mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder=preb_names(mp3_file_path,output_dicretory,
						abs_input_path)
	old_predictions_file_name=embeddings_file_name[:-17]+"_preds.npy"
	old_prediction_index_file_name=embeddings_file_name[:-17]+"_preds_index.npy"
	predictions_file_name=embeddings_file_name[:-15]+"_preds.npy"
	prediction_index_file_name=embeddings_file_name[:-15]+"_preds_index.npy"
	if not os.path.exists(old_predictions_file_name) and "/NUI_DATA/NUI_DATA/" in old_predictions_file_name:
		old_predictions_file_name=old_predictions_file_name.replace("/NUI_DATA/NUI_DATA/","/NUI_DATA/")
		predictions_file_name=predictions_file_name.replace("/NUI_DATA/NUI_DATA/","/NUI_DATA/")
		old_prediction_index_file_name=old_prediction_index_file_name.replace("/NUI_DATA/NUI_DATA/","/NUI_DATA/")
		prediction_index_file_name=prediction_index_file_name.replace("/NUI_DATA/NUI_DATA/","/NUI_DATA/")
		# print(k)
		print(old_predictions_file_name,predictions_file_name)
		print(old_prediction_index_file_name,prediction_index_file_name)
		os.rename(old_predictions_file_name,predictions_file_name)
		os.rename(old_prediction_index_file_name,prediction_index_file_name)
		# print(mp3_file_path)
		# print("-----")
	# else:
	# 	os.rename(old_predictions_file_name,predictions_file_name)
	# 	os.rename(old_prediction_index_file_name,prediction_index_file_name)
