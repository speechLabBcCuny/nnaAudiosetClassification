# this one does VGGish inference
from pre_process_func import *
import tensorflow as tf

from time import time
import argparse
import sys
import numpy as np

sys.path.insert(0, './audioset')
import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess

def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
	"""Define VGGish model, load the checkpoint, and return a dictionary that points
	to the different tensors defined by the model.
	"""

	vggish_slim.define_vggish_slim(training=False)
	checkpoint_path = 'vggish_model.ckpt'
	vggish_params.EXAMPLE_HOP_SECONDS = hop_size

	vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

	features_tensor = sess.graph.get_tensor_by_name(
	  vggish_params.INPUT_TENSOR_NAME)
	embedding_tensor = sess.graph.get_tensor_by_name(
	  vggish_params.OUTPUT_TENSOR_NAME)

	layers = {'conv1': 'vggish/conv1/Relu',
			'pool1': 'vggish/pool1/MaxPool',
			'conv2': 'vggish/conv2/Relu',
			'pool2': 'vggish/pool2/MaxPool',
			'conv3': 'vggish/conv3/conv3_2/Relu',
			'pool3': 'vggish/pool3/MaxPool',
			'conv4': 'vggish/conv4/conv4_2/Relu',
			'pool4': 'vggish/pool4/MaxPool',
			'fc1': 'vggish/fc1/fc1_2/Relu',
			'fc2': 'vggish/fc2/Relu',
			'embedding': 'vggish/embedding',
			'features': 'vggish/input_features',
		 }
	g = tf.get_default_graph()
	for k in layers:
		layers[k] = g.get_tensor_by_name( layers[k] + ':0')
	pca_params_path="./vggish_pca_params.npz"
	pproc = vggish_postprocess.Postprocessor(pca_params_path)

	return {'features': features_tensor,
		  'embedding': embedding_tensor,
		  'layers': layers,
		 },pproc

def inference(pre_processed_npy_files,vgg,sess,embeddings_file_name,batch_size=256):
	# print("len sounds",len(sounds))
	embeddings = np.array([], dtype=np.int16).reshape(0,128)
	for npy_file in pre_processed_npy_files:
		sound=np.load(npy_file)
	# for sound in sounds:
		# print(len(sound))
		numberoffiles=sound.shape[0]
		for batch_index in range(0,numberoffiles,batch_size):
			# print("inference")
			if (batch_index+batch_size) <numberoffiles:
				a_batch=sound[batch_index:batch_index+batch_size]
			else:
				a_batch=sound[batch_index:]
			[embedding_batch] = sess.run([vgg['embedding']],
									 feed_dict={vgg['features']: a_batch})
			embeddings = np.concatenate((embeddings,embedding_batch))
	# print("postprocess")
	# del sounds
	raw_embeddings_file_name=embeddings_file_name[:-15]+"rawembeddings.npy"
	np.save(raw_embeddings_file_name,embeddings)
	postprocessed_batch = pproc.postprocess(embeddings)
	# del embeddings
	# print("saving")
	np.save(embeddings_file_name,postprocessed_batch)
	# del postprocessed_batch
	return embeddings_file_name

# find /home/data/nna/stinchcomb/ -name "*.*3" -print0 | xargs -0 python end2end.py --input_files &> endlogs.txt &
if __name__ == "__main__":

	# mp3_file_path="/home/data/nna/stinchcomb/NUI_DATA/18 Fish Creek 4/July 2016/FSHCK4_20160629_194935.MP3"
	# output_dicretory="/scratch/enis/data/nna/wav_files/"
	# abs_input_path="/home/data/nna/stinchcomb/NUI_DATA/"
	parser = argparse.ArgumentParser(description='mp3 to embeddings')
	parser.add_argument('--output_folder', type=str,
					   help='output folder',default="/scratch/enis/data/nna/wav_files/")
	parser.add_argument('--abs_input_path', type=str,
					   help='absoulute input folder such as',default="/home/data/nna/stinchcomb/NUI_DATA/")
	parser.add_argument('--segment_len', type=str,
						   help='length of segments',default="01:00:00")
	parser.add_argument('--input_files',nargs='+',type=str,default=None)
	# parser.add_argument('--input_folder',nargs=1,type=str,default=None)
	args = parser.parse_args()

	# args.output_folder= "/scratch/enis/data/nna/wav_files/"
	# args.abs_input_path = "/home/data/nna/stinchcomb/NUI_DATA/"
	# args.input_files=[" "," "," "]
	# create directory tree from original folder

	# input_files=[" "," "," "]
	#find all mp3 files
	# input_files=!find '{abs_input_path}' -name "*.*3"
	input_files=args.input_files
	output_folder=args.output_folder
	abs_input_path=args.abs_input_path
	if not os.path.exists(output_folder):
		SRC=abs_input_path
		DEST=output_folder
		shutil.copytree(SRC, DEST, ignore=ig_f)

	input_files.sort()
	tf.reset_default_graph()
	sess = tf.Session()
	vgg,pproc = CreateVGGishNetwork()

	for i,input_file in enumerate(input_files):
		start=time()
		print("{} - file: {}".format(i,input_file))
		sys.stdout.flush()
		##### step - 0 get prepare names
		mp3_file_path,segments_folder,embeddings_file_name,pre_processed_folder=preb_names(input_file,
											  output_folder,abs_input_path)
		if os.path.exists(embeddings_file_name):
			continue

		##### step 1 -  divide files into parts
		# mp3_segments=divide_mp3(mp3_file_path,segments_folder,segment_len=args.segment_len)
		#### step 2 - pre-process
		# mp3_segments=os.listdir(segments_folder)
		# pre_process(mp3_segment,segments_folder,pre_processed_folder,saveNoReturn=True)
		#### step 3 - inference
		pre_processed_npy_files=[pre_processed_folder+file for file in os.listdir(pre_processed_folder)]

		embeddings_file_name=inference(pre_processed_npy_files,vgg,sess,embeddings_file_name,batch_size=256)
		rmv_segmets(pre_processed_folder)
		end=time()
		print("It took {} seconds".format(end-start))
		sys.stdout.flush()
	sess.close()
