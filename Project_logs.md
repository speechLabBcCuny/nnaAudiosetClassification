### Progress:

* **(08/3/2019)** Understand commmon tools and data structure for Speech and Audio Processsing
  * [kaggle_speech.ipynb](./notebooks/kaggle_speech.ipynb) Going through dataset and kernels from [TensorFlow Speech Recognition Challenge](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
* **(15/3/2019)** I used vanilla classifiers for predicting location and recording hour of the sound given VGGish embeddings,highest accuracy results are as following ([notebook](./notebooks/Classify_time&location.ipynb):
    * LinearSVC 0.882 for location
    * LinearSVC 0.867 for hour of the day
* **(22/03/2019)** I classified embeddings generated with VGGish model using an attention model [audioset classification model w/ attention](https://github.com/qiuqiangkong/audioset_classification). Using post-processed (PCA & whitening) embeddings did not provide any meaningfully classification.  When I used unprocessed embeddings, class assignments were not meaningful. I also tried to cluster embeddings with t-sne and analyze results, however those cluster did not provide so much information as well. [notebook](./notebooks/Audioset_model_inference.ipynb)
* **(29/03/2019)** I created a pipeline that can process audio files in parallel so I can work with big files without a lot of hands on work. [pipeline md file](./src/hpc_pipeline.md)
* **(05/04/2019)** Since I could not find meaningful clusters on sampled data, I took 24 hours continuous recordings, and analyzed those. You can see t-sne visualization of those . Every point represents 10 seconds of recording (embeddings are averaged). Embeddings [averaged_embeddings_1](./vis/averaged_embeddings_1.html) or post-processed embeddings [averaged_post_1](./vis/averaged_post_1.html). However there is not any obvious clustering.
* **(12/04/2019)**
    * Testing VGGish and attention model with original dataset to make sure pipeline is working:
        * Using original audioset dataset (128 8bit quantized features) with attention model:
            * (128 8bit quantized features) --> [attention model] --> labels
        * From scracth:
            * (youtube video) --> [VGGish] -->  (128 8bit quantized features) --> [attention model] --> labels

    * I found the bug in my code that causes attention model to output random results. It was a type casting issue.
  * **(08/05/2019)**
    * I realized problems with pre-processing parameters causing input/output ratio to be different than 1. Refer to hop-size title.
  * **(07/06/2019)**
    * When we divide mp3 files into 1 hour segments, precision is +/- 1 second because we are not re-encoding/decoding. Since pre-processing is done on 1 hour segments separately, this might result in shifts between input and outputs again. So we decided to keep results for each file separately.
      * When it is slightly shorter, pre-processed version still becomes 20
      * when it is slightly  longer, we cannot pre-process that small extra part, so gets ignored
    * I also found a array indexing bug and fixed it, correct one is at the
  * **(16/06/2019)**
    * mp3's logical frame have 2 chunks per channel and each chunk stores 576 frequency samples. Since 44100 is not multiple of 576, we might not be able to divide mp3 with second granularity. As a result, while one file have extra 576 samples other one misses 576 samples for an exact second. Here how we handle those two files in python code:
      ```python
        #longer 1 hour file
        len_wav_data=158760576
        #shorther 1 hour file
        len_wav_data=158759424
        print("left raw:",len_wav_data%(sr*excerpt_len))
        left=len_wav_data%(sr*excerpt_len)
        print("left seconds:",left/sr)
        if left<22712:
           print("cannot")
        elif left<42998:
          print("result will be emtpy")
        else:
           print("extra samples:", ((left-22712)//20286))
      ```



### Next:
* Prepare a notebook with playing those specific parts, and share with you.
* Running Audioset model over all the data would be good to do
* Creating spatio-temporal distributions of the tags coming from Audioset model
    * ( _So "honk" probably means waterfowl, and you could see what date they show up.  Not sure about "turkey" and "gobble".  Maybe "fowl" or "music" are songbirds?  See what the frog is.  See when the insects show up.  I think there's lots that we can learn by just counting these occurrences in different time chunks (every 4 hours over the summer, time of day aggregated over all days, etc)._)
*  Date at which migrating birds arrive

### Long Term:
* Take the clips that have a strong prediction and use them in the game.
* Build a simple search engine over weighted tag predictions using whoosh or something similar


##### pre-Processsing
  * Hop-size and channel count is important


##### Notes related to Hop size
(I am using hop_size=0.46)
Attention model is using 10 embeddings from VGGish which each one corresponds to one second according to [paper](ttps://hyp.is/q4G_WHEdEemVkSvHB9vWGA/arxiv.org/pdf/1803.02353.pdf). However their hop size creates 20 samples for 10 seconds. I will just use 20 samples for 10 seconds and make 1 prediction since it is an RNN.

Another problem was, pre_processing function does not handle discarding data when input is bigger than 10 seconds.
This causes such results:

when hop_size=0.96:
10 second —> 10 samples
100 second —>   104 samples
3600 second —> 3749 samples

Since there are extra 149 samples for 3600 seconds and 149 is not divisible by 360, I did not know how to handle this situation. So, now I just call same function over 10 second segments.

Related information from audiset google group:   
"The context window of log mel spectrogram that serves as input to the model is in fact 0.96 s. However, for the feature release we standardized to a 1 Hz embedding frame rate, which means our hop size was longer than the context window (i.e. 40 ms of each second of input is discarded).” [src](https://groups.google.com/d/msg/audioset-users/4O6DzbePVAo/o5f-aIgfAQAJ)

"The embeddings are generated from 0.96 s segments of audio left-aligned on integer second marks (i.e. the first depends on audio from 0-0.96s, the second from 1.0-1.96s, etc).” [src](https://groups.google.com/d/msg/audioset-users/4O6DzbePVAo/H4o6usomAQAJ).



##### Jupyter notebooks over SSH
ssh  -L localhost:8890:localhost:8890 Momentsnotice

jupyter notebook stop 8888

jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --no-browser \
  --NotebookApp.port_retries=0
