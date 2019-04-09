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
    
### Next:
* Update notebooks and codes to remove bug and organise codes on src
* Prepare a notebook with playing those specific parts, and share with you. 
* Running Audioset model over all the data would be good to do
* Creating spatio-temporal distributions of the tags coming from Audioset model
    * ( _So "honk" probably means waterfowl, and you could see what date they show up.  Not sure about "turkey" and "gobble".  Maybe "fowl" or "music" are songbirds?  See what the frog is.  See when the insects show up.  I think there's lots that we can learn by just counting these occurrences in different time chunks (every 4 hours over the summer, time of day aggregated over all days, etc)._)
*  Date at which migrating birds arrive

### Long Term:
* Take the clips that have a strong prediction and use them in the game. 
* Build a simple search engine over weighted tag predictions using whoosh or something similar
