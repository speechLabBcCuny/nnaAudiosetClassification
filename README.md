# Nature Sound Processing Project

This repository contains code to instantiate and deploy an audio classification model.

### Example usage
! Requires model files
```python
import tensorflow as tf
from models_api import AudioSet

# create and instance of Audioset classifier
classifier=AudioSet()

#classify mp3 file
class_prob = classifier.classify_sound("./assets/10seconds.mp3")

# sort probabilities(scores) and get corresponding labels for first 5 class
labels,prob = classifier.prob2labels(class_prob,first_k=5)

#print results
for sample_labels,sample_probabilities in zip(labels,prob):
    for a_label,a_prob in zip(sample_labels,sample_probabilities):
        print(a_label,a_prob)

```

### Model files
```bash
wget https://max-assets-prod.s3.us-south.cloud-object-storage.appdomain.cloud/max-audio-classifier/1.0.0/assets.tar.gz
tar -xzvf assets.tar.gz
mv classifier_model.h5 ./assets/
mv vggish_pca_params.npz ./assets/
mv vggish_model.ckpt ./assets/
```

### Suggested virtual env usage with conda
Assumes Python3.7
```bash
#Conda cheatsheet:
envName="soundEnv"  
conda create --name $envName python=3.7
conda activate speechEnv  
conda config --add channels anaconda,conda-forge
conda install --file Requirements.txt  
conda clean --yes --all  
```

Add kernel to jupyter kernels (use python that have ipython installed):  
```bash
conda install ipykernel
python -m ipykernel install --user --name "$envName" --display-name "Python3-$envName"  
```


[Project Logs.md](https://github.com/speechLabBcCuny/nnaAudiosetClassification/blob/master/Project_logs.md)
