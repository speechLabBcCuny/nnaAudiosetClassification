# Nature Sound Processing Project

[Project Logs.ipynb](https://github.com/EnisBerk/speech_audio_understanding/blob/master/Project%20Logs.ipynb)

Assumes Python3.7
```
#Conda cheatsheet:  
envName="speechEnv"  
conda create --name $envName python=3.7
conda activate speechEnv  
conda config --add channels anaconda  
conda confic --add channels pytorch  
conda install --file req.txt  
conda clean --yes --all  
```




Add kernel to jupyter kernels (use python that have ipython installed):  
```
conda install ipykernel 
python -m ipykernel install --user --name "$envName" --display-name "Python3-$envName"  
```
Requirements
* pytorch
* scipy
* kaggle


kaggle uses 7zip with dataset:  
sudo apt-get update  
sudo apt install p7zip-full  
```
# for Getting data from cloud on unix with credentials
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

pip install kaggle
```
