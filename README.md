# speech and udio understanding course project

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
np_0=np.random.rand(122,128)
tr_0=torch.tensor(np_0)
np_1=np.array(tr_0)
tr_0_numbers,tr_0_index=torch.topk(tr_0, 128, dim=1, largest=True, sorted=True)
np_0_numbers=np.argsort(np_0,axis=1)

```
echo "10:00/09/July\n" &>> logs_run0.96.txt &&  python main.py &>> logs_run0.96.txt && python sendmail.py -s -m "all files 0.96" || python sendmail.py -m "all files 0.96" &

cat "/home/enis/projects/nna/mp3file.txt" | parallel --xargs CUDA_VISIBLE_DEVICES=1 python  pipe4-2.py --input_files {} &>> logs_run_last.txt && python sendmail.py -s -m "audioset inf" || python sendmail.py -m "audioset inference" &

```
