## Nature Sounds Processing Project

This repository contains utility functions for processing nature sounds and extracting features from them.


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
