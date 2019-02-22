# speech and udio understanding course project

Assumes Python3.7

Conda cheatsheet:  
envName="speechEnv"  
conda create --name $envName  
conda activate speechEnv  
conda config --add channels anaconda  
conda confic --add channels pytorch  
conda install --file req.txt  
conda clean --yes --all  

Add kernel to jupyter kernels (use python that have ipython installed):  
python -m ipykernel install --user --name $envName --display-name "Python speechEnv"  

Requirements
* pytorch
* scipy
* kaggle


kaggle uses 7zip with dataset:  
sudo apt-get update  
sudo apt install p7zip-full  
