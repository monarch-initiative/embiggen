#!/bin/bash

# gcc, make, ecc
sudo apt-get update -qyy
sudo apt-get install build-essential htop byobu wget snapd libopenblas-dev libgflags-dev -qyy

###########################################################
# Install anaconda to have an easily reporducible python environments
###########################################################
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O anaconda.sh
bash ./anaconda.sh -b
echo "export PATH=\$PATH:$HOME/anaconda3/bin" >>$HOME/.bashrc

###########################################################
# Install CUDA 10.1
###########################################################

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
# sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

###########################################################
# Install CUDA Latest
###########################################################

curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda -y

################################################
# Install TensorFlow for GPU
################################################

conda install tensorflow-gpu

################################################
# Install CMake Latest
################################################

sudo snap install core
sudo reboot # We need to restart the system after having installed the code of snap
sudo snap install cmake --classic

#################################################
# Install CUDA TSNE
##################################################
git clone https://github.com/rmrao/tsne-cuda.git && cd tsne-cuda
git submodule init
git submodule update
cd build/
cmake ..
make
cd python
pip install -e .