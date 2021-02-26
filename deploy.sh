# jt0
sudo apt install nvtop
export WORKSPACE=~/llv
mkdir -p $WORKSPACE
cd $WORKSPACE
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p ./miniconda3 
$WORKSPACE/miniconda3/bin/pip install PyHamcrest
$WORKSPACE/miniconda3/bin/pip install twisted
$WORKSPACE/miniconda3/bin/pip install --upgrade pip
$WORKSPACE/miniconda3/bin/pip install opencv-python
$WORKSPACE/miniconda3/bin/pip install tensorboard
$WORKSPACE/miniconda3/bin/pip install pandas
$WORKSPACE/miniconda3/bin/pip install scipy
# $WORKSPACE/miniconda3/bin/pip install deprecated
$WORKSPACE/miniconda3/bin/conda install -y -c pytorch -c conda-forge pytorch
$WORKSPACE/miniconda3/bin/pip install torchvision

sudo mkdir /data/llv
sudo chown jittor:jittor /data/llv -R
# scp -P 54147 DemosaicDataset.tar jittor@jittor00.randonl.me:/data/llv
scp -P 30022 llv@166.111.139.23:/ssd/DemosaicDataset.tar  /data/llv
cd /data/llv
tar xf DemosaicDataset.tar
cd -


git clone https://github.com/llk2why/Retina2021.git
git fetch --all
git checkout dev


# jt3
export WORKSPACE=/mnt/disk/llv
mkdir -p $WORKSPACE
cd $WORKSPACE
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p ./miniconda3 
$WORKSPACE/miniconda3/bin/pip install PyHamcrest
$WORKSPACE/miniconda3/bin/pip install twisted
$WORKSPACE/miniconda3/bin/pip install --upgrade pip
$WORKSPACE/miniconda3/bin/pip install opencv-python
$WORKSPACE/miniconda3/bin/pip install tensorboard
$WORKSPACE/miniconda3/bin/pip install pandas
$WORKSPACE/miniconda3/bin/pip install scipy
# $WORKSPACE/miniconda3/bin/pip install deprecated
$WORKSPACE/miniconda3/bin/conda install -y -c pytorch -c conda-forge pytorch
$WORKSPACE/miniconda3/bin/pip install torchvision

mkdir $WORKSPACE/data
scp -P 30022 llv@166.111.139.23:/ssd/DemosaicDataset.tar  $WORKSPACE/data/
cd $WORKSPACE/data/
tar xf DemosaicDataset.tar
cd -


git clone https://github.com/llk2why/Retina2021.git
git fetch --all
git checkout dev