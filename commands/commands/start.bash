mkdir SAM

cd SAM

git clone https://github.com/facebookresearch/segment-anything

cd segment-anything

#create a conda env
conda create -n SAM python=3.9

pip install git+https://github.com/facebookresearch/segment-anything.git

conda install -c "nvidia/label/cuda-11.7.1" cuda cuda-toolkit

module load gcc/gcc-11.2.0

pip install -e .

pip install -r requirements.txt

pip install -q git+https://github.com/huggingface/transformers.git
pip install e .
pip install datasets
pip install -q monai
pip install tifffile
pip install scipy
mkdir data
mkdir checkpoints

python /home/cap5415.student17/VisionProject/SAM2/segment-anything/customTraining.py

bash /home/cap5415.student17/VisionProject/SAM2/segment-anything/generateMasks.bash

bash /home/cap5415.student17/VisionProject/SAM2/segment-anything/commands/checkpoint.bash

bash /home/cap5415.student17/VisionProject/SAM2/segment-anything/commands/dataFileMaker.sh

bash commands/BBC038v1.sh


