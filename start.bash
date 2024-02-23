

pip install git+https://github.com/facebookresearch/segment-anything.git

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


