# FineT Tuning SAM for Medical Data
Precise segmentation of medical images is essential for accurate diagnosis and successful
planning of therapy. In many medical image segmentation tasks, the U-shaped architecture,
commonly known as U-Net and conventional segmentation techniques often encounter
difficulties when dealing with the intricate and diverse nature of medical imaging. Furthermore,
U-Net architects often face challenges in explicitly modeling long-range dependencies due to the
inherent locality of convolution operations. Additionally, such architects normally require large
annotated training dataset, a task that is often difficult given challenges present in the labor
intensive essence of medical image annotation.
The objective of this work is to use the Segment Anything Model (SAM) on medical imaging
datasets in order to enhance the accuracy and efficiency of segmentation. The preparation of a
large number of datasets and the acquisition of masks for each of them consisted of the most
significant obstacle. We conducted two separate experimentations one using SAM’s own pretarined checkpoints provided in repository and one by fine tuning the last decoder layer.
The quality and accuracy of the segmentation were quantitatively evaluated using the
Intersection Over Union (IOU) metric. In general terms, SAM demonstrates the potential for use
in the field of medical images, provided that appropriate prompting mechanisms are utilized for
the dataset and tasks of choice.

# How to run the code


```
kdir SAM

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


bash commands/checkpoint.bash
#!/bin/bash

mkdir Modelcheckpoints
cd Modelcheckpoints
wget wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..



bash commands/BBC038v1.sh
#!/bin/bash

cd dataset/BBBC038v1

wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip

unzip -o dataset/BBBC038v1/stage1_train.zip -d dataset/BBBC038v1/stage1_train


To do so we use the provide script in the Segment Anything repo called amg.py
bash commands/maskerGenerater.sh
#!/bin/bash

python scripts/amg.py \
    --checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model-type 'vit_h' \
    --input dataset/BBBC038v1/images \
    --output output/BBBC038v1



To train

python commands/customTraining.py

```

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
