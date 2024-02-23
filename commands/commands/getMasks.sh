#!/bin/bash

python scripts/amg.py \
    --checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model-type 'vit_h' \
    --input dataset/BBBC038v1/images \
    --output output/BBBC038v1

