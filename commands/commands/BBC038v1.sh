#!/bin/bash

cd dataset/BBBC038v1

wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip

unzip -o dataset/BBBC038v1/stage1_train.zip -d dataset/BBBC038v1/stage1_train

cd ..