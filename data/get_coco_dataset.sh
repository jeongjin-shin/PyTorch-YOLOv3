#!/bin/bash

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# Download Images
#wget -c "https://pjreddie.com/media/files/train2014.zip" --header "Referer: pjreddie.com"
#wget -c "https://pjreddie.com/media/files/val2014.zip" --header "Referer: pjreddie.com"

# Unzip
unzip -q /content/ggdrive/MyDrive/train2014.zip
unzip -q /content/ggdrive/MyDrive/val2014.zip

cd ..

# Download COCO Metadata
wget -c "https://pjreddie.com/media/files/instances_train-val2014.zip" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/trainvalno5k.part" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/labels.tgz" --header "Referer: pjreddie.com"
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
cp 5k.txt ../coco/
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
