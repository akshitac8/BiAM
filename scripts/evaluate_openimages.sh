#!/bin/bash

export PATH=$PATH:$HOME/miniconda2/bin
source activate mlzsl

python evaluate_openimages.py --batch_size 32 \
--workers 16 --lr 0.0001 \
--manualSeed 3483 \
--cuda \
--nepoch 10 \
--SESSION 'CLEANED_OpenImages_vgg_conv' \
--save_path "results" --beta1 0.9 --test_batch_size 32 --heads 16