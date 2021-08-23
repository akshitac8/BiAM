#!/bin/bash

export PATH=$PATH:$HOME/miniconda2/bin
source activate mlzsl

python train_nus.py --batch_size 32 --workers 16 --channel_dim 256 \
--manualSeed 3483 --cuda --nepoch 40 --SESSION 'nus_wide_paper' --src datasets/NUS-WIDE \
--save_path "nus_results" --lr_min 0.0005 --beta1 0.9 --cosinelr_scheduler \
--val_batch_size 100 --test_batch_size 100 --train --train_full_lr 0.000333 --heads 8

