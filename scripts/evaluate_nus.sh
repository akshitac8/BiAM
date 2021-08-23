#!/bin/bash

export PATH=$PATH:$HOME/miniconda2/bin
source activate mlzsl

python evaluate_nus.py --batch_size 32 --workers 16 --channel_dim 256 \
--manualSeed 3483 --cuda --SESSION 'nus_wide_paper' \
--save_path "nus_results" --lr_min 0.0005 --beta1 0.9 --cosinelr_scheduler \
--test_batch_size 100 --heads 8