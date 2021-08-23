#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

export PATH=$PATH:$HOME/miniconda2/bin
source activate mlzsl


python train_cleaned.py --batch_size 32 \
--workers 16 --lr 0.0001 \
--job_id $SLURM_JOB_ID \
--manualSeed 3483 \
--cuda \
--nepoch 10 \
--SESSION 'CLEANED_OpenImages_vgg_conv_sa_ff_gc_lr0.0001_lrelu_heads_16_gcontext_1_LRELU_seed_3483_less_than_40_labels' \
--summary "Removed standard LRSched, topk, 1 convrelu, Vgg4kDP0p2_VggEval from names from T6p7. VGG base model separated. Initial conv+relu+bn. Linear Gcontext V7 of 512 channels for channel attention 512 (mean of 14x14, sigmoid BSx512 atn) passed through 3x3 Conv-Relu-BN with skip connection, then concat with SAFF/SA out, 512-1024-1024 used as FF (No Relu at out) and passed to 1x1 Conv-LRelu and to 512 W.  No Relu after GContext. Changed W1 and Wcyc params to linear layers. Finetune LR 0.00033." \
--save_path "results" --beta1 0.9 --val_batch_size 32 --test_batch_size 32 --heads 16
# --cosinelr_scheduler --lr_min 0.00002 --after_batch_scheduler \