

python evaluate_SN_3.py --batch_size 32 \
--lamb_att_dist 0.0 --lamb_att_global 0.0 --lamb_att_span 0.0 \
--workers 16 --lr 0.0001 \
--manualSeed 3483 \
--cuda --is_save \
--nepoch 10 \
--SESSION 'OpenImages_vgg_conv_sa_ff_gc_lr0.0001_lrelu_heads_8_gcontext_1_LRELU_seed_3483_less_than_40_labels' \
--summary "Removed standard LRSched, topk, 1 convrelu, Vgg4kDP0p2_VggEval from names from T6p7. VGG base model separated. Initial conv+relu+bn. Linear Gcontext V7 of 512 channels for channel attention 512 (mean of 14x14, sigmoid BSx512 atn) passed through 3x3 Conv-Relu-BN with skip connection, then concat with SAFF/SA out, 512-1024-1024 used as FF (No Relu at out) and passed to 1x1 Conv-LRelu and to 512 W.  No Relu after GContext. Changed W1 and Wcyc params to linear layers. Finetune LR 0.00033." \
--save_path "results" --mul_dropout --optim "adam" \
--beta1 0.9 --bias \
--val_batch_size 32 --test_batch_size 32 \
--ff_ksz 1 \
--gcontext_ch 512 --linear_gcontext \
--global_context --global_context_detach --bias_W \
--second_conv --gcff_ksz 3 --gcontext_heads 1 \
--num_labels 40 --heads 8