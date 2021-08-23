#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: naraysa & akshitac8
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import model as model
import util_openimages as util
from config import opt
import numpy as np
import random
import time
import os
from torch.utils.data import DataLoader
import h5py
import logging
from warmup_scheduler import GradualWarmupScheduler
from shutil import copy
import pandas as pd


if not os.path.exists("logs"):
    os.mkdir("logs")

log_filename = os.path.join("logs",opt.SESSION + '.log')
logging.basicConfig(level=logging.INFO, filename=log_filename)
logging.info(("Process JOB ID :{}").format(opt.job_id))

print(opt)
logging.info(opt)
#############################################
#setting up seeds
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
torch.set_default_tensor_type('torch.FloatTensor')
cudnn.benchmark = True  # For speed i.e, cudnn autotuner
########################################################

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

name='OPENIMAGES_{}'.format(opt.SESSION)
opt.save_path += '/'+name      
os.system("mkdir -p " + opt.save_path)

data = util.DATA_LOADER(opt) ### INTIAL DATALOADER ###
print('===> Loading datasets')
print(opt.src)
print('===> Result path ')
print(opt.save_path)
print('===> total samples')
print(data.ntrain)

logging.info('===> Loading datasets')
logging.info(opt.src)
logging.info('===> total samples')
logging.info(data.ntrain)

def train_sample():
    #train dataloader
    train_batch_feature, train_batch_labels = data.next_train_batch(opt.batch_size)
    return train_batch_feature, train_batch_labels

def val_sample():
    #val dataloader
    val_feature, val_labels_925, val_labels_81  = data.next_val()
    return val_feature, val_labels_925, val_labels_81

### attention model and loss function ####
model_vgg = None
model_vgg = model.vgg_net()
model_BiAM = model.BiAM(opt, dim_feature=[196,512])

print(model_BiAM)
logging.info(model_BiAM)

optimizer = torch.optim.Adam(model_BiAM.parameters(), opt.lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))

## saving files to result folders
copy('train_cleaned.py', opt.save_path)
copy('model_cleaned.py', opt.save_path)
copy('config_cleaned.py', opt.save_path)
copy('util.py', opt.save_path)
copy('scripts/train.sh', opt.save_path)

start_epoch = 1
num_epochs = opt.nepoch+1

if opt.cosinelr_scheduler:
    print("------------------------------------------------------------------")
    print("USING LR SCHEDULER")
    print("------------------------------------------------------------------")
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

print("initial learning rate", opt.lr)
logging.info(("initial learning rate {}".format(opt.lr)))


logger = util.Logger(cols=['index','mF1','mAP','lr', 'val_loss'],filename=opt.save_path+'/log.csv',is_save=True)
print(optimizer)
logging.info(optimizer)

if opt.cuda:
    model_BiAM = model_BiAM.cuda()
    model_vgg.cuda()
    model_vgg.eval()
    data.vecs_400 = data.vecs_400.cuda()
    data.vecs_7186 = data.vecs_7186.cuda()

gzsl_vecs = torch.cat([data.vecs_7186,data.vecs_400],0)


def train(epoch):
    print("TRAINING MODE")
    logging.info("TRAINING MODE")

    for i in range(0, len(data.h5_files)):
        train_features = data.train_data()
        train_feature_keys = list(train_features.keys())
        image_names = np.unique(np.array([m.split('-')[0] for m in train_feature_keys]))
        mean_loss = 0
        batch_start_time = time.time()
        for batch in range(0, 640, opt.batch_size):
            optimizer.zero_grad()
            train_inputs, train_labels = data.next_train_batch(opt.batch_size, train_features, image_names, batch)
            temp_label = torch.sum(train_labels>0,1)>0 #remove those images that don not have even a single 1 (positive label).
            train_labels   = train_labels[temp_label]
            train_inputs   = train_inputs[temp_label]

            ## Train with images containing 40 or less than 40 labels
            _train_labels = train_labels[torch.clamp(train_labels,0,1).sum(1)<=40]
            train_inputs = train_inputs[torch.clamp(train_labels,0,1).sum(1)<=40]
            train_inputs = train_inputs.cuda()
            _train_labels = _train_labels.cuda()

            vgg_4096 = model_vgg(train_inputs) if model_vgg is not None else None
            logits = model_BiAM(train_inputs, data.vecs_7186, vgg_4096)
            loss = model.ranking_lossT(logits, _train_labels.float())

            vggloss = torch.zeros(1).cuda()
            mean_loss += loss.item() + vggloss.item()

            if torch.isnan(loss) or loss.item() > 100:
                print('Unstable/High Loss:', loss)
                import pdb; pdb.set_trace()

            loss.backward()
            optimizer.step()
        mean_loss /= data.ntrain / opt.batch_size

        if opt.cosinelr_scheduler:
            learning_rate = scheduler.get_lr()[0]
        else:
            learning_rate = opt.lr

        print("------------------------------------------------------------------")
        print("Epoch:{}/{} \tBatch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, i, len(data.h5_files), time.time()-batch_start_time,mean_loss, learning_rate))
        print("------------------------------------------------------------------")

        logging.info("------------------------------------------------------------------")
        logging.info("Epoch:{}/{} \tBatch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, i, len(data.h5_files), time.time()-batch_start_time,mean_loss, learning_rate))
        logging.info("------------------------------------------------------------------")

        if (i > 3 and i % 100 == 0):
            torch.save(model_BiAM.state_dict(), os.path.join(opt.save_path,("model_latest_{}_{}.pth").format(i,epoch)))

for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
    train(epoch)
    if opt.cosinelr_scheduler:
        scheduler.step()