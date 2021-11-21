#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import model as model
import util_nus as util
from config import opt
import numpy as np
import random
import time
import os
import socket
from torch.utils.data import DataLoader
import h5py
import pickle
import logging

if not os.path.exists("logs"):
    os.mkdir("logs")
log_filename = os.path.join("logs",opt.SESSION + '.log')
logging.basicConfig(level=logging.INFO, filename=log_filename)

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

model_test = model.BiAM(opt, dim_feature=[196,512])
print(model_test)
logging.info(model_test)
name='NUS_WIDE_{}'.format(opt.SESSION)
opt.save_path += '/'+name

if opt.cuda:
    model_att = model_att.cuda()
    data.vecs_81 = data.vecs_81.cuda()
    data.vecs_925 = data.vecs_925.cuda()
gzsl_vecs = torch.cat([data.vecs_925,data.vecs_81],0)

test_start_time = time.time()
print("EVALUATION MODE")
logging.info("EVALUATION MODE")
model_path = os.path.join(opt.save_path, 'model_best.pth')
print(model_path)
logging.info(model_path)
util.load_checkpoint(model_test, model_path)
model_test.eval()

src = opt.src
test_loc = os.path.join(src, 'features', 'nus_wide_test.h5')
test_features = h5py.File(test_loc, 'r')
test_feature_keys = list(test_features.keys())
image_filenames = util.load_dict(os.path.join(src, 'test_img_names.pkl'))
test_image_filenames = image_filenames['img_names']
ntest = len(test_image_filenames)
test_batch_size = opt.test_batch_size

print(ntest)
logging.info(ntest)

prediction_81 = torch.empty(ntest,81)
prediction_1006 = torch.empty(ntest,1006)

lab_81 = torch.empty(ntest,81)
lab_1006 = torch.empty(ntest,1006)

for m in range(0, ntest, test_batch_size):
    strt = m
    endt = min(m+test_batch_size, ntest)
    bs = endt-strt
    c=m
    c+=bs
    features, labels_1006, labels_81 = np.empty((bs,512,196)), np.empty((bs,1006)), np.empty((bs,81))
    for i, key in enumerate(test_image_filenames[strt:endt]):
        features[i,:,:] = np.float32(test_features.get(key+'-features'))
        labels_1006[i,:] =  np.int32(test_features.get(key+'-labels'))
        labels_81[i,:] =  np.int32(test_features.get(key+'-labels_81'))

    features = torch.from_numpy(features).float()
    labels_1006 = torch.from_numpy(labels_1006).long()
    labels_81 = torch.from_numpy(labels_81).long()

    with torch.no_grad():
        logits_81  = model_test(features.cuda(), data.vecs_81)
        logits_1006  = model_test(features.cuda(), gzsl_vecs) ##seen-unseen

    prediction_81[strt:endt,:] = logits_81
    prediction_1006[strt:endt,:] = logits_1006

    lab_81[strt:endt,:] = labels_81
    lab_1006[strt:endt,:] = labels_1006

print(("completed calculating predictions over all {} images".format(c)))
logging.info(("completed calculating predictions over all {} images".format(c)))
logits_81_5 = prediction_81.clone()
ap_81 = util.compute_AP(prediction_81.cuda(), lab_81.cuda())
F1_3_81,P_3_81,R_3_81 = util.compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
F1_5_81,P_5_81,R_5_81 = util.compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

print('ZSL AP',torch.mean(ap_81).item())
print('k=3',torch.mean(F1_3_81).item(),torch.mean(P_3_81).item(),torch.mean(R_3_81).item())
print('k=5',torch.mean(F1_5_81).item(),torch.mean(P_5_81).item(),torch.mean(R_5_81).item())

logging.info('ZSL AP: %.4f',torch.mean(ap_81).item())
logging.info('k=3: %.4f,%.4f,%.4f',torch.mean(F1_3_81).item(),torch.mean(P_3_81).item(),torch.mean(R_3_81).item())
logging.info('k=5: %.4f,%.4f,%.4f',torch.mean(F1_5_81).item(),torch.mean(P_5_81).item(),torch.mean(R_5_81).item())

logits_1006_5 = prediction_1006.clone()
ap_1006 = util.compute_AP(prediction_1006.cuda(), lab_1006.cuda())
F1_3_1006,P_3_1006,R_3_1006 = util.compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
F1_5_1006,P_5_1006,R_5_1006 = util.compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

print('GZSL AP',torch.mean(ap_1006).item())
print('g_k=3',torch.mean(F1_3_1006).item(), torch.mean(P_3_1006).item(), torch.mean(R_3_1006).item())
print('g_k=5',torch.mean(F1_5_1006).item(), torch.mean(P_5_1006).item(), torch.mean(R_5_1006).item())

logging.info('GZSL AP:%.4f',torch.mean(ap_1006).item())
logging.info('g_k=3:%.4f,%.4f,%.4f',torch.mean(F1_3_1006).item(), torch.mean(P_3_1006).item(), torch.mean(R_3_1006).item())
logging.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_1006).item(), torch.mean(P_5_1006).item(), torch.mean(R_5_1006).item())


print("------------------------------------------------------------------")
print("TEST Time: {:.4f}".format(time.time()-test_start_time))
print("------------------------------------------------------------------")

logging.info("------------------------------------------------------------------")
logging.info("TEST Time: {:.4f}".format(time.time()-test_start_time))
logging.info("------------------------------------------------------------------")
