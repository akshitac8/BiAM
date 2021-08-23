#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: naraysa & akshitac8
"""


import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import model as model
import util_nus as util
from config import opt
import numpy as np
import random
import time
import os
import socket
import h5py
import pickle
import logging
from warmup_scheduler import GradualWarmupScheduler

## setting up the logs folder ##
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

name='NUS_WIDE_{}'.format(opt.SESSION)
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
logging.info('===> Result path ')
logging.info(opt.save_path)
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

### Intialize attention model and global feature extractor ####
model_vgg = model.vgg_net()
model_biam = model.BiAM(opt, dim_feature=[196,512])
model_test = model.BiAM(opt, dim_feature=[196,512])
print(model_biam)
logging.info(model_biam)

## initialize optimizer ###
optimizer = torch.optim.Adam(model_biam.parameters(), opt.lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))
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
logger = util.Logger(cols=['index','mF1','mF1_u_val','mAP','lr','val_loss'],filename=opt.save_path+'/log.csv',is_save=True)
eval_interval = max((opt.eval_interval),2)

print(optimizer)
logging.info(optimizer)

if opt.cuda:
    model_biam = model_biam.cuda()
    model_test = model_test.cuda()
    model_vgg.cuda()
    model_vgg.eval()
    data.vecs_81 = data.vecs_81.cuda()
    data.vecs_925 = data.vecs_925.cuda()
gzsl_vecs = torch.cat([data.vecs_925,data.vecs_81],0)

##  train function ###
def train(epoch):
    print("TRAINING MODE")
    logging.info("TRAINING MODE")
    epoch_start_time = time.time()
    mean_loss = 0
    for i in range(0, data.ntrain, opt.batch_size):
        optimizer.zero_grad()
        train_inputs, train_labels = train_sample()

        ### remove empty label images while training ###
        temp_label = torch.clamp(train_labels,0,1)
        temp_seen_labels = temp_label.sum(1)
        temp_label = temp_label[temp_seen_labels>0]
        train_labels   = train_labels[temp_seen_labels>0]
        train_inputs   = train_inputs[temp_seen_labels>0]
        ###
        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        vgg_4096 = model_vgg(train_inputs)
        vgg_4096 = vgg_4096.detach()
        logits = model_biam(train_inputs, data.vecs_925, vgg_4096)
        loss = model.ranking_lossT(logits, train_labels.float())
        mean_loss += loss.item()
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

    if opt.train:
        print("------------------------------------------------------------------")
        print("Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, time.time()-epoch_start_time,mean_loss, learning_rate))
        print("------------------------------------------------------------------")

        logging.info("------------------------------------------------------------------")
        logging.info("Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, time.time()-epoch_start_time,mean_loss, learning_rate))
        logging.info("------------------------------------------------------------------")
    else:
        learning_rate = opt.train_full_lr
        print("------------------------------------------------------------------")
        print("FINETUNING Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, time.time()-epoch_start_time,mean_loss, learning_rate))
        print("------------------------------------------------------------------")

        logging.info("------------------------------------------------------------------")
        logging.info("FINETUNING Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs, time.time()-epoch_start_time,mean_loss, learning_rate))
        logging.info("------------------------------------------------------------------")

        torch.save(model_biam.state_dict(), os.path.join(opt.save_path,("model_best_train_full_{}.pth").format(epoch)))

##  validation function ###

def val(epoch):
    print("validation mode")
    logging.info("validation mode")
    val_start_time = time.time()
    mean_val_loss = 0

    ### load val data ###
    seen_val_visual_features, seen_925_val_visual_labels, seen_81_val_visual_labels = val_sample()
    seen_val_visual_features = seen_val_visual_features
    seen_925_val_visual_labels = seen_925_val_visual_labels
    seen_81_val_visual_labels =  seen_81_val_visual_labels

    prediction_81 = torch.empty(len(seen_81_val_visual_labels),81)
    prediction_925 = torch.empty(len(seen_81_val_visual_labels),925)
    val_batch_size = opt.val_batch_size

    if model_vgg is not None:
        model_vgg.eval()
    for i in range(0, len(seen_81_val_visual_labels), val_batch_size):
        strt = i
        endt = min(i+val_batch_size, len(seen_81_val_visual_labels))
        with torch.no_grad():
            vgg_4096 = model_vgg(seen_val_visual_features[strt:endt,:,:].cuda()) #if model_vgg is not None else None
            vgg_4096 = vgg_4096.detach() #check if this is needed
            logits_81 = model_biam(seen_val_visual_features[strt:endt,:,:].cuda(), data.vecs_81, vgg_4096)
            logits_925 = model_biam(seen_val_visual_features[strt:endt,:,:].cuda(), data.vecs_925, vgg_4096)
            loss_925 = model.ranking_lossT(logits_925.cuda(), seen_925_val_visual_labels[strt:endt,:].cuda().float())
            prediction_81[strt:endt,:] = logits_81
            prediction_925[strt:endt,:] = logits_925
            mean_val_loss += loss_925.item()

    mean_val_loss /= len(seen_81_val_visual_labels) / val_batch_size
    if opt.cosinelr_scheduler:
        learning_rate = scheduler.get_lr()[0]
    else:
        learning_rate = opt.lr
    print("------------------------------------------------------------------")
    print("Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs,time.time()-val_start_time, mean_val_loss, learning_rate))
    print("------------------------------------------------------------------")
    logging.info("------------------------------------------------------------------")
    logging.info("Epoch: {}/{} \tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, num_epochs,time.time()-val_start_time, mean_val_loss, learning_rate))
    logging.info("------------------------------------------------------------------")

    ap_val = util.compute_AP(prediction_925.cuda(), seen_925_val_visual_labels.cuda())
    F1_val,P_val,R_val = util.compute_F1(prediction_925.cuda(), seen_925_val_visual_labels.cuda(), 'overall', k_val=5)
    F1_u_val,P_u_val,R_u_val = util.compute_F1(prediction_81.cuda(), seen_81_val_visual_labels.cuda(), 'overall', k_val=5)
    mF1_val,mP_val,mR_val,mAP_val = [torch.mean(F1_val),torch.mean(P_val),torch.mean(R_val),torch.mean(ap_val)]
    mF1_u_val,mP_u_val,mR_u_val = [torch.mean(F1_u_val),torch.mean(P_u_val),torch.mean(R_u_val)]
    print('SEEN AP',mAP_val.item())
    print('k=5 AT 925',mF1_val.item(),mP_val.item(),mR_val.item())
    print('k=5 AT 81 ',mF1_u_val.item(),mP_u_val.item(),mR_u_val.item())
    logging.info('SEEN AP=%.4f',mAP_val.item())
    logging.info('k=5 AT 925: %.4f,%.4f,%.4f',mF1_val.item(),mP_val.item(),mR_val.item())
    logging.info('k=5 AT 81: %.4f,%.4f,%.4f ',mF1_u_val.item(),mP_u_val.item(),mR_u_val.item())
    values = [epoch, mF1_val,mF1_u_val,mAP_val,learning_rate, mean_val_loss]
    logger.add(values)
    print('{} mF1: {} mF1_u_val: {} mAP: {} lr: {}'.format(*values))
    print('Precision: {} Recall: {}'.format(mP_val,mR_val))
    logging.info('{} mF1: {} mF1_u_val: {} mAP: {} lr: {}'.format(*values))
    logging.info('Precision: {} Recall: {}'.format(mP_val,mR_val))
    logger.save()
    if mF1_val >= logger.get_max('mF1'):
        print("model saved")
        logging.info("model saved")
        torch.save(model_biam.state_dict(), os.path.join(opt.save_path,"model_best.pth"))
    torch.save(model_biam.state_dict(), os.path.join(opt.save_path,"model_latest.pth"))

def test(epoch):
    print("=======================EVALUATION MODE=======================")
    logging.info("=======================EVALUATION MODE=======================")
    test_start_time = time.time()
    if not opt.train:
        model_path = os.path.join(opt.save_path, ('model_best_train_full_{}.pth').format(epoch))
    else:
        model_path = os.path.join(opt.save_path, 'model_best.pth')

    print(model_path)
    logging.info(model_path)
    model_test.load_state_dict(torch.load(model_path))
    model_test.eval()
    src = opt.src
    test_loc = os.path.join(src, 'NUS-WIDE','features', 'nus_wide_test.h5')
    test_features = h5py.File(test_loc, 'r')
    test_feature_keys = list(test_features.keys())
    image_filenames = util.load_dict(os.path.join(src, 'NUS-WIDE', 'test_img_names.pkl'))
    test_image_filenames = image_filenames['img_names']
    ntest = len(test_image_filenames)
    print(ntest)
    logging.info(ntest)

    prediction_81 = torch.empty(ntest,81)
    prediction_1006 = torch.empty(ntest,1006)
    lab_81 = torch.empty(ntest,81)
    lab_1006 = torch.empty(ntest,1006)
    test_batch_size = opt.test_batch_size
    if model_vgg is not None:
        logging.info("model vgg not none")
        model_vgg.eval()

    for m in range(0, ntest, test_batch_size):
        strt = m
        endt = min(m+test_batch_size, ntest)
        bs = endt-strt
        features, labels_1006, labels_81 = np.empty((bs,512,196)), np.empty((bs,1006)), np.empty((bs,81))
        for i, key in enumerate(test_image_filenames[strt:endt]):
            features[i,:,:] = np.float32(test_features.get(key+'-features'))
            labels_1006[i,:] =  np.int32(test_features.get(key+'-labels'))
            labels_81[i,:] =  np.int32(test_features.get(key+'-labels_81'))

        features = torch.from_numpy(features).float()
        labels_1006 = torch.from_numpy(labels_1006).long()
        labels_81 = torch.from_numpy(labels_81).long()
        with torch.no_grad():
            vgg_4096 = model_vgg(features.cuda()) #if model_vgg is not None else None
            vgg_4096 = vgg_4096.detach()
            logits_81 = model_test(features.cuda(), data.vecs_81, vgg_4096)
            logits_1006 = model_test(features.cuda(), gzsl_vecs, vgg_4096)
        
        prediction_81[strt:endt,:] = logits_81
        prediction_1006[strt:endt,:] = logits_1006
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006

    print("completed calculating predictions over all images")
    logging.info("completed calculating predictions over all images")
    logits_81_5 = prediction_81.clone()
    ap_81 = util.compute_AP(prediction_81.cuda(), lab_81.cuda())
    F1_3_81,P_3_81,R_3_81 = util.compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
    F1_5_81,P_5_81,R_5_81 = util.compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

    print('ZSL AP',torch.mean(ap_81))
    print('k=3',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
    print('k=5',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))
    logging.info('ZSL AP: %.4f',torch.mean(ap_81))
    logging.info('k=3: %.4f,%.4f,%.4f',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
    logging.info('k=5: %.4f,%.4f,%.4f',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))

    logits_1006_5 = prediction_1006.clone()
    ap_1006 = util.compute_AP(prediction_1006.cuda(), lab_1006.cuda())
    F1_3_1006,P_3_1006,R_3_1006 = util.compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
    F1_5_1006,P_5_1006,R_5_1006 = util.compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)
    print('GZSL AP',torch.mean(ap_1006))
    print('g_k=3',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
    print('g_k=5',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))
    logging.info('GZSL AP:%.4f',torch.mean(ap_1006))
    logging.info('g_k=3:%.4f,%.4f,%.4f',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
    logging.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))

    print("------------------------------------------------------------------")
    print("TEST Time: {:.4f}".format(time.time()-test_start_time))
    print("------------------------------------------------------------------")
    logging.info("------------------------------------------------------------------")
    logging.info("TEST Time: {:.4f}".format(time.time()-test_start_time))
    logging.info("------------------------------------------------------------------")

if not opt.train_full_data:
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
        train(epoch)
        if (epoch > 3 and epoch % eval_interval == 0) or epoch == num_epochs - 1:
            model_biam.eval()
            val(epoch)
            model_biam.train()
        if opt.cosinelr_scheduler:
            scheduler.step()
        if (epoch > 3 and epoch % 10 == 0) or epoch == num_epochs-1:
            test(epoch)  
else:
    src = ' results/NUS_WIDE_' + opt.pretrained_model + '/model_best.pth '
    dst = os.path.join(opt.save_path, 'model_best.pth')
    cmd = 'cp ' + src + ' ' + dst
    print(cmd)
    os.system(cmd)


## load the best model for training on full data
opt.train = False
data = util.DATA_LOADER(opt) ### INTIAL DATALOADER ###
print('===> total samples')
print(data.ntrain)
logging.info('===> total samples')
logging.info(data.ntrain)
optimizer = torch.optim.Adam(model_biam.parameters(), opt.train_full_lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))
path_chk_rest = os.path.join(opt.save_path, 'model_best.pth')
print(path_chk_rest)
logging.info(path_chk_rest)
model_biam.load_state_dict(torch.load(path_chk_rest))
start_epoch = 1

if opt.cuda:
    model_biam = model_biam.cuda()
    model_test = model_test.cuda()
    data.vecs_81 = data.vecs_81.cuda()
    data.vecs_925 = data.vecs_925.cuda()
gzsl_vecs = torch.cat([data.vecs_925,data.vecs_81],0)

for epoch in range(start_epoch, start_epoch+5):
    train(epoch)
    test(epoch)