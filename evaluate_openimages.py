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
import socket
from torch.utils.data import DataLoader
import h5py
import pickle
import logging
import csv
import pandas as pd

if not os.path.exists("logs"):
    os.system("mkdir -p " + "logs") #os.mkdir("logs")
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

model_vgg = None
model_vgg = model.vgg_net()
model_test = model.BiAM(opt, dim_feature=[196,512])

print(model_test)
logging.info(model_test)
name='NUS_WIDE_{}'.format(opt.SESSION)
opt.save_path += '/'+name

if opt.cuda:
    model_test = model_test.cuda()
    data.vecs_400 = data.vecs_400.cuda()
    data.vecs_7186 = data.vecs_7186.cuda()
    if model_vgg is not None:
        model_vgg.cuda()
        if not opt.vgg_base_trainmode:
            model_vgg.eval()
gzsl_vecs = torch.cat([data.vecs_7186,data.vecs_400],0)


print("=======================EVALUATION MODE=======================")
logging.info("=======================EVALUATION MODE=======================")
test_start_time = time.time()

gzsl_model_path="pretrained_weights/model_best_gzsl.pth"
zsl_model_path="pretrained_weights/model_best_zsl.pth"

paths = [gzsl_model_path, zsl_model_path]
for model_path in paths:
    print(model_path)
    logging.info(model_path)
    model_test.load_state_dict(torch.load(model_path))
    logging.info("model loading finished")
    model_test.eval()

    src = opt.src
    test_loc = os.path.join(src, 'OpenImages', 'test_features_lesa', 'OPENIMAGES_TEST_CONV5_4_LESA_VGG_NO_CENTERCROP.h5')
    test_features = h5py.File(test_loc, 'r')
    test_feature_keys = list(test_features.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in test_feature_keys]))
    ntest = len(image_names)
    test_batch_size = opt.test_batch_size

    path_top_unseen = os.path.join(src, 'OpenImages','2018_04', 'top_400_unseen.csv')
    df_top_unseen = pd.read_csv(path_top_unseen, header=None)
    idx_top_unseen = df_top_unseen.values[:, 0]
    assert len(idx_top_unseen) == 400

    print('===> total TEST samples')
    print(ntest)
    logging.info('===> total TEST samples')
    logging.info(ntest)

    prediction_400 = torch.empty(ntest,400)
    prediction_7586 = torch.empty(ntest,7586)
    prediction_7186 = torch.empty(ntest,7186)
    lab_400 = torch.empty(ntest,400)
    lab_7586 = torch.empty(ntest,7586)
    lab_7186 = torch.empty(ntest,7186)

    if model_vgg is not None:
        model_vgg.eval()

    for m in range(0, ntest, test_batch_size):
        strt = m
        endt = min(m+test_batch_size, ntest)
        bs = endt-strt
        c=m
        c+=bs
        features, labels_7186, labels_2594 = np.empty((bs,512,196)), np.empty((bs,7186)), np.empty((bs,2594))
        for i, key in enumerate(image_names[strt:endt]):
            features[i,:,:] = np.float32(test_features.get(key+'-features'))
            labels_7186[i,:] =  np.int32(test_features.get(key+'-seenlabels'))
            labels_2594[i,:] =  np.int32(test_features.get(key+'-unseenlabels'))

        features = torch.from_numpy(features).float()
        labels_7186 = torch.from_numpy(labels_7186).long()
        labels_400 = torch.from_numpy(labels_2594).long()[:,idx_top_unseen]
        labels_7586 = torch.cat((labels_7186,labels_400),1)

        with torch.no_grad():
            vgg_4096 = model_vgg(features.cuda()) if model_vgg is not None else None
            logits_400 = model_test(features.cuda(), data.vecs_400, vgg_4096)
            logits_7586  = model_test(features.cuda(), gzsl_vecs, vgg_4096) ##seen-unseen
            logits_7186 = model_test(features.cuda(), data.vecs_7186, vgg_4096) ##seen-unseen
        
        prediction_400[strt:endt,:] = logits_400
        prediction_7586[strt:endt,:] = logits_7586
        prediction_7186[strt:endt,:] = logits_7186

        lab_400[strt:endt,:] = labels_400
        lab_7586[strt:endt,:] = labels_7586
        lab_7186[strt:endt,:] = labels_7186

    print(("completed calculating predictions over all {} images".format(c)))
    logging.info(("completed calculating predictions over all {} images".format(c)))

    ############################# SEEN ##############################################
    lab_7186 = lab_7186.cuda()
    prediction_7186 = prediction_7186.cuda()
    temp_7186 = torch.clamp(lab_7186,0,1).sum(1).nonzero().flatten() ## take only the images with positive annotations
    lab_7186 = lab_7186[temp_7186]
    prediction_7186 = prediction_7186[temp_7186]

    ## AP ##
    temp_lab_7186=(lab_7186!=0)
    temp_lab_7186 = torch.clamp(temp_lab_7186,0,1)
    mask = temp_lab_7186.sum(0).nonzero().flatten()

    map_lab_7186 = lab_7186[:,mask]
    imgs_per_label = torch.clamp(map_lab_7186,0,1).sum(0)

    map_prediction_7186 = prediction_7186[:,mask]

    ap_7186 = util.compute_AP(map_prediction_7186, map_lab_7186)
    print('SEEN AP on 4728 classes',torch.mean(ap_7186).item())
    logging.info('SEEN AP on 4728 classes:%.4f',torch.mean(ap_7186).item())

    weighted_map_7186 = (imgs_per_label.float() * ap_7186).sum()/imgs_per_label.sum().float()
    print('WEIGHTED SEEN AP on 4728 classes',weighted_map_7186.item())
    logging.info('WEIGHTED SEEN AP on 4728 classes:%.4f',weighted_map_7186.item())

    del weighted_map_7186, ap_7186, imgs_per_label, temp_lab_7186, lab_7186, prediction_7186, mask
    torch.cuda.empty_cache()

    logits_7186_20 = map_prediction_7186.clone()

    F1_20_7186,P_20_7186,R_20_7186 = util.compute_F1(map_prediction_7186, map_lab_7186, 'overall', k_val=20)
    print('g_k=20',torch.mean(F1_20_7186).item(), torch.mean(P_20_7186).item(), torch.mean(R_20_7186).item())
    logging.info('g_k=20:%.4f,%.4f,%.4f',torch.mean(F1_20_7186).item(), torch.mean(P_20_7186).item(), torch.mean(R_20_7186).item())

    del map_prediction_7186
    torch.cuda.empty_cache()

    logits_7186_5 = logits_7186_20.clone()

    F1_10_7186,P_10_7186,R_10_7186 = util.compute_F1(logits_7186_20, map_lab_7186, 'overall', k_val=10)
    print('g_k=10',torch.mean(F1_10_7186).item(), torch.mean(P_10_7186).item(), torch.mean(R_10_7186).item())
    logging.info('g_k=10:%.4f,%.4f,%.4f',torch.mean(F1_10_7186).item(), torch.mean(P_10_7186).item(), torch.mean(R_10_7186).item())

    del logits_7186_20
    torch.cuda.empty_cache()

    logits_7186_1 = logits_7186_5.clone()

    F1_5_7186,P_5_7186,R_5_7186 = util.compute_F1(logits_7186_5, map_lab_7186, 'overall', k_val=5)
    print('g_k=5',torch.mean(F1_5_7186).item(), torch.mean(P_5_7186).item(), torch.mean(R_5_7186).item())
    logging.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_7186).item(), torch.mean(P_5_7186).item(), torch.mean(R_5_7186).item())


    del logits_7186_5
    torch.cuda.empty_cache()

    F1_1_7186,P_1_7186,R_1_7186 = util.compute_F1(logits_7186_1, map_lab_7186, 'overall', k_val=1)
    print('g_k=1',torch.mean(F1_1_7186).item(), torch.mean(P_1_7186).item(), torch.mean(R_1_7186).item())
    logging.info('g_k=1:%.4f,%.4f,%.4f',torch.mean(F1_1_7186).item(), torch.mean(P_1_7186).item(), torch.mean(R_1_7186).item())

    del logits_7186_1
    torch.cuda.empty_cache()

    del map_lab_7186, F1_20_7186, P_20_7186, R_20_7186, F1_10_7186, P_10_7186, R_10_7186, F1_5_7186, P_5_7186, R_5_7186, F1_1_7186, P_1_7186, R_1_7186
    torch.cuda.empty_cache()
    
    ################################################################################################################################
    ############################# ZSL ##############ß################################
    prediction_400 = prediction_400.cuda()
    lab_400 = lab_400.cuda()

    temp_400 = torch.clamp(lab_400,0,1).sum(1).nonzero().flatten() ## take only the images with positive annotations
    lab_400 = lab_400[temp_400]
    prediction_400 = prediction_400[temp_400]

    logits_400_20 = prediction_400.clone()
    logits_400_3 = prediction_400.clone()
    logits_400_1 = prediction_400.clone()


    ap_400 = util.compute_AP(prediction_400, lab_400)
    print('ZSL AP',torch.mean(ap_400).item())
    logging.info('ZSL AP: %.4f',torch.mean(ap_400).item())

    imgs_per_label = torch.clamp(lab_400,0,1).sum(0)
    weighted_map_400 = (imgs_per_label.float() * ap_400).sum()/imgs_per_label.sum().float()

    print('WEIGHTED ZSL AP',torch.mean(weighted_map_400).item())
    logging.info('WEIGHTED ZSL AP: %.4f',torch.mean(weighted_map_400).item())

    F1_20_400,P_20_400,R_20_400 = util.compute_F1(logits_400_20, lab_400, 'overall', k_val=20)
    print('k=20',torch.mean(F1_20_400).item(),torch.mean(P_20_400).item(),torch.mean(R_20_400).item())
    logging.info('k=20: %.4f,%.4f,%.4f',torch.mean(F1_20_400).item(),torch.mean(P_20_400).item(),torch.mean(R_20_400).item())

    del logits_400_20, temp_400
    torch.cuda.empty_cache()

    F1_10_400,P_10_400,R_10_400 = util.compute_F1(prediction_400, lab_400, 'overall', k_val=10)
    print('k=10',torch.mean(F1_10_400).item(),torch.mean(P_10_400).item(),torch.mean(R_10_400).item())
    logging.info('k=10: %.4f,%.4f,%.4f',torch.mean(F1_10_400).item(),torch.mean(P_10_400).item(),torch.mean(R_10_400).item())

    del prediction_400
    torch.cuda.empty_cache()

    F1_3_400,P_3_400,R_3_400 = util.compute_F1(logits_400_3, lab_400, 'overall', k_val=3)
    print('k=3',torch.mean(F1_3_400).item(),torch.mean(P_3_400).item(),torch.mean(R_3_400).item())
    logging.info('k=3: %.4f,%.4f,%.4f',torch.mean(F1_3_400).item(),torch.mean(P_3_400).item(),torch.mean(R_3_400).item())

    del logits_400_3
    torch.cuda.empty_cache()

    F1_1_400,P_1_400,R_1_400 = util.compute_F1(logits_400_1, lab_400, 'overall', k_val=1)
    print('k=1',torch.mean(F1_1_400).item(),torch.mean(P_1_400).item(),torch.mean(R_1_400).item())
    logging.info('k=1: %.4f,%.4f,%.4f',torch.mean(F1_1_400).item(),torch.mean(P_1_400).item(),torch.mean(R_1_400).item())

    del logits_400_1
    torch.cuda.empty_cache()

    del features, lab_400
    torch.cuda.empty_cache()

    ############################# GZSL ##############################################
    lab_7586 = lab_7586.cuda()
    prediction_7586 = prediction_7586.cuda()

    temp_7586 = torch.clamp(lab_7586,0,1).sum(1).nonzero().flatten() ## take only the images with positive annotations
    lab_7586 = lab_7586[temp_7586]
    prediction_7586 = prediction_7586[temp_7586]

    ## AP ##
    temp_lab_7586=(lab_7586!=0)
    temp_lab_7586 = torch.clamp(temp_lab_7586,0,1)
    mask = temp_lab_7586.sum(0).nonzero().flatten()
    # imgs_per_label = temp_lab_7586.sum(0)

    map_lab_7586 = lab_7586[:,mask]
    imgs_per_label = torch.clamp(map_lab_7586,0,1).sum(0)

    map_prediction_7586 = prediction_7586[:,mask]

    ap_7586 = util.compute_AP(map_prediction_7586, map_lab_7586)
    print('GZSL AP on 4728+400 classes',torch.mean(ap_7586).item())
    logging.info('GZSL AP on 4728+400 classes:%.4f',torch.mean(ap_7586).item())

    weighted_map_7586 = (imgs_per_label.float() * ap_7586).sum()/imgs_per_label.sum().float()
    print('WEIGHTED GZSL AP on 4728+400 classes',weighted_map_7586.item())
    logging.info('WEIGHTED GZSL AP on 4728+400 classes:%.4f',weighted_map_7586.item())

    del weighted_map_7586, ap_7586, imgs_per_label, map_lab_7586, map_prediction_7586, temp_lab_7586, mask
    torch.cuda.empty_cache()

    logits_7586_20 = prediction_7586.clone()

    F1_20_7586,P_20_7586,R_20_7586 = util.compute_F1(prediction_7586, lab_7586, 'overall', k_val=20)
    print('g_k=20',torch.mean(F1_20_7586).item(), torch.mean(P_20_7586).item(), torch.mean(R_20_7586).item())
    logging.info('g_k=20:%.4f,%.4f,%.4f',torch.mean(F1_20_7586).item(), torch.mean(P_20_7586).item(), torch.mean(R_20_7586).item())

    del prediction_7586
    torch.cuda.empty_cache()

    logits_7586_5 = logits_7586_20.clone()

    F1_10_7586,P_10_7586,R_10_7586 = util.compute_F1(logits_7586_20, lab_7586, 'overall', k_val=10)
    print('g_k=10',torch.mean(F1_10_7586).item(), torch.mean(P_10_7586).item(), torch.mean(R_10_7586).item())
    logging.info('g_k=10:%.4f,%.4f,%.4f',torch.mean(F1_10_7586).item(), torch.mean(P_10_7586).item(), torch.mean(R_10_7586).item())

    del logits_7586_20
    torch.cuda.empty_cache()

    logits_7586_1 = logits_7586_5.clone()

    F1_5_7586,P_5_7586,R_5_7586 = util.compute_F1(logits_7586_5, lab_7586, 'overall', k_val=5)
    print('g_k=5',torch.mean(F1_5_7586).item(), torch.mean(P_5_7586).item(), torch.mean(R_5_7586).item())
    logging.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_7586).item(), torch.mean(P_5_7586).item(), torch.mean(R_5_7586).item())

    del logits_7586_5
    torch.cuda.empty_cache()

    F1_1_7586,P_1_7586,R_1_7586 = util.compute_F1(logits_7586_1, lab_7586, 'overall', k_val=1)
    print('g_k=1',torch.mean(F1_1_7586).item(), torch.mean(P_1_7586).item(), torch.mean(R_1_7586).item())
    logging.info('g_k=1:%.4f,%.4f,%.4f',torch.mean(F1_1_7586).item(), torch.mean(P_1_7586).item(), torch.mean(R_1_7586).item())

    del logits_7586_1
    torch.cuda.empty_cache()

    del lab_7586, F1_20_7586, P_20_7586, R_20_7586, F1_10_7586, P_10_7586, R_10_7586, F1_5_7586, P_5_7586, R_5_7586, F1_1_7586, P_1_7586, R_1_7586
    torch.cuda.empty_cache()

    print("------------------------------------------------------------------")
    print("TEST Time: {:.4f}".format(time.time()-test_start_time))
    print("------------------------------------------------------------------")

    logging.info("------------------------------------------------------------------")
    logging.info("TEST Time: {:.4f}".format(time.time()-test_start_time))
    logging.info("------------------------------------------------------------------")
