#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: naraysa & akshitac8
"""

import torch
from sklearn.preprocessing import normalize
import os
import pickle
import h5py
import time
import pandas as pd
import numpy as np
import random

random.seed(3483)
np.random.seed(3483)

## when seed doesn't reproduce the number save random states
# rand_states = np.load('random_states.npy', allow_pickle=True)[0]
# torch.set_rng_state(torch.from_numpy(rand_states[2]))
# torch.cuda.set_rng_state(torch.from_numpy(rand_states[3]))

class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
    def get_max(self,col):
        return np.max(self.df[col])
    def get_min(self,col):
        return np.min(self.df[col])


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr  
        
def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).cuda()
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        mask = label.abs() == 1
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label[mask], min=0, max=1)
        sorted_pred, sort_idx = prediction[mask].sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap

def compute_F1(predictions, labels, mode_F1, k_val):
    idx = predictions.topk(dim=1, k=k_val)[1]
    predictions.fill_(0)
    predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).cuda())
    if mode_F1 == 'overall':
        # print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = (labels[mask] == 1).sum().float()
        tpfp = mask.sum().float()
        tpfn = (labels == 1).sum().float()
        p = TP / tpfp
        r = TP/tpfn
        f1 = 2*p*r/(p+r)
    else:
        num_class = predictions.shape[1]
        # print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:, idx_cls])
            label = np.squeeze(labels[:, idx_cls])
            if np.sum(label > 0) == 0:
                continue
            binary_label = np.clip(label, 0, 1)
            f1[idx_cls] = f1_score(binary_label, prediction)
            p[idx_cls] = precision_score(binary_label, prediction)
            r[idx_cls] = recall_score(binary_label, prediction)
    
    return f1, p, r

def get_seen_unseen_classes(file_tag1k, file_tag81):
    with open(file_tag1k, "r") as file:
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81, "r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx, unseen_cls_idx

import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict   

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)

    def read_matdataset(self, opt):
        tic = time.time()
        print("Data loading started")
        self.src = opt.src
        att_path = os.path.join(self.src,'NUS-WIDE', 'wiki_contexts','NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
        file_tag1k = os.path.join(self.src,'NUS-WIDE', 'NUS_WID_Tags','TagList1k.txt')
        file_tag81 = os.path.join(self.src,'NUS-WIDE', 'ConceptsList','Concepts81.txt')
        self.seen_cls_idx, _ = get_seen_unseen_classes(file_tag1k, file_tag81)
        src_att = pickle.load(open(att_path, 'rb'))
        self.vecs_925 = torch.from_numpy(normalize(src_att[0][self.seen_cls_idx]))
        self.vecs_81 = torch.from_numpy(normalize(src_att[1]))

        train_loc = os.path.join(self.src, 'NUS-WIDE', 'features' ,'nus_wide_train.h5')
        self.train_features = h5py.File(train_loc, 'r')
        img_names = load_dict(os.path.join(self.src, 'NUS-WIDE', 'img_names.pkl'))
        self.image_filenames = img_names['img_names']

        if opt.train:
            print("SPLIT TRAIN DATA INTO TRAIN AND VAL")
            train_seen_idx = np.arange(int(0.8*(len(self.image_filenames))))
            val_seen_idx = np.arange(int(0.8*(len(self.image_filenames))), len(self.image_filenames))
            assert len(np.intersect1d(train_seen_idx,val_seen_idx)) == 0
            self.train_image_names = np.array(self.image_filenames)[train_seen_idx]
            self.val_image_names = np.array(self.image_filenames)[val_seen_idx]
        else:
            print("USING FULL TRAIN DATA")
            self.train_image_names = np.array(self.image_filenames)

        self.ntrain = len(self.train_image_names)
        print("Data loading finished, Time taken: {}".format(time.time()-tic))

    def next_train_batch(self, batch_size):
        batch_features, batch_labels = np.empty((batch_size,512,196)), np.empty((batch_size,925))
        idx = torch.randperm(self.ntrain)[0:batch_size]
        for i, key in enumerate(self.train_image_names[idx]):
            try:
                batch_features[i,:,:] = np.float32(self.train_features.get(key+'-features'))
                batch_labels[i,:] = np.int32(self.train_features.get(key+'-labels'))
            except:
                continue

        batch_features = torch.from_numpy(batch_features).float()
        batch_labels = torch.from_numpy(batch_labels).long()

        return batch_features, batch_labels

    def next_val(self):
        val_train_feature, val_train_label_925, val_train_label_81 = \
                        np.empty((len(self.val_image_names),512,196)), np.empty((len(self.val_image_names),925)), np.empty((len(self.val_image_names),81))

        for i, key in enumerate(self.val_image_names):
            try:
                val_train_feature[i,:,:] = np.float32(self.train_features.get(key+'-features'))
                val_train_label_925[i,:] = np.int32(self.train_features.get(key+'-labels'))
                val_train_label_81[i,:] = np.int32(self.train_features.get(key+'-labels_81'))
            except:
                continue

        val_train_feature = torch.from_numpy(val_train_feature).float()
        val_train_label_925 = torch.from_numpy(val_train_label_925).long()
        val_train_label_81 = torch.from_numpy(val_train_label_81).long()

        return val_train_feature, val_train_label_925, val_train_label_81
