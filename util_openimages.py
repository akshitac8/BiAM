#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: naraysa & akshitac8
"""

import torch
import numpy as np
import random
from sklearn.preprocessing import normalize
import os
import pickle
import h5py
import time
import pandas as pd
from glob import glob
import torch.utils.data as data
import pickle

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
          
def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).cuda()
    for idx_cls in range(num_class):
        prediction = predictions[(labels != 0)[:, idx_cls]]
        label = labels[(labels != 0)[:, idx_cls]]
        mask = label.abs() == 1
        if (label > 0).sum() == 0:
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

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message 
    
class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)

    def read_matdataset(self, opt):
        tic = time.time()
        print("Data loading started")
        data_set = 'train'
        print(data_set)
        src = opt.src
        path = os.path.join(src, 'openimages','2018_04')
        att_path = os.path.join(src, 'wiki_contexts','OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl')
        print('loading data')
        self.h5_path = os.path.join(src, 'openimages','train_features_lesa')
        self.h5_files = glob(os.path.join(self.h5_path, '*.h5'))
        self.ntrain = len(self.h5_files)
        print('number of batches : {} version: {}'.format(self.ntrain, data_set))
        path_top_unseen = os.path.join(path,'top_400_unseen.csv')
        df_top_unseen = pd.read_csv(path_top_unseen, header=None)
        self.idx_top_unseen = df_top_unseen.values[:, 0]
        assert len(self.idx_top_unseen) == 400
        src_att = pickle.load(open(att_path, 'rb'))
        self.vecs_7186 = torch.from_numpy(normalize(src_att[0]))
        self.vecs_400 = torch.from_numpy(normalize(src_att[1][self.idx_top_unseen,:]))
        print("Data loading finished, Time taken: {}".format(time.time()-tic))

    def train_data(self):
        idx = torch.randperm(len(self.h5_files))[0] #randomly return single h5_file from the folder
        filename = self.h5_files[idx]
        train_loc = filename #os.path.join(self.h5_path, filename)
        train_features = h5py.File(train_loc, 'r')
        return train_features

    def next_train_batch(self, batch_size, train_features, image_names, batch):
        batch_features, batch_labels = np.empty((batch_size,512,196)), np.empty((batch_size,7186))
        train_image_names = image_names[batch:batch+batch_size]
        for i, key in enumerate(train_image_names):
            try:
                batch_features[i,:,:] = np.float32(train_features.get(key+'-features'))
                batch_labels[i,:] = np.int32(train_features.get(key+'-seenlabels'))
            except:
                continue

        batch_features = torch.from_numpy(batch_features).float()
        batch_labels = torch.from_numpy(batch_labels).long()
        return batch_features, batch_labels

def load_checkpoint(model_test, model_path):
        print("* Loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model_dict = model_test.state_dict()
        for k, v in checkpoint.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
            else:
                print ('\tMismatched layers: {}'.format(k))
        model_test.load_state_dict(model_dict)
        print(" Loading succeed ")
