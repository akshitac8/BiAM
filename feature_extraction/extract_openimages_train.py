#author: akshitac8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import pandas as pd
import tensorflow as tf
import torchvision
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import joblib
import time
import concurrent.futures
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from model import Net

import logging
logging.basicConfig(level=logging.INFO, filename='logs/openimages_LESA_features_trainset_new_resnet.log')
import os
import sys
pwd = os.getcwd()
sys.path.insert(0, pwd)
logging.info('-'*30)
logging.info(os.getcwd())
logging.info('-'*30)

model = Net()

model = model.eval()
logging.info(model)
GPU = True
if GPU:
    gpus = '0,1,2,3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      logging.info(("\n\nLet's use {} GPUs!\n\n".format(torch.cuda.device_count())))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model = model.cuda()

version = '2018_04'
path = 'datasets/OpenImages/{}'.format(version)
data_set = 'train'
df_label = pd.read_csv(path+'/{}/{}-annotations-human-imagelabels.csv'.format(data_set, data_set))
logging.info(np.unique(df_label['ImageID']).shape[0])
labelmap_path = path+'/classes-trainable.txt'
dict_path = path+'/class-descriptions.csv'
data_path = 'datasets/OpenImages/train/'

logging.info('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])

##reading files from the local open-images folder
# files = []
# partition_idxs = []
# for idx_partition, partition in enumerate(partition_df):
#     file_partition = [img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
#     files.extend(file_partition)
#     partition_idxs.extend([idx_partition]*len(file_partition))

partition_idxs = np.load('full_train_partition_idxs.npy')
files =  np.load('full_train_files.npy')

import collections
duplicate_files = [item for item, count in collections.Counter(files).items() if count > 1]
feat = {key: [] for key in duplicate_files}

n_samples = len(files)
logging.info("numpy array saved")
logging.info('number of sample: {} dataset: {}'.format(n_samples, data_set))

def LoadLabelMap(labelmap_path, dict_path):
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]
    label_dict = {}
    for line in tf.gfile.GFile(dict_path):
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]

    return labelmap, label_dict

import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict   

predictions_eval = 0
predictions_eval_resize = 0

seen_labelmap, label_dict = LoadLabelMap(labelmap_path, dict_path)
num_classes = len(seen_labelmap)
logging.info(('num_class {}'.format(num_classes)))


def get_label(file,partition_idx):
    # img_id = file.decode('utf-8').split('.')[0]
    img_id = file.split('.')[0]
    df_img_label=partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    label = np.zeros(num_classes,dtype=np.int32)
    #logging.info(len(df_img_label))
    for index, row in df_img_label.iterrows():
        if row['LabelName'] not in seen_labelmap:
            continue #not trainable classes
        idx=seen_labelmap.index(row['LabelName'])
        label[idx] = 2*row['Confidence']-1
    #logging.info(partition_idx)
    return label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

scaler = MinMaxScaler()


class DatasetExtract(Dataset):
    def __init__(self):
        super(DatasetExtract, self).__init__()

    def __len__(self):
        return len(files)

    def __getitem__(self, index):

        file, partition_idx = files[index], partition_idxs[index]

        seen_label  = get_label(file, partition_idx)
        filename = os.path.join(data_path, file)
        try:
            img = Image.open(filename).convert('RGB')
            img = transform(img)
            flag=0
        except:
            img = torch.zeros(3,224,224)
            flag=1
        return file.encode("ascii", "ignore"), img, seen_label, flag

loader = DataLoader(dataset=DatasetExtract(), batch_size=128, shuffle=True, num_workers=64, drop_last=False)
src = 'datasets/OpenImages/train_features'
_iter = 0
count = 0
n=0
c=0
xx = []
_xx = {}
scaler = MinMaxScaler()
_imgs, features, seenlabels = np.empty((640), dtype=np.dtype('U20')), np.empty((640,1024,196)), np.empty((640,7186))
for i, data in enumerate(tqdm(loader), 0):
    count = n*128
    logging.info(count)
    logging.info(i)
    _iter+=1
    _file, imgs, seen_label, flag = data
    keep = []
    for j,f in enumerate(flag):
        if f == 0:
            keep.append(j)
    _file, imgs, seen_label = np.array(_file)[keep], imgs[keep], seen_label[keep]

    strt = count
    endt = count + _file.shape[0]

    with torch.no_grad():
        outs = model(imgs)#.view(imgs.shape[0],512,-1)

    outs = np.float32(outs.cpu().numpy())
    seen_label = np.int8(seen_label.numpy())

    _imgs[strt:endt,] = _file
    features[strt:endt,:] = outs
    seenlabels[strt:endt,:] = seen_label
    c+=endt
    n+=1
    if _iter%5 == 0:
        tic1 = time.time()
        bs = _imgs.shape[0]
        filename = ('CHUNK_{}_CONV5_4_NO_CENTERCROP.h5').format(_iter)
        fn = os.path.join(src, filename)
        with h5py.File(fn, mode='w') as h5f:
            for m in range(bs):
                dict_files = _imgs[m]#.decode("utf-8")
                if dict_files in duplicate_files:
                    if len(feat[dict_files]) == 1:
                        seenlabels[m] = seenlabels[m] + feat[dict_files]
                        h5f.create_dataset(dict_files+'-features', data=features[m], dtype=np.float32, compression="gzip")
                        h5f.create_dataset(dict_files+'-seenlabels', data=seenlabels[m], dtype=np.int8, compression="gzip")
                    else:
                        feat[dict_files].append(seenlabels[m])
                else:
                    h5f.create_dataset(dict_files+'-features', data=features[m], dtype=np.float32, compression="gzip")
                    h5f.create_dataset(dict_files+'-seenlabels', data=seenlabels[m], dtype=np.int8, compression="gzip")
        _imgs, features, seenlabels = np.empty((640), dtype=np.dtype('U20')), np.empty((640,1024,196)), np.empty((640,7186))
        n = 0
        c = 0
        logging.info("Data saving finished in {}".format(time.time()-tic1))
    logging.info(c)
