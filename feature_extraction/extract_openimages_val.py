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
from model import Net
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import joblib
import time
import concurrent.futures
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import logging
logging.basicConfig(level=logging.INFO, filename='logs/openimages_val.log')
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
data_set = 'validation'
df_label = pd.read_csv(path+'/{}/{}-annotations-human-imagelabels.csv'.format(data_set, data_set))
logging.info(np.unique(df_label['ImageID']).shape[0])
labelmap_path = path+'/classes-trainable.txt'
dict_path = path+'/class-descriptions.csv'
data_path = 'datasets/OpenImages/validation/'

logging.info('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])

##reading files from the local open-images folder
files = []
partition_idxs = []
for idx_partition, partition in enumerate(partition_df):
    file_partition = [img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
    files.extend(file_partition)
    partition_idxs.extend([idx_partition]*len(file_partition))

# partition_idxs = np.load('full_train_partition_idxs.npy')
# files =  np.load('full_train_files.npy')

n_samples = len(files)
logging.info("numpy array saved")
logging.info('number of sample: {} dataset: {}'.format(n_samples, data_set))
print('number of sample: {} dataset: {}'.format(n_samples, data_set))
print('number of unique sample: {}'.format(len(np.unique(files))))

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

loader = DataLoader(dataset=DatasetExtract(), batch_size=1, shuffle=True, num_workers=1, drop_last=False)
src = 'datasets/OpenImages/val_features_lesa'
fn = os.path.join(src, 'OPENIMAGES_VAL_CONV5_4_NO_CENTERCROP.h5')
xx = {}
with h5py.File(fn, mode='w') as h5f:
    for i, data in enumerate(loader, 0):
        file, imgs, seen_label, flag = data
        keep = []
        for j,f in enumerate(flag):
            if f == 0:
                keep.append(j)
        file, imgs, seen_label = np.array(file)[keep], imgs[keep], seen_label[keep]
        imgs = imgs.cuda()
        with torch.no_grad():
            outs = model(imgs)
        outs = np.float32(outs.cpu().numpy())
        seen_label = np.int8(seen_label.numpy())
        bs = outs.shape[0]
        
        for m in range(bs):
            file = file[m].decode("utf-8")
            print(file)
            if file in xx.keys():
                with h5py.File(fn, mode='a') as h5f_1:
                    del h5f_1[file+'-features']
                    del h5f_1[file+'-seenlabels']
                h5f_1.close()
                seen_label[m] = seen_label[m] + xx[file]['seen_label']

            
            xx[file] = {}
            xx[file]['seen_label'] = seen_label[m]

            h5f.create_dataset(file+'-features', data=outs[m], dtype=np.float32, compression="gzip")
            h5f.create_dataset(file+'-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")


# src = '../data'
val_loc = os.path.join(src, 'val_features_lesa', 'OPENIMAGES_VAL_CONV5_4_LESA_VGG_NO_CENTERCROP_compressed_gzip.h5')
val_features = h5py.File(val_loc, 'r')
val_feature_keys = list(val_features.keys())
image_names = np.unique(np.array([m.split('-')[0] for m in val_feature_keys]))
nval = len(image_names)
print(nval)