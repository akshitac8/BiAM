#author: akshitac8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
import json
from tqdm import tqdm
import numpy as np
import h5py
import argparse
import torchvision
import tensorflow as tf
import pandas as pd
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO, filename='logs/test_feature_extract_openimages.log')


model = Net()
model = model.eval()
print(model)
GPU = True
if GPU:
    gpus = '0,1,2,3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model = model.cuda()

version = '2018_04'
path = 'datasets/OpenImages/{}'.format(version)
data_set = 'test'
df_label = pd.read_csv(path+'/{}/{}-annotations-human-imagelabels.csv'.format(data_set, data_set))
seen_labelmap_path = path+'/classes-trainable.txt'
dict_path = path+'/class-descriptions.csv'
unseen_labelmap_path = path+'/unseen_labels.pkl'
data_path = 'datasets/OpenImages/test/'


print('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])

files = []
partition_idxs = []
for idx_partition, partition in enumerate(partition_df):
    file_partition = [img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
    files.extend(file_partition)
    partition_idxs.extend([idx_partition]*len(file_partition))

import collections
duplicate_files = [item for item, count in collections.Counter(files).items() if count > 1]
feat = {key: [] for key in duplicate_files}

# np.save('full_test_partition_idxs.npy', partition_idxs)
# np.save('full_test_files.npy', files)
# partition_idxs = np.load('full_test_partition_idxs.npy')
# files = np.load('full_test_files.npy')

def LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path):
  seen_labelmap = [line.rstrip() for line in tf.gfile.GFile(seen_labelmap_path)]
  with open(unseen_labelmap_path, 'rb') as infile:
    unseen_labelmap = pickle.load(infile).tolist()
  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]
  return seen_labelmap, unseen_labelmap, label_dict


predictions_eval = 0
predictions_eval_resize = 0

seen_labelmap, unseen_labelmap, label_dict = LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
num_seen_classes = len(seen_labelmap)
num_unseen_classes = len(unseen_labelmap)
print('num_seen_classes', num_seen_classes,
      'num_unseen_classes', num_unseen_classes)

def get_label(file,partition_idx):
    img_id = file.split('.')[0] #file.decode('utf-8').split('.')[0]
    df_img_label=partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    seen_label = np.zeros(num_seen_classes,dtype=np.int32)
    unseen_label = np.zeros(num_unseen_classes,dtype=np.int32)
    for index, row in df_img_label.iterrows():
        if row['LabelName'] in seen_labelmap:
            idx=seen_labelmap.index(row['LabelName'])
            seen_label[idx] = 2*row['Confidence']-1
        if row['LabelName'] in unseen_labelmap:
            idx=unseen_labelmap.index(row['LabelName'])
            unseen_label[idx] = 2*row['Confidence']-1
    return seen_label,unseen_label

n_samples = len(files)
print("numpy array saved")
print('number of sample: {} dataset: {}'.format(n_samples, data_set))
print('number of unique sample: {} '.format(len(np.unique(files))))

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
        seen_label, unseen_label = get_label(file, partition_idx)
        filename = os.path.join(data_path, file)
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        return file.encode("ascii", "ignore"), img, seen_label, unseen_label


loader = DataLoader(dataset=DatasetExtract(), batch_size=64, shuffle=False, num_workers=32, drop_last=False)
src = 'datasets/OpenImages/test_features'
fn = os.path.join(src, 'OPENIMAGES_TEST_CONV5_4_NO_CENTERCROP.h5')
xx = {}
with h5py.File(fn, mode='w') as h5f:
    for i, data in enumerate(tqdm(loader), 0):
        _file, imgs, seen_label, unseen_label = data
        imgs = imgs.cuda()
        with torch.no_grad():
            outs = model(imgs)#.view(imgs.shape[0],512,-1)

        outs = np.float32(outs.cpu().numpy())
        seen_label = np.int8(seen_label.numpy())
        unseen_label = np.int8(unseen_label.numpy())
        bs = outs.shape[0]
        for m in range(bs):
            file = _file[m].decode("utf-8")
            if file in duplicate_files:
                if len(feat[file]) == 2:
                    seen_label[m] = seen_label[m] + feat[file][0]
                    unseen_label[m] = unseen_label[m] + feat[file][1]

                    h5f.create_dataset(file+'-features', data=outs[m], dtype=np.float32, compression="gzip")
                    h5f.create_dataset(file+'-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                    h5f.create_dataset(file+'-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")
                else:
                    feat[file].append(seen_label[m])
                    feat[file].append(unseen_label[m])
            else:
                h5f.create_dataset(file+'-features', data=outs[m], dtype=np.float32, compression="gzip")
                h5f.create_dataset(file+'-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                h5f.create_dataset(file+'-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")

test_loc = os.path.join(src, 'test_features', 'OPENIMAGES_TEST_CONV5_4_NO_CENTERCROP.h5')
test_features = h5py.File(test_loc, 'r')
test_feature_keys = list(test_features.keys())
image_names = np.unique(np.array([m.split('-')[0] for m in test_feature_keys]))
print(len(image_names))




# import pdb;pdb.set_trace()

# with h5py.File(fn, mode='a') as h5f_1:
# del h5f_1[file+'-features']
# del h5f_1[file+'-seenlabels']
# del h5f_1[file+'-unseenlabels']
# h5f_1.close()
# seen_label[m] = seen_label[m] + xx[file]['seen_label']
# unseen_label[m] = unseen_label[m] + xx[file]['unseen_label']


# xx[file] = {}
# xx[file]['seen_label'] = seen_label[m]
# xx[file]['unseen_label'] = unseen_label[m]