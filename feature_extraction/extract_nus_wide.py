import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
import json
from tqdm import tqdm
import numpy as np
import h5py
from data import get_extract_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(1234)
import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict   

model = Net()
model = model.eval()
print(model)
GPU = True
if GPU:
    gpus = '0,1,2,3,4,5,6,7'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    if len(device_ids)>1:
        model = nn.DataParallel(model, device_ids = device_ids).cuda()
    else:
        model = model.cuda()

src = 'datasets/NUS-WIDE/features/'
jsons = ['81_only_full_nus_wide_train', '81_only_full_nus_wide_test']

#     img_names = []
#     for data_ in tqdm(data_loader):
#         filenames = data_[0]
#         for filename in filenames:
#             img_names.append(filename.replace('/', '_'))
#     dict_ = {'img_names':img_names}
#     save_dict(dict_, os.path.join(src, 'test_img_names.pkl'))
    
for json_ in jsons:
    if json_ == '81_only_full_nus_wide_train':
        pickle_name='img_names_train_81.pkl'
    else:
        pickle_name='img_names_test_81.pkl'

    type_ = 'Flickr'
    dataset_ = get_extract_data(
        dir_ = os.path.join('/home/ag1/akshita/multi-label-zsl','data/{}'.format(type_)),
        json_file = os.path.join(src, json_+'.json')) 

    data_loader = DataLoader(dataset=dataset_, batch_size=128, shuffle=False, num_workers=32, drop_last=False)
    img_names = []

    fn = os.path.join(src, json_+'_CONV5_4_VGG_NO_CENTERCROP_compressed_gzip.h5')

    with h5py.File(fn, mode='w') as h5f:
        for data_ in tqdm(data_loader):
            filename, img, lab = data_[0], data_[1], data_[2]
            # filename, img, lab, lab_81, lab_925 = data_[0], data_[1], data_[2], data_[3], data_[4]
            bs = img.size(0)
            if GPU:
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            out = np.float32(out.cpu().numpy())
            labels = np.int8(lab.numpy())
            # labels_81 = np.int8(lab_81.numpy())
            # labels_925 = np.int8(lab_925.numpy())

            # import pdb;pdb.set_trace()

            for i in range(bs):
                if np.isnan(out[i].any()):
                    print(filename[i])
                    import pdb; pdb.set_trace()
                img_names.append(filename[i].replace('/', '_'))
                h5f.create_dataset(filename[i].replace('/', '_')+'-features', data=out[i], dtype=np.float32, compression="gzip")
                h5f.create_dataset(filename[i].replace('/', '_')+'-labels', data=labels[i], dtype=np.int8, compression="gzip")

        dict_ = {'img_names':img_names}
        save_dict(dict_, os.path.join(src, pickle_name))

    h5f.close()