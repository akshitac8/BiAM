import os
from torch.utils.data import Dataset
from PIL import Image
import json
import torch 
import numpy as np
import torchvision.transforms as transforms

#NOTE: IMAGE TRANSFORMATION ACCORDING TO LESA
transform=transforms.Compose([
                    transforms.Resize((224,224)),  #bilinear interpolation
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

class DatasetExtract(Dataset):
    def __init__(self, dir_, json_file):
        super(DatasetExtract, self).__init__()

        self.json_file = json_file
        self.data_ = json.load(open(json_file,'r'))

        self.dir_ = dir_
        self.keys_ = list(self.data_.keys())

    def __len__(self):
        return len(self.keys_)

    def __getitem__(self, index):
        key = self.keys_[index]
        val = self.data_[key]

        filename = os.path.join(self.dir_, key)
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        labels = torch.tensor(np.int_(val['labels']))
   
        return filename, img, labels

        #for ONLY coco
        # key = os.path.split(key)[-1]
        # if os.path.split( self.dir_)[-1] != 'images':
        #     self.dir_ = self.dir_ + '/images'
        # print(self.dir_)
