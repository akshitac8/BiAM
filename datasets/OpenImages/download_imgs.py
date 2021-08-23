import json
import os
import random
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import pandas as pd
import pdb
from PIL import Image
from io import BytesIO
import sys
import hashlib
import base64
import urllib3
import math
import tensorflow as tf
#%%
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Downloads all image files contained in dataset, if an image fails to download lets skip it.
n_jobs = 32
path = './data/OpenImages/2018_04/'
# This is a nice parallel processing tool that uses tqdm
# to help visualize time-to-completion.
def parallel_process(array, function, n_jobs, use_kwargs=False, front_num=1):
    """
        A parallel version of the map function with a progress bar.
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    print('run worker')
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'image',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def process_images(saved_images_path, image_content):
    
    im = Image.open(BytesIO(image_content))
    im.verify()
    im = Image.open(BytesIO(image_content))
    (w,h) = im.size
    small_side= min(w,h)
    large_side = max(w,h)
    ratio = small_side/256.0
    #im=PreprocessImage(im)
    new_size = math.ceil(large_side/ratio)
    im.thumbnail((new_size, new_size))
    im.save(saved_images_path, 'JPEG')

def checksum(str_md5,image_content):
    return base64.b64decode(str_md5) == hashlib.md5(image_content).digest()

def download(element):
    image_content = None
    dir_path = save_directory_path
    if os.path.isfile(os.path.join(dir_path, element[1]+'.jpg')):
        return
    for i in range(5):
        try:
            response = requests.get(element[0],headers=random.choice(browser_headers),verify=False,allow_redirects=False)
            image_content = response.content
            if not checksum(element[2],image_content):
                continue
        except:
            pass
        if image_content:
            try:
                complete_file_path = os.path.join(dir_path, element[1]+'.jpg')
                process_images(complete_file_path,image_content)
                return
            except:
                print(sys.exc_info()[0])
        else:
            print('drop image')
#%%
labelmap_path=path+'classes-trainable.txt'
dict_path=path+'class-descriptions.csv'
def LoadLabelMap():
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]
    label_dict = {}
    for line in tf.gfile.GFile(dict_path):
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]
    return labelmap, label_dict

labelmap, label_dict = LoadLabelMap()
#%%
def get_groundtruth(idx):
    idx_name=labelmap[idx]#df_train_name.iloc[idx,1]
    return  df_label[df_label['LabelName']==idx_name]['ImageID'].values

def get_img_class(idx,df_image):
    if idx == -1:
        print('download all images')
        return list(zip(df_image['OriginalURL'].values,df_image['ImageID'],df_image['OriginalMD5']))
    print('download image class: '+label_dict[labelmap[idx]])
    img_ids=get_groundtruth(idx)
    urls = []
    md5s = []
    for img_id in img_ids:
        info=df_image[df_image['ImageID']==img_id]
        urls.append(info['OriginalURL'].values[0])
        md5s.append(info['OriginalMD5'].values[0])
    return list(zip(urls,img_ids,md5s))
#%%
if __name__ == "__main__":
    browser_headers = [
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704 Safari/537.36"},
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:44.0) Gecko/20100101 Firefox/44.0"}
    ]
    print('load data')
    data_set = 'train'
    df_label = pd.read_csv(path+data_set+'/annotations-human.csv')
    df_image = pd.read_csv(path+data_set+'/images.csv')
    print('extract image class')
    image_infos = get_img_class(-1,df_image)
    save_directory_path = './image_data/'+data_set+'/'
    print('start downloading')
    res=parallel_process(image_infos, download,n_jobs)
