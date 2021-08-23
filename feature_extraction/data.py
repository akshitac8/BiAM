import os
from dataset_nus import DatasetExtract

def get_extract_data(dir_, json_file):
    assert os.path.exists(dir_) , ('{} does not exist'.format(dir_))
    assert os.path.isfile(json_file) , ('{} does not exist'.format(json_file))
    #assert len(cats)!=0 , ('{} should be >0'.format(len(cats)))
    return DatasetExtract(dir_, json_file)

