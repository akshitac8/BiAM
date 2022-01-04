[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-based-multi-label-zero/multi-label-zero-shot-learning-on-nus-wide)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-nus-wide?p=discriminative-region-based-multi-label-zero)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-based-multi-label-zero/multi-label-zero-shot-learning-on-open-images)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-open-images?p=discriminative-region-based-multi-label-zero)
# Discriminative Region-based Multi-Label Zero-Shot Learning (ICCV 2021)
[[`Paper`](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayan_Discriminative_Region-Based_Multi-Label_Zero-Shot_Learning_ICCV_2021_paper.pdf)][[`Video`](https://www.youtube.com/watch?v=0MZxWozdRiM)][[`Project Page`](https://akshitac8.github.io/BiAM/)]

#### [Sanath Narayan](https://sites.google.com/view/sanath-narayan)<sup>\*</sup>, [Akshita Gupta](https://akshitac8.github.io/)<sup>\*</sup>, [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) ####

(:star2: denotes equal contribution)


## Installation
The codebase is built on PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.6, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
 
```
conda create -n mlzsl python=3.6
conda activate mlzsl
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image scikit-learn opencv-python yacs joblib natsort h5py tqdm pandas
```
Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

```

## Attention Visualization

<img src = "https://i.imgur.com/LJujDPx.png" width="900">

## Results
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/DzhhRH0.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/B6XWZmR.png" width="450"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Our approach on NUS-WIDE Dataset.</b></p></td>
    <td><p align="center"><b>Our approach on OpenImages Dataset.</b></p></td>
  </tr>
</table>


## Training and Evaluation

### NUS-WIDE

### Step 1: Data preparation

1) Download pre-computed features from [here](https://drive.google.com/drive/folders/1jvJ0FnO_bs3HJeYrEJu7IcuilgBipasA?usp=sharing) and store them at `features` folder inside `BiAM/datasets/NUS-WIDE` directory.
2) [Optional] You can extract the features on your own by using the original NUS-WIDE dataset from [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) and run the below script:

```
python feature_extraction/extract_nus_wide.py

```

### Step 2: Training from scratch

To train and evaluate multi-label zero-shot learning model on full NUS-WIDE dataset, please run:

```
sh scripts/train_nus.sh
```

### Step 3: Evaluation using pretrained weights

To evaluate the multi-label zero-shot model on NUS-WIDE. You can download the pretrained weights from [here](https://drive.google.com/drive/folders/1o03bqr_yNPblwAPjv2J83tMsHEDiEKPk?usp=sharing) and store them at `NUS-WIDE` folder inside `pretrained_weights` directory.

```
sh scripts/evaluate_nus.sh
```

### OPEN-IMAGES

### Step 1: Data preparation

1) Please download the annotations for [training](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv), [validation]( https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv), and [testing](https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv) into this folder.

2) Store the annotations inside `BiAM/datasets/OpenImages`.

3) To extract the features for OpenImages-v4 dataset run the below scripts for crawling the images and extracting features of them:

```
## Crawl the images from web
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `train`: download images into `./image_data/train/`
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `validation`: download images into `./image_data/validation/`
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `test`: download images into `./image_data/test/`

## Run feature extraction codes for all the 3 splits
python feature_extraction/extract_openimages_train.py
python feature_extraction/extract_openimages_test.py
python feature_extraction/extract_openimages_val.py

```

### Step 2: Training from scratch

To train and evaluate multi-label zero-shot learning model on full OpenImages-v4 dataset, please run:

```
sh scripts/train_openimages.sh
sh scripts/evaluate_openimages.sh

```

### Step 3: Evaluation using pretrained weights

To evaluate the multi-label zero-shot model on OpenImages. You can download the pretrained weights from [here](https://drive.google.com/drive/folders/1gW0rBofvVXiqfplQWGJLzao8v1bJ3Z8T?usp=sharing) and store them at `OPENIMAGES` folder inside `pretrained_weights` directory.

```
sh scripts/evaluate_openimages.sh
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :confetti_ball::

    @article{narayan2021discriminative,
    title={Discriminative Region-based Multi-Label Zero-Shot Learning},
    author={Narayan, Sanath and Gupta, Akshita and Khan, Salman and  Khan, Fahad Shahbaz and Shao, Ling and Shah, Mubarak},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    publisher = {IEEE},
    year={2021}
    }

## Contact
Should you have any question, please contact :e-mail: akshita.gupta@inceptioniai.org
