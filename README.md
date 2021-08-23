# Discriminative Region-based Multi-Label Zero-Shot Learning (ICCV 2021)

#### [Sanath Narayan](https://sites.google.com/view/sanath-narayan)<sup>\*</sup>, [Akshita Gupta](https://akshitac8.github.io/)<sup>\*</sup>, [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) ####

(* denotes equal contribution)



**Paper**: 

<hr />

> **Abstract:** *Multi-label zero-shot learning (ZSL) is a more realistic counter-part of standard single-label ZSL since several objects can co-exist in a natural image. 
However, the occurrence of multiple objects complicates the reasoning and requires region-specific processing of visual features to preserve their contextual cues.
We note that the best existing multi-label ZSL method takes a shared approach towards attending to region features with a common set of attention maps for all the classes.
Such shared maps lead to diffused attention, which does not discriminatively focus on relevant locations when the number of classes are large. Moreover, mapping spatially-pooled visual features to the class semantics leads to inter-class feature entanglement, thus hampering the classification. Here, we propose an alternate approach towards region-based discriminability-preserving multi-label zero-shot classification. Our approach maintains the spatial resolution to preserve region-level characteristics and utilizes a bi-level attention module (BiAM) to enrich the features by incorporating both region and scene context information. The enriched region-level features are then mapped to the class semantics and only their class predictions are spatially pooled to obtain image-level predictions, thereby keeping the multi-class features disentangled. Our approach sets a new state of the art on two large-scale multi-label zero-shot benchmarks: NUS-WIDE and Open Images. On NUS-WIDE, our approach achieves an absolute gain of 6.9\% mAP for ZSL, compared to the best published results.*

## Network Architecture
<table>
  <tr>
    <td> <img src = "" width="500"> </td>
    <td> <img src = "" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of BiAM</b></p></td>
    <td><p align="center"> <b>Region-based classification framework</b></p></td>
  </tr>
</table>

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
## Quick Run

## Training and Evaluation

Training and Testing codes for 

## Results
Experiments are performed for .

### OPEN-IMAGES

<img src = " width="900">

### NUS-WIDE

<p align="center"> <img src = "" width="450"> </p>

## Citation
If you use BiAM, please consider citing:

    @article{narayan2021discriminative,
    title={Discriminative Region-based Multi-Label Zero-Shot Learning},
    author={Narayan, Sanath and Gupta, Akshita and Khan, Salman and  Khan, Fahad Shahbaz and Shao, Ling and Shah, Mubarak},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    publisher = {IEEE},
    year={2021}
    }

## Contact
Should you have any question, please contact akshita.gupta@inceptioniai.org
