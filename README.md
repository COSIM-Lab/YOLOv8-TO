---
title: YOLOv8-TO Demo
emoji: 🏗️
colorFrom: yellow
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---


# YOLOv8-TO
Code for the article "From Density to Geometry: YOLOv8 Instance Segmentation for Reverse Engineering of Optimized Structures"

## Table of Contents
- [Overview](#overview)
- [Reference](#reference)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [Datasets](#datasets)
- [Training](#training)
- [Inference](#inference)

## Overview
Brief description of what the project does and the problem it solves. Include a link or reference to the original article that inspired or is associated with this implementation.

## Demo
Try it at:

##  Reference
This code aims to reproduce the results presented in the research article:

```bibtex
@misc{rochefortbeaudoin2024density,
      title={From Density to Geometry: YOLOv8 Instance Segmentation for Reverse Engineering of Optimized Structures}, 
      author={Thomas Rochefort-Beaudoin and Aurelian Vadean and Sofiane Achiche and Niels Aage},
      year={2024},
      eprint={2404.18763},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Installation

### Prerequisites
This package comes with a fork of the ultralytics package in the yolov8-to directory. The fork is necessary to add the functionality of the design variables regression.

### Installing

```bash
git clone https://github.com/COSIM-Lab/YOLOv8-TO.git
cd YOLOv8-TO
pip install -e .
```
## Datasets
Links to the dataset on HuggingFace:
- [YOLOv8-TO_Data](https://huggingface.co/datasets/tomrb/yolov8to_data)

The Huggingface dataset contains the following datasets (see paper for details):
- MMC
- MMC-random
- SIMP
- SIMP_5%
- OOD


If you want to use one of the linked datasets, please unzip it inside of the datasets folder. Training labels are provided for the MMC and MMC-random data. To train on the data, please update the data.yaml file with the correct path to the dataset.
```yaml
path:  # dataset root dir
```


## Training

To train the model, make sure the train dataset is setup according to the above section and according to the documentation from ultralytics:
https://docs.ultralytics.com/datasets/

Refer to the notebook `YOLOv8_TO.ipynb` for an example of how to train the model.

## Inference
Refer to the notebook `YOLOv8_TO.ipynb` for an example of how to perform inference with the trained model.