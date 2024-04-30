# YOLOv8-TO (WORK IN PROGRESS)
Code for the article "From Density to Geometry: YOLOv8 Instance Segmentation for Reverse Engineering of Optimized Structures"

## Table of Contents
- [Overview](#overview)
- [Reference](#reference)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [Datasets](#datasets)

## Overview
Brief description of what the project does and the problem it solves. Include a link or reference to the original article that inspired or is associated with this implementation.

##  Reference
This code aims to reproduce the results presented in the research article:

> Author(s). (Year). Title. *Journal*, Volume(Issue), Pages. DOI

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
