# Repruducing EMA in Tensorflow

## Introduction
This is a tensorflow implementation of the 'Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation'

## Running the code

### train 
'python3 -m vfi.src.train'

### evaluation
'python3 -m vfi.src.eval'

### Training data
EMA is trained on a proprietary dataset with one million internet video clips, each comprising 3 frames.

- We use the publcicly available Vimeo-90k dataset ([Xue et al., 2019](https://arxiv.org/abs/1711.09078)), which is commonly used dataset for video frame interpolation


