# Human-Protein-Atlas-Challenge

This is the code for my submissions to the Human Protein Atlas Challenge. Unfortunately I ran out of time to refine it towards the end, it is however fully functional.

# Prerequisites 
* Python3 
* numpy, pandas, deepdish
* pytorch
* albumentations

Please place the challenge data in the data folder

# Features

* Heavy Data Augmentation using Albumentations
* Utilizes PyTorch and a ResNet34 for the task
* Runs on less than 8 GB VRAM

### Adaptive data augmentation
As the training (and test) class distributions are heavily unbalanced, we perform an adaptive data augmentation such that we create more augmentations for underrepresented classes and vice versa for overrepresented ones.

# N.B.
This code loads all training data into RAM, so you'll need some RAM. Otherwise, it might be a better fit to use a streaming data augmentation, however, this is currently not compatible with the adaptive augmentation
