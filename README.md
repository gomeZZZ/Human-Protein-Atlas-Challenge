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
* Train with external data
* Utilizes PyTorch and a ResNet34 for the task
* Runs on less than 8 GB VRAM

### Adaptive data augmentation
As the training (and test) class distributions are heavily unbalanced, we perform an adaptive data augmentation such that we create more augmentations for underrepresented classes and vice versa for overrepresented ones.

# Structure
* Utils - contains several utilities functions for preprocessing, plotting, etc.
* data - put the training/test data here
* notebooks - contains several notebooks
  * *Baseline* - provides a weak baseline result
  * *createAugmentedTrainingData* - creates adaptively augmented training data (as hdf5 files)
  * *createHDF5TrainingData* - converts the provided images to hdf5 (faster loading)
  * *loadExternalData* - loads external data from the web
  * *resnet34_augmented* - trains and validates a resnet34 on the provided (augmented) hdf5 data
  * *runTestSet* - used to perform inference on the test set and create a submission
  

# N.B.
This code loads all training data into RAM, so you'll need some RAM. Otherwise, it might be a better fit to use a streaming data augmentation, however, this is currently not compatible with the adaptive augmentation
