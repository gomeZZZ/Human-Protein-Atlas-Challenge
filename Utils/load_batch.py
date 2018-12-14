import cv2
from Utils.center_images import *
import torch

def load_batch(filenames, all_channels = True):
    """Reads the passed images and returns a pytorch tensor with them
    
    Args:
        filenames (np array): filepaths to the images in this batch
    Returns:
        X_batch (pytorch tensor): tensor with the images
    """    
    imgs = []
    for filename in filenames:
        if all_channels:
            blue = cv2.imread(filename+'_blue.png',0)
            green = cv2.imread(filename+'_green.png',0)
            red = cv2.imread(filename+'_red.png',0)
            yellow = cv2.imread(filename+'_yellow.png',0)
            img = np.asarray([green,red,blue,yellow]).transpose(1,2,0)
        else:
            img = cv2.imread(filename+'_green.png',0)[...,None]

        imgs.append(img)
    
    imgs = np.asarray(imgs).transpose(0,3,1,2).astype(np.float32) / 255.0
    imgs = center_images(imgs)

    return torch.tensor(imgs)
