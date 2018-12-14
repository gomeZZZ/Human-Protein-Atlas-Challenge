import numpy as np
from Utils.utils import *

def center_images(X, verbose = False):
    """Converts images to 0 mean, unit variance. Expects NCHW
    
    Args:
        X (np array): images
        verbose (bool, optional): Defaults to False. print progress
    
    Returns:
        np array: z scored images
    """

    for idx,image in enumerate(X):
        if verbose and idx % 25 == 0:
            printProgressBar (idx, X.shape[0], prefix = 'Centering images...', suffix = '(' + str(idx) + '/' + str(X.shape[0]) + ')')
        for j,channel in enumerate(image):
            image[j] = channel - np.mean(channel)
            image[j] = channel / np.std(channel)
        
        X[idx] = image

    return X

