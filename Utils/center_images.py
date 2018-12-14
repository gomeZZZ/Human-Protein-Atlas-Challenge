import numpy as np
from Utils.utils import *

def center_images(X, verbose = False):
    for idx,image in enumerate(X):
        if verbose and idx % 25 == 0:
            printProgressBar (idx, X.shape[0], prefix = 'Centering images...', suffix = '(' + str(idx) + '/' + str(X.shape[0]) + ')')
        for j,channel in enumerate(image):
            image[j] = channel - np.mean(channel)
            image[j] = channel / np.std(channel)
        
        X[idx] = image

    return X

