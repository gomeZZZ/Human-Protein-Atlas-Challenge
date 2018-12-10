import numpy as np
from Utils.utils import *

def center_images(X, verbose = False):
    for idx,image in enumerate(X):
        if verbose and idx % 25 == 0:
            printProgressBar (idx, X.shape[0], prefix = 'Centering images...', suffix = '(' + str(idx) + '/' + str(X.shape[0]) + ')')
        image = image - np.mean(image)
        image = image / np.std(image)
        
        X[idx] = image

    return X

