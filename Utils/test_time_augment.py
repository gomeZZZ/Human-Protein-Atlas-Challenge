import skimage.transform as transform
import numpy as np 

def test_time_augment(img, rotate = True, flipLR = True, flipUD = True):
    """Creates all permutations of rotations,transpositions and flips for the passed image and returns them

    Args:
        img (np array): image as uint8
        rotate (bool, optional): Defaults to True. [rotate image?]
        flipLR (bool, optional): Defaults to True. [flip image LR?]
        flipUD (bool, optional): Defaults to True. [flip image UD?]
    Returns:
        batch (np array): augmented images

    """

    batch = []
    batch.append(img)

    if rotate:
        #create rotations
        rot90 = transform.rotate(img,90)
        rot180 = transform.rotate(img,180)
        rot270 = transform.rotate(img,270)

        #add rotations
        batch.append(rot90)
        batch.append(rot180)
        batch.append(rot270)

    if flipLR:
        #create LR flips
        imgLR= np.fliplr(img) 
        if rotate:
            rot90LR = np.fliplr(rot90)
            rot180LR = np.fliplr(rot180)
            rot270LR = np.fliplr(rot270)

        batch.append(imgLR)
        if rotate:
            batch.append(rot90LR)
            batch.append(rot180LR)
            batch.append(rot270LR)
    
    if flipUD:
        #create UD flips
        imgUD = np.flipud(img) 
        if flipLR:
            imgUDLR = np.flipud(imgLR)

        if rotate:    
            rot90UD = np.flipud(rot90)
            rot180UD = np.flipud(rot180)
            rot270UD = np.flipud(rot270)
    
        if flipLR and rotate:
            rot90UDLR = np.flipud(rot90LR)
            rot180UDLR = np.flipud(rot180LR)
            rot270UDLR = np.flipud(rot270LR)

        batch.append(imgUD)
        if flipLR:
            batch.append(imgUDLR)
        
        if rotate:  
            batch.append(rot90UD)
            batch.append(rot180UD)
            batch.append(rot270UD)
        
        if flipLR and rotate:
            batch.append(rot90UDLR)
            batch.append(rot180UDLR)
            batch.append(rot270UDLR)
    
    return np.asarray(batch)