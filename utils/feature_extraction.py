#import libraries 
import numpy as np
import pandas as pd 
import os
import math
import cv2 as cv
from PIL import Image
import skimage
from skimage import measure


def femhead_centre(mask):
    '''
    Given a femoral head binary mask, return the centroid coordinates of each femoral head.
    
    '''
    l_contour, r_contour = get_contours(mask)

    if l_contour is not None:
        l_x = np.mean(l_contour[:,1])
        l_y = np.mean(l_contour[:,0])
        lxy = [l_y,l_x]
    else:
        lxy = None
        
    if r_contour is not None:
        r_x = np.mean(r_contour[:,1])
        r_y = np.mean(r_contour[:,0])
        rxy = [r_y,r_x]
    else:
        rxy = None
              
    return rxy, lxy
               
    
def get_contours(mask):
    '''
    Extract the left and right contours from a mask.
    
    '''

    msk = Image.open(mask)
    msk = msk.convert('1')
    msk = np.asarray(msk)
    contours = skimage.measure.find_contours(msk,0)# binary image
    
    length1 = 0
    length2 = 0
    contr = None
    contl = None
    first = True
    
    if len(contours) > 1:
        for cont in contours:
            if cont.size > length1 and first:
                contl = cont
                contr = cont
                length1 = cont.size
                first = False
            elif cont.size > length1:
                contr = contl
                contl = cont
                length1 = cont.size
                length2 = contr.size
            elif cont.size > length2 and cont.size <= length1:
                contr = cont
                length2 = cont.size

        contl = contl.reshape((-1,2))
        contr = contr.reshape((-1,2))

        if np.mean(contl[:,1]) < np.mean(contr[:,1]):
            temp = contl.copy()
            contl = contr
            contr = temp
    
    elif len(contours) == 1:
        cont = contours[0]
        if np.mean(cont[:,1]) < msk.shape[1]/2:
            contr = cont
        else:
            contl = cont
        
    return contl, contr


def extract_image_size(img):
    '''
    Return the dimensions of an image.
    
    '''
    
    im = cv.imread(img)
    
    return [im.shape[1],im.shape[0]]