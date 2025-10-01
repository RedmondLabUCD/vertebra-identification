#import libraries 
import numpy as np
import pandas as pd 
import os
import itertools
import math


def resize_lm(landmarks,old_dim,new_dim):
    '''
    Rescale landmarks to suit a change in the corresponding image size.
    '''
    
    landmarks[:,0] = landmarks[:,0] * new_dim/old_dim[0]
    landmarks[:,1] = landmarks[:,1] * new_dim/old_dim[1]
    
    return landmarks 