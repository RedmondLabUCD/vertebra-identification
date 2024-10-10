#import libraries 
import numpy as np
import pandas as pd 
import os
import math
import itertools
import skimage
from glob import glob
from sklearn.utils import shuffle
from torchvision.datasets.utils import list_files

from utils.process_predictions import pixel_to_mm
from utils.landmark_prep import prep_landmarks

def landmark_method_compare(compare_dir,save_name):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    tar_dir = os.path.join(root,"Dataset","CSVs")
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(tar_dir,"*.csv"))]
    
    df = pd.DataFrame(columns = ['Image','LM1','LM2','LM3','LM4','LM5','LM6','LM7','LM8','LM9','LM10','LM11',
                                 'LM12','LM13','LM14','LM15', 'LM16','LM17','LM18','LM19','LM20','LM21','LM22'])
    df['Image'] = filenames
    
    for filename in filenames:
        targets, __ = prep_landmarks(filename,tar_dir)
        targets = np.asarray(targets).astype(float).reshape((-1,2))
        
        preds = pd.read_csv(os.path.join(compare_dir,filename+".csv"))
        preds = np.asarray(preds).astype(float)[:22,:]
        
        
        for i in range(22):
            dist =  math.sqrt((targets[i,1]-preds[i,1])**2 + (targets[i,0]-preds[i,0])**2)
            mm = pixel_to_mm(filename,dist)
            df.loc[df["Image"]==filename,"LM"+str(i+1)] = mm

        df.to_csv(os.path.join(root,"Results","Statistics",save_name+'.csv'),index=False)