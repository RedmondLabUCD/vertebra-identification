import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2
from numpy import fliplr
import math
from tqdm import tqdm
from PIL import Image,ImageEnhance
from scipy import ndimage
from skimage import io
from utils import datasets
from utils.landmark_prep import prep_landmarks


def final_mean_and_std(data_dir, params):
    '''
    Calculates mean and standard deviation of images to be 
    used in image normalization.
    Inspired by: towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    '''
    Dataset = getattr(datasets,"SpineDataset")
    
    # Define basic transform (resize and make tensor)
    transform = transforms.ToTensor()

    csv_file = os.path.join(data_dir,'annotations/annotations.csv')
    csv_df = pd.read_csv(csv_file)

    train = []
    val = []
    test = []

    train_id = 0
    val_id = 0

    for index, row in csv_df.iterrows():
        image_name = row['image']

        if index < int(0.8*len(csv_df)):
            train.append(image_name)
            train_id = row['id']
        elif index < int(0.9*len(csv_df)):
            if int(row['id']) == int(train_id):
                train.append(image_name)
            else:
                val.append(image_name)
                val_id = row['id']
        elif index >= int(0.9*len(csv_df)):
            if int(row['id']) == int(val_id):
                val.append(image_name)
            else:
                test.append(image_name)

    # Define and load training dataset
    train_data = Dataset(data_dir,train,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                input_tf=transform,output_tf=transform)

    loader = DataLoader(train_data,batch_size=params.batch_size,shuffle=False)
    
    # Calculate mean and std for each batch 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    # Get mean and std across the batches
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    mean2 = 1.*mean[0]
    
    std = std.cpu().numpy()
    std2 = 1.*std[0]

    return mean2, std2


def apply_clahe(data_dir):
    '''
    Applies CLAHE to data to increase contrast.
    '''

    images = glob(os.path.join(data_dir,"*"))
    if not os.path.exists(data_dir+" CLAHE"): os.makedirs(data_dir+" CLAHE")
    
    for image in images:
        image_name = image.split("\\")[-1].split(".")[0]
        
        img = cv2.imread(image, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_img = clahe.apply(img)
        
        cv2.imwrite(os.path.join(data_dir+' CLAHE',image_name+'.png'),cl_img)
        