import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import list_files
import random
import pydicom
import cv2 as cv
from pydicom import dcmread


class SpineDataset(Dataset):
    def __init__(self,root,filenames,image_dir,target_dir,target_sfx='.png',
                 input_tf=None,
                 output_tf=None,
                 loader=pil_loader):
        self.root = root
        self.input_tf = input_tf
        self.output_tf = output_tf
        self.loader = loader
        self.target_sfx = target_sfx
        self.input_dir = os.path.join(self.root,image_dir) #set image directory
        self.output_dir = os.path.join(self.root,target_dir) 
        self.file_list = filenames
            
    def __getitem__(self, index): #getitem method
        filename = self.file_list[index]
        input_filename = os.path.join(self.input_dir, filename+'.dcm')
        output_filename = os.path.join(self.output_dir, filename+'.npy')
        
        dicom_image = dcmread(input_filename)
        input = dicom_image.pixel_array
        input = input.astype(float)
        input = (input-input.min())/(input.max()-input.min())*255.0
        input = input.astype(np.uint8)
        input = Image.fromarray(input)
        input = input.resize((256,256))
        
        # input = self.loader(input_filename)
        output = np.load(output_filename) 
        if self.input_tf is not None: 
            input = self.output_tf(input)
        if self.output_tf is not None:
            output = self.output_tf(output)
        return input, output, filename
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    
    
class HipSegDatasetTEST(Dataset):
    def __init__(self,data_dir,image_dir,
                 input_tf=None,
                 loader=pil_loader):
        self.root = data_dir
        self.input_tf = input_tf
        self.loader = loader
        self.input_dir = os.path.join(self.root,image_dir) #set image directory
        self.file_list = list_files(self.input_dir,".png",prefix=False) 
        self.file_list.sort() #sorts images in descending order
            
    def __getitem__(self, index): #getitem method
        filename = self.file_list[index]
        input_filename = os.path.join(self.input_dir,filename)
        # Load target and image
        input = self.loader(input_filename) # converts to PIL image RGB
        # Apply transforms if given
        if self.input_tf is not None: 
            input = self.input_tf(input)
        return input, filename
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    
    
