import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import list_files
import random


class HipSegDataset(Dataset):
    def __init__(self,data_dir,fold_num,image_dir,target_dir,target_sfx='.png',
                 input_tf=None,
                 output_tf=None,
                 loader=pil_loader):
        self.root = data_dir
        self.input_tf = input_tf
        self.output_tf = output_tf
        self.loader = loader
        self.target_sfx = target_sfx
        self.input_dir = os.path.join(self.root,"Fold "+str(fold_num),image_dir) #set image directory
        self.output_dir = os.path.join(self.root,"Fold "+str(fold_num),target_dir) 
        self.file_list = list_files(self.output_dir,target_sfx) 
        self.file_list.sort() #sorts images in descending order
            
    def __getitem__(self, index): #getitem method
        output_filename = self.file_list[index]
        filename = output_filename[:-4] 
        input_filename = filename + '.png'
        input_filename = os.path.join(self.input_dir, input_filename)
        output_filename = os.path.join(self.output_dir, output_filename)
        # Load target and image
        input = self.loader(input_filename) # converts to PIL image RGB
        if self.target_sfx == '.png':
            output = self.loader(output_filename)
        else:
            output = np.load(output_filename)
        # Apply transforms if given
        if self.input_tf is not None: 
            input = self.input_tf(input)
        if self.output_tf is not None:
            output = self.output_tf(output)
        if self.target_sfx == '.png':   
            output[output < 0.5] = 0.0
            output[output >= 0.5] = 1.0
        return input, output, output_filename
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    
class HipSegDatasetTESTFEM(Dataset):
    def __init__(self,data_dir,image_dir,target_dir,target_sfx='.png',
                 input_tf=None,
                 output_tf=None,
                 loader=pil_loader):
        self.root = data_dir
        self.input_tf = input_tf
        self.output_tf = output_tf
        self.loader = loader
        self.target_sfx = target_sfx
        self.input_dir = os.path.join(self.root,"FINAL TEST",image_dir) #set image directory
        self.output_dir = os.path.join(self.root,"FINAL TEST",target_dir) 
        self.file_list = list_files(self.output_dir,target_sfx) 
        self.file_list.sort() #sorts images in descending order
            
    def __getitem__(self, index): #getitem method
        output_filename = self.file_list[index]
        filename = output_filename[:-4] 
        input_filename = filename + '.png'
        input_filename = os.path.join(self.input_dir, input_filename)
        output_filename = os.path.join(self.output_dir, output_filename)
        # Load target and image
        input = self.loader(input_filename) # converts to PIL image RGB
        if self.target_sfx == '.png':
            output = self.loader(output_filename)
        else:
            output = np.load(output_filename)
        # Apply transforms if given
        if self.input_tf is not None: 
            input = self.input_tf(input)
        if self.output_tf is not None:
            output = self.output_tf(output)
        if self.target_sfx == '.png':   
            output[output < 0.5] = 0.0
            output[output >= 0.5] = 1.0
        return input, output, output_filename
    
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
    
    
class HipSegDatasetAll(Dataset):
    def __init__(self,data_dir,image_dir,target_dir,target_sfx='.png',
                 input_tf=None,
                 output_tf=None,
                 loader=pil_loader):
        self.root = data_dir
        self.input_tf = input_tf
        self.output_tf = output_tf
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.loader = loader
        self.target_sfx = target_sfx
        self.file_list = []
        subsets = ["Train","Val","Test"]
        for subset in subsets:
            subdir_list = []
            self.subdir = os.path.join(self.root, subset) #set sub directory
            self.input_dir = os.path.join(self.subdir,self.image_dir) #set image directory
            self.output_dir = os.path.join(self.subdir,self.target_dir) 
            subdir_list = list_files(self.output_dir,target_sfx,prefix=False) 
            for index, file in enumerate(subdir_list):
                subdir_list[index] = os.path.join(self.output_dir, file)
            self.file_list.extend(subdir_list)
        random.shuffle(self.file_list)
            
    def __getitem__(self, index): #getitem method
        output_filename = self.file_list[index]
        filename = output_filename.split('\\')[-1]
        subdir = os.path.join(*output_filename.split('\\')[:-2])
        input_filename = os.path.join(subdir,self.image_dir,filename[:-4]+'.png')
        # Load target and image
        input = self.loader(input_filename) # converts to PIL image RGB
        if self.target_sfx == '.png':
            output = self.loader(output_filename)
        else:
            output = np.load(output_filename)
        # Apply transforms if given
        if self.input_tf is not None: 
            input = self.input_tf(input)
        if self.output_tf is not None:
            output = self.output_tf(output)
        if self.target_sfx == '.png':   
            output[output < 0.5] = 0.0
            output[output >= 0.5] = 1.0
        return input, output
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    