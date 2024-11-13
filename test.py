from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import shutil
from glob import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import models
from utils.earlystopping import EarlyStopping
from utils import datasets
from torchvision.datasets.utils import list_files
from utils.params import Params
from utils.plotting import plot_training
from utils.train_progress_tools import run_train_generator, track_running_average_loss, monitor_progress
import utils.eval_metrics as e_metric
from utils.data_prep import mean_and_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",type=str,help="Pass name of model as defined in hparams.yaml.")
    parser.add_argument("--k",required=False,type=int,default=10,help="Number of times to train and evaluate model")
    parser.add_argument("--cl",required=False,default=False,help="Set to true to use the UNet with Custom Loss as base.")
    parser.add_argument("--ckpt",required=False,type=str,default='Checkpoint/Test1',help="Set a checkpoint folder.")
    args = parser.parse_args()

    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
   
    # # Use GPU if available
    # use_gpu= torch.cuda.is_available()
    # device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.eval_metric)
    metrics = eval_metric()
    
    # Get root for dataset
    root = '//data/scratch/r094879/data'

    csv_file = os.path.join(root,'annotations/annotations.csv')
    csv_df = pd.read_csv(csv_file)

    train = []
    val = []
    test = []

    for index, row in csv_df.iterrows():
        image_name = row['image']

        if index < int(0.1*len(csv_df)):
            train.append(image_name)
        elif index < int(0.11*len(csv_df)):
            val.append(image_name)
        elif index >= int(0.90*len(csv_df)):
            test.append(image_name)
    
    Dataset = getattr(datasets,params.dataset_class)
    
    extra = ""
    extra2 = extra
                        
    if args.cl:
        extra2 = extra2 + "_CL"
        
    acc_scores = []
    pred_acc_scores = []
    
    # Make directories to save results 
    prediction_save = os.path.join(root,"Results",args.model_name,
                                   "Predicted" + extra + " " + params.target_dir)
    if not os.path.exists(prediction_save): os.makedirs(prediction_save)
    
        
    # # Calculate mean and std for dataset normalization 
    # norm_mean = [np.float32(0.99997693),np.float32(0.99997693),np.float32(0.99997693)]
    # norm_std = [np.float32(0.0009455526),np.float32(0.0009455526),np.float32(0.0009455526)]

    # # Define transform for images
    # transform=transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
    #                               transforms.ToTensor(),
    #                               transforms.Normalize(mean=norm_mean,std=norm_std)
    #                               ])

    # Set up transforms for targets
    target_transform = transforms.ToTensor()
      
    # Define test dataset
    test_data = Dataset(root,test,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                       input_tf=target_transform,output_tf=target_transform)
           
    # Define dataloaders
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)
    
    # Define model
    model = model_module.net(params.num_classes)
    
    # Grap test function for your network.
    test = model_module.test

    params.checkpoint_dir = str(args.ckpt)
    # Load relevant checkpoint for the fold
    chkpt = os.path.join(root,params.checkpoint_dir,"chkpt_{}".format(args.model_name+extra+"_lr0001"))
    
    acc = test(model,test_loader,metrics,params,checkpoint=chkpt,name=args.model_name,extra=extra, 
               prediction_dir=prediction_save)
    print("Test Accuracy: {}".format(acc))
    acc_scores.append(float(acc))
        
    print("Total: {}".format(np.mean(acc_scores)))


if __name__ == '__main__':

    main()