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
    parser.add_argument("--AUG",required=False,default=False,help="Set to true to use AUG images.")
    parser.add_argument("--AUG2",required=False,default=False,help="Set to true to use AUG images.")
    parser.add_argument("--roi",required=False,type=str,default=None,help="Uses ROI predictions as base.")
    parser.add_argument("--k",required=False,type=int,default=10,help="Number of times to train and evaluate model")
    parser.add_argument("--baseAUG",required=False,default=False,help="Use Augmented ROI predictions as base.")
    parser.add_argument("--alt_loss",required=False,default=False,help="Use Augmented ROI predictions as base.")
    parser.add_argument("--clahe",required=False,default=False,help="Set to true to use CLAHE images.")
    parser.add_argument("--attn",required=False,default=False,help="Set to true to use Attn UNet images as base.")
    parser.add_argument("--cl",required=False,default=False,help="Set to true to use the UNet with Custom Loss as base.")
    args = parser.parse_args()

    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
   
    # Use GPU if available
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.eval_metric)
    metrics = eval_metric()
    
    # Get root for dataset
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root,"Dataset")
        
    Dataset = getattr(datasets,params.dataset_class)
    
    extra = ""
    extra2 = extra
           
    if args.clahe:
        params.image_dir = params.image_dir + " CLAHE"
        extra = extra + "_clahe"
        
    if args.baseAUG:
        extra2 = extra + "_baseAUG"
        
    if args.alt_loss:
        extra2 = extra + "_MSE"
        
    if args.attn:
        extra2 = extra + "_Attn"
                        
    if args.cl:
        extra2 = extra2 + "_CL"
        
    if args.AUG:
        extra = extra + '_AUG'
        extra2 = extra + '_AUG'
        
    if args.AUG2:
        extra = extra + '_AUG2'
        extra2 = extra2 + '_AUG2'
        
    acc_scores = []
    pred_acc_scores = []
    
    # Make directories to save results 
    prediction_save = os.path.join(root,"Results",args.model_name,
                                   "Predicted" + extra + " " + params.target_dir)
    if not os.path.exists(prediction_save): os.makedirs(prediction_save)
    
    if args.roi:
        prediction_roi = os.path.join(root,"Results",args.model_name,
                                       "Predicted" + extra2 + " Pred " + params.target_dir)
        if not os.path.exists(prediction_roi): os.makedirs(prediction_roi)
        params_roi = Params("hparams.yaml", args.model_name)
        insize = ""
        if args.attn:
            insize = insize + "_Attn"
        if args.cl:
            insize = insize + "_CL"
        if args.baseAUG:
            insize = insize + "_AUG"
            
        params_roi.image_dir = "Predicted" + insize + " " + params.image_dir 
        params_roi.target_dir = "Predicted" + insize + " " + params.target_dir
    
    if args.AUG2:
        params.target_dir = params.target_dir + " AUG2"
        params.image_dir = params.image_dir + " AUG2"
    
    # ==================== EVALUATE MODEL FOR EACH FOLD ====================
    
    for index in range(1,args.k+1):
        
        # Calculate mean and std for dataset normalization 
        norm_mean,norm_std = mean_and_std(index,data_dir,params,args.AUG)

        # Define transform for images
        transform=transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=norm_mean,std=norm_std)
                                      ])

        # Set up transforms for targets
        if "Masks" in params.target_dir:
            target_transform = transforms.Compose([transforms.Grayscale(),
                                                   transforms.Resize((params.input_size,params.input_size)),
                                                   transforms.ToTensor()
                                                   ])
        else:
            target_transform = transforms.ToTensor()
          
       
        # Define test dataset
        test_data = Dataset(data_dir,index,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                           input_tf=transform,output_tf=target_transform)
        
        if args.roi:
            pred_test_data = Dataset(data_dir,index,params_roi.image_dir,params_roi.target_dir,
                                     target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
            pred_test_loader = DataLoader(pred_test_data, batch_size=1, shuffle=False, pin_memory=False)
               
        # Define dataloaders
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)
        
        # Define model
        model = model_module.net(params.num_classes).to(device)
        
        # Grap test function for your network.
        test = model_module.test
    
        # Load relevant checkpoint for the fold
        chkpt = os.path.join(params.checkpoint_dir,"chkpt_{}_fold_{}".format(args.model_name+extra+"_lr0001",index))
        
        acc = test(model,device,test_loader,metrics,params,checkpoint=chkpt,name=args.model_name,extra=extra, 
                   prediction_dir=prediction_save)
        print("Test Accuracy: {}".format(acc))
        acc_scores.append(float(acc))
        
        if args.roi:
            pred_acc = test(model, device, pred_test_loader, metrics, params_roi, checkpoint=chkpt, 
                            name=args.model_name,extra="_Pred"+extra2, prediction_dir=prediction_roi)
            pred_acc_scores.append(float(pred_acc))
            print("Test Accuracy: {}".format(pred_acc))
        
        # ==================== FREE UP CUDA GPU MEMORY ====================

        gc.collect()
        del model
        del test_data
        del test_loader
        
        torch.cuda.empty_cache()
        
    print("Total: {}".format(np.mean(acc_scores)))
    if args.roi:
        print("Total: {}".format(np.mean(pred_acc_scores)))


if __name__ == '__main__':

    main()