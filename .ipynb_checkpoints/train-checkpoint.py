from __future__ import print_function
import os
import time
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import models
import shutil
from glob import glob

from utils.earlystopping import EarlyStopping
from utils import datasets
from torchvision.datasets.utils import list_files
from utils.params import Params
from utils.plotting import plot_training
from utils.train_progress_tools import run_train_generator, track_running_average_loss, monitor_progress
import utils.eval_metrics as e_metric


def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",type=str,help="Pass name of model as defined in hparams.yaml.")
    parser.add_argument("--data_aug",required=False,default=False,help="Set to true to augment data.")
    parser.add_argument("--clahe",required=False,default=False,help="Set to true to use CLAHE images.")
    parser.add_argument("--AUG",required=False,default=False,help="Set to true to use AUG images.")
    parser.add_argument("--lr",required=False,type=float,default=0.0001,help="Specify new learning rate")
    args = parser.parse_args()
    
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Set up "extra" bit to add to checkpoint name (i.e. specifying if it was pretrained, data augmented, etc.)
    extra = ""
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    model = model_module.net(params.num_classes)
    
    # Send the model to the chosen device. 
    model.to(device)
    
    # Grap your training and validation functions for your network.
    train = model_module.train
    val = model_module.val
    
    # Specify learning rate
    split = str(args.lr).split('.')[-1]
    extra = extra + '_lr' + split
    
    # Set up optimizer (Adam) with chosen learning rate
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    
    # Define loss function 
    loss_func = getattr(nn, params.loss)
    criterion = loss_func()
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.eval_metric)
    metrics = eval_metric()
    
    if args.clahe:
        params.image_dir = params.image_dir + " CLAHE"
        extra = extra + "_clahe"
        
    if args.AUG:
        extra = extra + "_AUG"
    
    transform_val=transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.1926, 0.1926, 0.1926], 
                                                           std=[0.2524, 0.2524, 0.2524])
                                      ])
    
    # Augment the data depending on command line argument 
    if args.data_aug:
        extra = extra + "_aug"
        transform=transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.1926, 0.1926, 0.1926], 
                                                           std=[0.2524, 0.2524, 0.2524]),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2)
                                      ])
    else:
        transform=transform_val 
    
                                      
    # Set up transforms for targets
    if "Masks" in params.target_dir:
        target_transform = transforms.Compose([transforms.Grayscale(),
                                               transforms.Resize((params.input_size,params.input_size)),
                                               transforms.ToTensor()
                                               ])
    else:
        target_transform = transforms.ToTensor()

    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root,"Dataset")
        
    # Define train and val datasets 
    Dataset = getattr(datasets, params.dataset_class)
    if args.AUG:
        train_data = Dataset(data_dir,"Train",params.image_dir+" AUG",params.target_dir+" AUG",
                         target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
    else:
        train_data = Dataset(data_dir,"Train",params.image_dir,params.target_dir,
                         target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)

    train_data_eval = Dataset(data_dir,"Train",params.image_dir,params.target_dir,
                         target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
    val_data = Dataset(data_dir,"Val",params.image_dir,params.target_dir,
                         target_sfx=params.target_sfx,input_tf=transform_val,output_tf=target_transform)
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=params.batch_size, shuffle=False, pin_memory=True)
    train_loader_eval = DataLoader(train_data_eval, batch_size=1, shuffle=False, pin_memory=True)
    val_loader_eval = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    train_list_target = list_files(os.path.join(data_dir,"Train",params.target_dir),
                                                   params.target_sfx,prefix=False)
    train_list_target.sort() 
    val_list_target = list_files(os.path.join(data_dir,"Val",params.target_dir),
                                                   params.target_sfx,prefix=False)
    val_list_target.sort() 
    
    # Make directories to save results 
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    if not os.path.exists("figs"): os.makedirs("figs")

    val_accs = []
    val_losses = []
    train_losses = []
    train_accs = []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(verbose=True, patience=20, up=params.early_stopping_up, path=params.checkpoint_dir)
    epoch_max = 201
    for epoch in range(1, epoch_max):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train_generator = train(model, device, train_loader, optimizer, criterion)
        train_generator = track_running_average_loss(train_generator)
        train_generator = monitor_progress(train_generator)
        train_loss_decay = run_train_generator(train_generator)

        # Evaluate on both the training and validation set.
        train_loss , train_acc = val(model, device, train_loader_eval, 
                                     train_list_target, criterion, metrics, params, subdir="Train")
        val_loss, val_acc = val(model, device, val_loader_eval, 
                                val_list_target, criterion, metrics, params)
        # Collect some data for logging purposes. 
        train_losses.append(float(train_loss))
        train_accs.append(train_acc)
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)
        
        early_stopping(val_acc, model, optimizer, args.model_name, extra, epoch)

        print('\n\ttrain Loss: {:.6f}\ttrain acc: {:.6f} \n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(train_loss, train_acc, val_loss, val_acc))       
        
        if early_stopping.early_stop:
            print("Early stopping")
            fig = plot_training(train_losses,train_accs,val_losses,val_accs,model_name=args.model_name+extra)
            fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name+extra)))
            break
            
        if epoch == epoch_max-1:
            print("Max epochs reached.")
            fig = plot_training(train_losses,train_accs,val_losses,val_accs,model_name=args.model_name+extra)
            fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name+extra)))
            break

    if params.early_stopping_up:
        best_val_epoch = int(np.argmax(val_accs)+1)
    else:
        best_val_epoch = int(np.argmin(val_accs)+1)
        
    # Some log information to help you keep track of your model information. 
    logs = {
        "model": args.model_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_epoch": best_val_epoch,
        "lr": args.lr,
        "batch_size": params.batch_size,
        "data_aug": args.data_aug,
        "AUG": args.AUG,
        "clahe": args.clahe
    }

    with open(os.path.join(params.log_dir,"{}{}_{}.json".format(args.model_name, extra, start_time)), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    main()


