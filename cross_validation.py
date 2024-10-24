from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import shutil
from glob import glob
import gc
from numba import cuda

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
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
    parser.add_argument("--lr",required=False,type=float,default=0.0001,help="Specify new learning rate")
    parser.add_argument("--k",required=False,type=int,default=10,help="Number of times to train and evaluate model")
    parser.add_argument("--roi",required=False,type=str,default=None,help="Uses ROI predictions as base.")
    parser.add_argument("--dice",required=False,type=str,default=None,help="Saves dice scores in csv file if true.")
    parser.add_argument("--custom_loss",required=False,default=False,help="Use custom loss function.")
    args = parser.parse_args()
    
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Set up "extra" bit to add to checkpoint name (i.e. specifying if it was pretrained, data augmented, etc.)
    extra = ""
    if args.AUG:
        extra = extra + '_AUG'
    if args.AUG2:
        extra = extra + '_AUG2'
        params.target_dir = params.target_dir + " AUG2"
        params.image_dir = params.image_dir + " AUG2"
        
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    
    # Specify learning rate
    split = str(args.lr).split('.')[-1]
    extra = extra + '_lr' + split
    
    # Define loss function 
    if args.custom_loss:
        loss_func = getattr(e_metric, params.loss)
    else:
        loss_func = getattr(nn, params.loss)
    criterion = loss_func()
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.eval_metric)
    metrics = eval_metric()

    # Get root for dataset
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root,"Dataset")
        
    Dataset = getattr(datasets,params.dataset_class)
    
    # Create checkpoint directory if not already existing
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    
    # Make directories to save results 
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    if not os.path.exists("figs"): os.makedirs("figs")
    
    # Empty to hold the results of each fold
    acc_scores = []
    pred_acc_scores = []
    best_epochs = []
    
    # ==================== START K-FOLD CROSS VALIDATION ====================
    
    for index in range(1,args.k+1):
#     for index in range(1,2):    
        print("Training to commence for fold " + str(index) + ":")
        
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
            pred_test_data = Dataset(data_dir,index,"Predicted "+params.image_dir,"Predicted "+params.target_dir,
                                     target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
            pred_test_loader = DataLoader(pred_test_data, batch_size=1, shuffle=False, pin_memory=False)
        
        # Define validation dataset
        if index+1 > args.k:
            val_index = 1
        else:
            val_index = index + 1
        val_data = Dataset(data_dir,val_index,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                           input_tf=transform,output_tf=target_transform)
        
        # Define training dataset
        train_data = []
        for i in range(1,args.k+1):
            if i != index and i != val_index:
                if args.AUG:
                    fold_data = Dataset(data_dir,i,params.image_dir+" AUG",params.target_dir+" AUG",
                                        target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
                else:
                    fold_data = Dataset(data_dir,i,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                        input_tf=transform,output_tf=target_transform)
                train_data = ConcatDataset([train_data, fold_data])
               
        # Define dataloaders
        train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, pin_memory=False)
#         val_loader = DataLoader(val_data, batch_size=params.batch_size, shuffle=False, pin_memory=False)
#         train_loader_eval = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=False)
        val_loader_eval = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)
        
        # Define model and optimizer
        model = model_module.net(params.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay= 0.0001)
        
        # Grap your training, val, and test functions for your network.
        train = model_module.train
        val = model_module.val
        test = model_module.test
        
        val_accs = []
        val_losses = []
        train_losses = []
        train_accs = []
    
        # Initialize early stopping
        early_stopping = EarlyStopping(index, verbose=True, patience=5, up=params.early_stopping_up, path=params.checkpoint_dir)
        
        # ==================== TRAIN THE MODEL FOR ONE FOLD ====================
        
        epoch_max = 100
        for epoch in range(1,epoch_max):
            print("Epoch: {}".format(epoch))
            # Call training function. 
            train_generator = train(model, device, train_loader, optimizer, criterion, params)
            train_generator = track_running_average_loss(train_generator)
            train_generator = monitor_progress(train_generator)
            train_loss_decay = run_train_generator(train_generator)

            # Evaluate on both the training and validation set.
#             train_loss, train_acc = val(model, device, train_loader_eval, criterion, metrics, params, AUG=args.AUG)
            val_loss, val_acc = val(model, device, val_loader_eval, criterion, metrics, params)
            
            # Collect some data for logging purposes. 
#             train_losses.append(float(train_loss))
#             train_accs.append(train_acc)
            val_losses.append(float(val_loss))
            val_accs.append(val_acc)

            early_stopping(val_acc, model, optimizer, args.model_name, extra, epoch)

#             print('\n\ttrain Loss: {:.6f}\ttrain acc: {:.6f} \n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(train_loss, train_acc, val_loss, val_acc))      
            print('\n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(val_loss, val_acc))   

            if early_stopping.early_stop:
                print("Early stopping")
#                 fig = plot_training(train_losses,train_accs,val_losses,val_accs,model_name=args.model_name+extra)
#                 fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name+extra+"_"+str(index))))
                break

            if epoch == epoch_max-1:
                print("Max epochs reached.")
#                 fig = plot_training(train_losses,train_accs,val_losses,val_accs,model_name=args.model_name+extra)
#                 fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name+extra+"_"+str(index))))
                break

        # Define "best" epoch
        if params.early_stopping_up:
            best_val_epoch = int(np.argmax(val_accs)+1)
        else:
            best_val_epoch = int(np.argmin(val_accs)+1)
        best_epochs.append(best_val_epoch)

        # ==================== SAVE RELEVANT INFORMATION FOR THE MODEL ====================
        
        # Some log information to help you keep track of your model information. 
        logs = {
            "model": args.model_name,
#             "train_losses": train_losses,
#             "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_val_epoch": best_val_epoch,
            "lr": args.lr,
            "batch_size": params.batch_size,
            "AUG": args.AUG,
            "AUG2": args.AUG2,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "fold": index
        }

        with open(os.path.join(params.log_dir,"{}{}_fold_{}.json".format(args.model_name, extra, index)), 'w') as f:
            json.dump(logs, f)
        
        chkpt = os.path.join(params.checkpoint_dir,"chkpt_{}_fold_{}".format(args.model_name+extra,index))
        
        print("Training finished. Evaluation begins...")
        
        # ==================== TEST THE MODEL ====================
        
        acc = test(model,device,test_loader,metrics,params,checkpoint=chkpt,name=args.model_name,extra=extra)
        print("Test Accuracy: {}".format(acc))
        acc_scores.append(float(acc))
        
        if args.roi:
            pred_acc = test(model,device,pred_test_loader,metrics,params,checkpoint=chkpt,name=args.model_name,extra=extra)
            pred_acc_scores.append(float(pred_acc))
            print("Test Accuracy: {}".format(pred_acc))
        
        # ==================== FREE UP CUDA GPU MEMORY ====================
        
#         print("Before")
#         print(torch.cuda.memory_allocated()/1024**2)
#         print(torch.cuda.memory_reserved()/1024**2)

        gc.collect()
        del (model, optimizer)
        del train_data
        del test_data
        del val_data
        del train_loader
        del test_loader
#         del val_loader
#         del train_loader_eval
        del val_loader_eval
        
        torch.cuda.empty_cache()
        
#         print("After")
#         print(torch.cuda.memory_allocated()/1024**2)
#         print(torch.cuda.memory_reserved()/1024**2)

    # ==================== SAVE RELEVANT SUMMARY INFORMATION FOR CROSS-VALIDATION ====================
    
    if args.roi:
        pred_mean = float(np.mean(pred_acc_scores))
        pred_std = float(np.var(pred_acc_scores))
    else:
        pred_mean = None
        pred_std = None
        
    logs ={
        "model": args.model_name, 
        "num_epochs": best_epochs,
        "lr": args.lr, 
        "roi": args.roi,
        "batch_size": params.batch_size,
        "k": args.k,
        "acc_scores": acc_scores,
        "mean_acc": float(np.mean(acc_scores)),
        "var_acc": float(np.var(acc_scores)),
        "AUG": args.AUG,
        "AUG2": args.AUG2,
        "pred_acc_scores": pred_acc_scores,
        "pred_mean": pred_mean,
        "pred_std": pred_std
        }

    with open(
        os.path.join(params.log_dir, "summary_{}_{}.json".format(args.model_name+extra, args.k)), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':

    main()