#import libraries 
import numpy as np
import pandas as pd
import time
import os
import itertools
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from PIL import Image
from utils.landmark_prep import prep_landmarks, prep_landmarks_no_femur
from utils.process_predictions import pixel_to_mm

class dice_metric(nn.Module):
    def __init__(self):
        super(dice_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        
        target[target < 0.5] = 0.0
        target[target >= 0.5] = 1.0            
        prediction[prediction < 0.5 ] = 0 
        prediction[prediction >= 0.5] = 1.0 

        # Dice coef
        smooth = 0
        intersection = torch.sum(target * prediction,axis=[1,2,3])
        union = torch.sum(target,axis=[1,2,3]) + torch.sum(prediction,axis=[1,2,3])
        dice = torch.mean((2. * intersection + smooth)/(union + smooth),axis=0)
        
        return dice.cpu().detach().numpy()
    
class test_dice_metric(nn.Module):
    def __init__(self):
        super(test_dice_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        
        target[target < 0.5] = 0.0
        target[target >= 0.5] = 1.0            
        prediction[prediction < 0.5 ] = 0 
        prediction[prediction >= 0.5] = 1.0 

        # Dice coef
        smooth = 0
        intersection = torch.sum(target * prediction,axis=[1,2,3])
        union = torch.sum(target,axis=[1,2,3]) + torch.sum(prediction,axis=[1,2,3])
        dice = torch.mean((2. * intersection + smooth)/(union + smooth),axis=0)
        
        return dice.cpu().detach().numpy()
    
# class dice_metric(nn.Module):
#     def __init__(self):
#         super(dice_metric, self).__init__()
    
#     def forward(self,target,prediction,filename,params,subdir="Val"):
        
#         target[target < 0.5] = 0.0
#         target[target >= 0.5] = 1.0            
#         prediction[prediction < 0.5 ] = 0 
#         prediction[prediction >= 0.5] = 1.0 

#         # Dice coef
#         smooth = 1
#         intersection = torch.sum(target * prediction)
#         union = torch.sum(target[target==1]) + torch.sum(prediction[prediction==1])
#         dice = torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)

#         return dice.cpu().detach().numpy()
    

class mse_metric(nn.Module):
    def __init__(self):
        super(mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset")
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        img = Image.open(os.path.join(root2,params.image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)

        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        if params.num_classes > 6:
            csv_dir = os.path.join(root2,"CSVs")
            lm_targets, __ = prep_landmarks(filename,csv_dir)
        else: 
            csv_dir = os.path.join(root2,"ROI CSVs")
            lm_targets = pd.read_csv(os.path.join(csv_dir,filename+'.csv'))
            lm_targets = np.asarray(lm_targets).astype(float)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse
    
    
# class pb_mse_metric(nn.Module):
#     def __init__(self):
#         super(pb_mse_metric, self).__init__()
    
#     def forward(self,target,prediction,filename,params,subdir,AUG):
#         prediction = prediction.cpu().detach().numpy()
#         lm_pred = np.zeros((params.num_classes-1,2))
#         root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#         root2 = os.path.join(root,"Dataset",subdir)
        
#         # Get most likely landmark locations based on heatmap predictions
#         for i in range(params.num_classes-1):
#             lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
#                                            (params.input_size,params.input_size))
#             lm_preds = np.asarray(lm_preds).astype(float)
#             lm_pred[i,0] = lm_preds[1]
#             lm_pred[i,1] = lm_preds[0]
            
#         # Use input image to resize predictions
#         if AUG:
#             image_dir = params.image_dir + " AUG"
#             target_dir = "CSVs AUG"
#         else:
#             image_dir = params.image_dir
#             target_dir = "CSVs"
#         img = Image.open(os.path.join(root2,image_dir,filename+".png"))
#         img_size = img.size
#         img_size = np.asarray(img_size).astype(float)
        
#         lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
#         lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

#         # Get targets
#         csv_dir = os.path.join(root2,target_dir)
#         lm_targets, __ = prep_landmarks(filename,csv_dir)
#         lm_targets = lm_targets.reshape((-1,2))
#         lm_targets = np.nan_to_num(lm_targets)
        
#         # Convert from pixels to mm
#         lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
#         lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
#         lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
#         lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

#         mse = mean_squared_error(lm_targets, lm_pred, squared=True)

#         return mse


class pb_mse_metric_back(nn.Module):
    def __init__(self):
        super(pb_mse_metric_back, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes-1,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset",subdir)
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes-1):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse


class pb_mse_metric(nn.Module):
    def __init__(self):
        super(pb_mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset",subdir)
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse
    

class test_pb_mse_metric(nn.Module):
    def __init__(self):
        super(test_pb_mse_metric, self).__init__()
    
    def forward(self,prediction,filename,params,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset","FINAL TEST")
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse
   
        
class pb_roi_mse_metric(nn.Module):
    def __init__(self):
        super(pb_roi_mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = [0,0]
        target = target.cpu().detach().numpy()
        lm_tar = [0,0]
        
        lm_num = int(filename.split("_")[-2])
        if lm_num >= 12:
            lm_num = lm_num-11
        # Get most likely landmark locations based on heatmap predictions
        lm_preds = np.unravel_index(prediction[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_preds = np.asarray(lm_preds).astype(float)
        lm_pred[0] = lm_preds[1]
        lm_pred[1] = lm_preds[0]
        
        # Get most likely landmark locations based on heatmap predictions
        lm_tars = np.unravel_index(target[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_tars = np.asarray(lm_tars).astype(float)
        lm_tar[0] = lm_tars[1]
        lm_tar[1] = lm_tars[0]
        
        # Convert from pixels to mm
        lm_tar[0] = pixel_to_mm(filename,lm_tar[0])
        lm_tar[1] = pixel_to_mm(filename,lm_tar[1])
        lm_pred[0] = pixel_to_mm(filename,lm_pred[0])
        lm_pred[1] = pixel_to_mm(filename,lm_pred[1])

        mse = mean_squared_error(lm_tar, lm_pred, squared=square)
#         mse = math.dist(lm_tar,lm_pred)

        return mse


class test_pb_roi_mse_metric(nn.Module):
    def __init__(self):
        super(test_pb_roi_mse_metric, self).__init__()
    
    def forward(self,prediction,filename,params,AUG,prediction_dir,square=True):
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        prediction = prediction.cpu().detach().numpy()
        lm_pred = [0,0]
        
        lm_num = int(filename.split("_")[-2])
        if lm_num >= 12:
            lm_num = lm_num-11
        # Get most likely landmark locations based on heatmap predictions
        lm_preds = np.unravel_index(prediction[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_preds = np.asarray(lm_preds).astype(float)
        
        image_num = filename.split("_")[0]
        lm_num = int(filename.split("_")[-2])
    
        targets, __ = prep_landmarks(image_num,os.path.join(root,"Dataset","FINAL TEST","CSVs"))
        targets = targets.reshape((-1,2))
        targets = np.nan_to_num(targets)
        
        end3 = prediction_dir.split('\\')[-1].split(" ")[0:2]
        ex = ""
        end = end3[0]
        if "_AUG2" in end:
            end2 = end[0:-5]
            ex = " AUG2"
        else:
            end2 = end
        end2 = end2 + " "

        if end3[1] == "Pred":
            centre_dir = os.path.join(root,"Dataset","FINAL TEST",end2+"ROI LM Top-Lefts")
            end = end3[0] + " " + end3[1]
        else:
            centre_dir = os.path.join(root,"Dataset","FINAL TEST","ROI LM Top-Lefts" + ex)
            end = end3[0]
        
        centres = pd.read_csv(os.path.join(centre_dir,image_num+'.csv'))
        centres = np.asarray(centres)
        
        if lm_num >= 12:
            lm_pred[0] = -lm_preds[1] + (centres[lm_num-3,1]+params.input_size-1)
            lm_pred[1] = lm_preds[0] + centres[lm_num-3,2]
        else:
            lm_pred[0] = lm_preds[1] + centres[lm_num-1,1]
            lm_pred[1] = lm_preds[0] + centres[lm_num-1,2]

        # Convert from pixels to mm
        targets[lm_num-1,0] = pixel_to_mm(filename,targets[lm_num-1,0])
        targets[lm_num-1,1] = pixel_to_mm(filename,targets[lm_num-1,1])
        lm_pred[0] = pixel_to_mm(filename,lm_pred[0])
        lm_pred[1] = pixel_to_mm(filename,lm_pred[1])

        mse = mean_squared_error(targets[lm_num-1,:], lm_pred, squared=square)
#         mse = math.dist(lm_tar,lm_pred)

        return mse


def pb_roi_mse_metric_alt_ver(targets,predictions,filenames,params,AUG,square=True):
        
    mses = []
    for k in range(targets.shape[0]):
        
        prediction = predictions[k,:,:,:]
        target = targets[k,:,:,:]
        filename = filenames[k].split("\\")[-1][:-4]
        subdir = filenames[k].split("\\")[-3]
        prediction = prediction.cpu().detach().numpy()
        
        lm_pred = [0,0]
        target = target.cpu().detach().numpy()
        lm_tar = [0,0]

        lm_num = int(filename.split("_")[-2])
        if lm_num >= 12:
            lm_num = lm_num-11
            
        # Get most likely landmark locations based on heatmap predictions
        lm_preds = np.unravel_index(prediction[lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_preds = np.asarray(lm_preds).astype(float)
        lm_pred[0] = lm_preds[1]
        lm_pred[1] = lm_preds[0]

        # Get most likely landmark locations based on heatmap predictions
        lm_tars = np.unravel_index(target[lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_tars = np.asarray(lm_tars).astype(float)
        lm_tar[0] = lm_tars[1]
        lm_tar[1] = lm_tars[0]

        # Convert from pixels to mm
        lm_tar[0] = pixel_to_mm(filename,lm_tar[0])
        lm_tar[1] = pixel_to_mm(filename,lm_tar[1])
        lm_pred[0] = pixel_to_mm(filename,lm_pred[0])
        lm_pred[1] = pixel_to_mm(filename,lm_pred[1])

        mse = mean_squared_error(lm_tar, lm_pred, squared=square)
#         mse = math.dist(lm_tar,lm_pred)

        mses.append(mse)

    alt_loss = sum(mses)/len(mses)
    alt_loss = alt_loss * 0.000001

    return alt_loss

    
def mse_metric_alt_ver(targets,predictions,filenames,params,AUG,square=True):
        
    mses = []
    for k in range(targets.shape[0]):
        prediction = predictions[k,:,:,:]
        filename = filenames[k].split("\\")[-1][:-4]
        subdir = filenames[k].split("\\")[-3]
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset",subdir)

        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]

        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)

        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)

        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        mses.append(mse)
    
    alt_loss = sum(mses)/len(mses)
    alt_loss = alt_loss * 0.000001

    return alt_loss


class custom_weighted_loss(nn.Module):
    def __init__(self):
        super(custom_weighted_loss, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()
        diff = (prediction - target)**2
        weighted = diff*(weight_map*9+1)
        loss = torch.mean(weighted)
        return loss
    
    
class custom_weighted_loss2(nn.Module):
    def __init__(self):
        super(custom_weighted_loss, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map*10).int()
        diff = (prediction - target)**2
        weighted = diff*(weight_map+1)
        loss = torch.mean(weighted)
        return loss
