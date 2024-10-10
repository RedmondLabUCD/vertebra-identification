from __future__ import print_function
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import torch.optim as optim
from tqdm import tqdm
from utils.train_progress_tools import RunningAverage
from PIL import Image, ImageFilter, ImageOps
from utils.feature_extraction import get_contours, femhead_centre
import cv2 as cv


def dice_post_process(dice,extra,root,data_dir,params,prediction_dir,predictions,filename,subdir,metric_avg):
    
    predictions[predictions < 0.5] = 0 
    predictions[predictions >= 0.5] = 1.0
    mask = tF.to_pil_image(predictions.cpu()[0,0])
    mask = mask.convert(mode='1')
    
    tar = Image.open(os.path.join(root,"Dataset","Fold "+subdir,params.target_dir,
                                  filename+params.target_sfx))
    tar = tar.convert(mode='1')
    tar = np.asarray(tar)
    tar = tar.copy()
    tar[tar < 0.5 ] = 0 
    tar[tar >= 0.5] = 1.0
    
    # Use input image to resize predictions√how 
    img = Image.open(os.path.join(root ,"Dataset","Fold "+subdir,params.image_dir,
                                  filename+".png"))
    mask = mask.resize(img.size)
    msk = np.asarray(mask)
    dice_score = dice_function(msk,tar)
    
    resize_column = str(dice) + "_Resized"
    smooth_column = str(dice) + "_Smooth"
    
    if "UNet_ROI_FemHead" in str(dice):
        half_dice_stats = half_dice(root,data_dir,dice,resize_column,smooth_column)
        half_dice_stats.loc[half_dice_stats.Image==str(filename),[str(dice),"Fold"]] = metric_avg, subdir
        # Save the prediction
        if not os.path.exists(prediction_dir + " halves"): os.makedirs(prediction_dir + " halves")
        mask.save(os.path.join(prediction_dir + " halves",filename+".png")) 
        half_dice_stats.loc[half_dice_stats.Image==str(filename),str(resize_column)] = dice_score
        half_dice_stats.to_csv(os.path.join(data_dir,'half_dice_stats.csv'),index=False)
                    
    if not os.path.exists(os.path.join(data_dir,"dice_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name)
        dice_stats = pd.DataFrame({"Image":names})
        dice_stats["Fold"] = ""
        dice_stats[str(dice)] = ""
        dice_stats[str(resize_column)] = ""
        dice_stats[str(smooth_column)] = ""
        dice_stats.to_csv(os.path.join(data_dir,'dice_stats.csv'),index=False)
        dice_stats = pd.read_csv(os.path.join(data_dir,"dice_stats.csv"))
    else:
        dice_stats = pd.read_csv(os.path.join(data_dir,"dice_stats.csv"))
        if str(dice) not in dice_stats.columns:
            dice_stats[str(dice)] = ""
        if str(resize_column) not in dice_stats.columns:
            dice_stats[str(resize_column)] = ""
        if str(smooth_column) not in dice_stats.columns:
            dice_stats[str(smooth_column)] = ""
        dice_stats.to_csv(os.path.join(data_dir,'dice_stats.csv'),index=False)
        dice_stats = pd.read_csv(os.path.join(data_dir,"dice_stats.csv"))
        
    if "UNet_ROI_FemHead" not in str(dice):
        dice_stats.loc[dice_stats.Image==int(filename),[str(dice),"Fold"]] = metric_avg, subdir    
                
    if "UNet_ROI_FemHead" in str(dice):
        full_mask = make_full_mask(filename,prediction_dir,subdir,root,dice)
        filename = filename[:-2]
    else:
        dice_stats.loc[dice_stats.Image==int(filename),str(resize_column)] = dice_score
        full_mask = mask
  
    if full_mask is not None:
        full_msk = np.asarray(full_mask)
        full_msk = full_msk.copy()

        full_msk[full_msk < 0.5 ] = 0 
        full_msk[full_msk >= 0.5] = 1.0

        full_tar = Image.open(os.path.join(root,"Dataset","Fold "+subdir,"FemHead Masks",
                                      filename+".png"))
        full_tar = full_tar.convert(mode='1')
        full_tar = np.asarray(full_tar)
        full_tar = full_tar.copy()
        full_tar[full_tar < 0.5 ] = 0 
        full_tar[full_tar >= 0.5] = 1.0

        dice_full = dice_function(full_msk,full_tar)

        dice_stats.loc[dice_stats.Image==int(filename),str(resize_column)] = dice_full

        # Save the prediction
        full_mask.save(os.path.join(prediction_dir,filename+".png"))

        post_process_masks(prediction_dir,filename)

        # Assess post-processed image
        mask = Image.open(os.path.join(prediction_dir + " Smooth",filename+".png"))
        smooth_full_msk = np.asarray(mask)
        smooth_full_msk = smooth_full_msk.copy()
        
        smooth_full_msk[smooth_full_msk < 0.5 ] = 0 
        smooth_full_msk[smooth_full_msk >= 0.5] = 1.0

        dice_smooth = dice_function(smooth_full_msk,full_tar)

        dice_stats.loc[dice_stats["Image"]==int(filename),str(smooth_column)] = dice_smooth

    dice_stats.to_csv(os.path.join(data_dir,'dice_stats.csv'),index=False)
    
    
def test_dice_post_process(dice,extra,root,data_dir,params,prediction_dir,predictions,filename,metric_avg):
    
    predictions[predictions < 0.5] = 0 
    predictions[predictions >= 0.5] = 1.0
    mask = tF.to_pil_image(predictions.cpu()[0,0])
    mask = mask.convert(mode='1')
    
    tar = Image.open(os.path.join(root,"Dataset","FINAL TEST",params.target_dir,
                                  filename+params.target_sfx))
    tar = tar.convert(mode='1')
    tar = np.asarray(tar)
    tar = tar.copy()
    tar[tar < 0.5 ] = 0 
    tar[tar >= 0.5] = 1.0
    
    # Use input image to resize predictions√how 
    img = Image.open(os.path.join(root,"Dataset","FINAL TEST",params.image_dir,
                                  filename+".png"))
    mask = mask.resize(img.size)
    msk = np.asarray(mask)
    
    resize_column = str(dice) + "_Resized"
    smooth_column = str(dice) + "_Smooth"
    
    if "UNet_ROI_FemHead" in str(dice):
        # Save the prediction
        if not os.path.exists(prediction_dir + " halves"): os.makedirs(prediction_dir + " halves")
        mask.save(os.path.join(prediction_dir + " halves",filename+".png"))   
                
    if "UNet_ROI_FemHead" in str(dice):
        full_mask = test_make_full_mask(filename,prediction_dir,root,extra)
        filename = filename[:-2]
        print("here")
    else:
        full_mask = mask
  
    if full_mask is not None:
        full_msk = np.asarray(full_mask)
        full_msk = full_msk.copy()

        full_msk[full_msk < 0.5 ] = 0 
        full_msk[full_msk >= 0.5] = 1.0

        full_tar = Image.open(os.path.join(root,"Dataset","FINAL TEST","FemHead Masks",
                                      filename+".png"))
        full_tar = full_tar.convert(mode='1')
        full_tar = np.asarray(full_tar)
        full_tar = full_tar.copy()
        full_tar[full_tar < 0.5 ] = 0 
        full_tar[full_tar >= 0.5] = 1.0

        # Save the prediction
        full_mask.save(os.path.join(prediction_dir,filename+".png"))

        post_process_masks(prediction_dir,filename)

        # Assess post-processed image
        mask = Image.open(os.path.join(prediction_dir + " Smooth",filename+".png"))
        smooth_full_msk = np.asarray(mask)
        smooth_full_msk = smooth_full_msk.copy()
        
        smooth_full_msk[smooth_full_msk < 0.5 ] = 0 
        smooth_full_msk[smooth_full_msk >= 0.5] = 1.0

                
def half_dice(root,data_dir,dice,resize_column,smooth_column):
    
    if not os.path.exists(os.path.join(data_dir,"half_dice_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name+"_l")
            names.append(image_name+"_r")
        half_dice_stats = pd.DataFrame({"Image":names})
        half_dice_stats["Fold"] = ""
        half_dice_stats[str(dice)] = ""
        half_dice_stats[str(resize_column)] = ""
        half_dice_stats[str(smooth_column)] = ""
        half_dice_stats.to_csv(os.path.join(data_dir,'half_dice_stats.csv'),index=False)
        half_dice_stats = pd.read_csv(os.path.join(data_dir,"half_dice_stats.csv"))
    else:
        half_dice_stats = pd.read_csv(os.path.join(data_dir,"half_dice_stats.csv"))
        if str(dice) not in half_dice_stats.columns:
            half_dice_stats[str(dice)] = ""
        if str(resize_column) not in half_dice_stats.columns:
            half_dice_stats[str(resize_column)] = ""
        if str(smooth_column) not in half_dice_stats.columns:
            half_dice_stats[str(smooth_column)] = ""
        half_dice_stats.to_csv(os.path.join(data_dir,'half_dice_stats.csv'),index=False)
        half_dice_stats = pd.read_csv(os.path.join(data_dir,"half_dice_stats.csv"))
        
    return half_dice_stats


def test_make_full_mask(filename,prediction_dir,root,dice):
    
    mask = Image.open(os.path.join(prediction_dir + " halves",filename+".png"))
                    
    if filename[-1] == "l":
        mask = ImageOps.mirror(mask)

    full_img = Image.open(os.path.join(root,"Dataset","FINAL TEST","Images",
                                  filename[:-2]+".png"))

    if "Pred" in str(dice):
        contl, contr = get_contours(os.path.join(root,"Results","Final_UNet_ROI","Predicted ROI Masks",
                                                 filename[:-2]+".png"))
        centroid_r, centroid_l = femhead_centre(os.path.join(root,"Results","Final_UNet_ROI","Predicted ROI Masks",
                                                 filename[:-2]+".png"))

        x_dist = max((np.max(contr[:,1])-centroid_r[1]),(centroid_r[1]-np.min(contr[:,1])),
                 (np.max(contl[:,1])-centroid_l[1]),(centroid_l[1]-np.min(contl[:,1])))
        y_dist = max((np.max(contr[:,0])-centroid_r[0]),(centroid_r[0]-np.min(contr[:,0])),
                     (np.max(contl[:,0])-centroid_l[0]),(centroid_l[0]-np.min(contl[:,0])))
    
        dist = max(x_dist,y_dist)
        
        # define half length of the square ROI
        dist = round(1.2*dist)

        if filename[-1] == "l":
            pos = (int(centroid_l[1]-dist), int(centroid_l[0]-dist))
        else:
            pos = (int(centroid_r[1]-dist), int(centroid_r[0]-dist))

    else:
        contl, contr = get_contours(os.path.join(root,"Dataset","FINAL TEST","ROI Masks",
                                                 filename[:-2]+".png"))
        if filename[-1] == "l":
            pos = (int(np.min(contl[:,1])), int(np.min(contl[:,0])))
        else:
            pos = (int(np.min(contr[:,1])), int(np.min(contr[:,0])))

    if not os.path.exists(os.path.join(prediction_dir,filename[:-2]+".png")):
        full_mask = Image.new('1',full_img.size)
        full_mask.paste(mask,pos)
        full_mask.save(os.path.join(prediction_dir,filename[:-2]+".png"))
        return None
    else:
        full_mask = Image.open(os.path.join(prediction_dir,filename[:-2]+".png"))
        full_mask.paste(mask,pos)
        full_mask = full_mask.convert(mode='1')
        return full_mask


def make_full_mask(filename,prediction_dir,subdir,root,dice):
    
    mask = Image.open(os.path.join(prediction_dir + " halves",filename+".png"))
                    
    if filename[-1] == "l":
        mask = ImageOps.mirror(mask)

    full_img = Image.open(os.path.join(root,"Dataset","Fold "+subdir,"Images",
                                  filename[:-2]+".png"))

    if "Pred" in str(dice):
        contl, contr = get_contours(os.path.join(root,"Results","UNet_ROI","Predicted ROI Masks",
                                                 filename[:-2]+".png"))
    else:
        contl, contr = get_contours(os.path.join(root,"Dataset","Fold "+str(subdir),"ROI Masks",
                                                 filename[:-2]+".png"))

    if filename[-1] == "l":
        pos = (int(np.min(contl[:,1])), int(np.min(contl[:,0])))
    else:
        pos = (int(np.min(contr[:,1])), int(np.min(contr[:,0])))

    if not os.path.exists(os.path.join(prediction_dir,filename[:-2]+".png")):
        full_mask = Image.new('1',full_img.size)
        full_mask.paste(mask,pos)
        full_mask.save(os.path.join(prediction_dir,filename[:-2]+".png"))
        return None
    else:
        full_mask = Image.open(os.path.join(prediction_dir,filename[:-2]+".png"))
        full_mask.paste(mask,pos)
        full_mask = full_mask.convert(mode='1')
        return full_mask

    
def post_process_masks(prediction_dir,filename):
    
    # Post-process image by dilating, eroding, and then filling in holes
    mask = cv.imread(os.path.join(prediction_dir,filename+".png"),0)
    closing = cv.medianBlur(mask, ksize=13)
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)

    cnts, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    
    if len(cnts) > 0:
        out2 = closing.copy()
        out2 = cv.bitwise_not(out2)

        h, w = closing.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)

        cv.floodFill(out2, flood_mask, (0,0), 0)

        closing = closing | out2

        cnts, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        out = np.zeros(closing.shape, np.uint8)
        for index,cnt in enumerate(cnts):
            cv.drawContours(out, [cnt], -1, 255, cv.FILLED, 8)
            if index == 1:
                break
        out = cv.bitwise_and(closing, out)
    else:
        out = closing

    # Save post-processed image
    if not os.path.exists(prediction_dir + " Smooth"): os.makedirs(prediction_dir + " Smooth")
    cv.imwrite(os.path.join(prediction_dir + " Smooth",filename+".png"),out)
    
    
def dice_function(msk,tar):
    smooth = 1
    intersection = np.sum(msk[tar==True])
    union = np.sum(tar) + np.sum(msk)
    dice_score = np.mean((2. * intersection + smooth)/(union + smooth))
    return dice_score