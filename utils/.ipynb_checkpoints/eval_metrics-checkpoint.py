#import libraries 
import numpy as np
import pandas as pd
import time
import os
import itertools
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt
from pydicom import dcmread


class pb_mse_metric(nn.Module):
    def __init__(self):
        super(pb_mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = '//data/scratch/r094879/data/'

        cumulative_sum = np.zeros(prediction.shape[2:])
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            # cumulative_sum += prediction[0,i,:,:]
            # plt.imshow(prediction[0,i,:,:], cmap='gray')
            # plt.axis('off')
            # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_long",filename+'_'+str(i)+'.png'))
            # plt.close()     

        # plt.imshow(cumulative_sum, cmap='gray')
        # plt.title("Cumulative Sum of All Slices")
        # plt.axis('off')
        # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_long",filename+'.png'))
        # plt.close()
    
        # Use input image to resize predictions
        image_dir = params.image_dir
        target_dir = "annotations/"
        img = dcmread(os.path.join(root,image_dir,filename+".dcm"))
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * float(img_size[1])/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * float(img_size[0])/float(params.input_size)

        # Get targets

        csv_file = os.path.join(root,'annotations/annotations.csv')
        csv_df = pd.read_csv(csv_file)

        filtered_row = csv_df[csv_df['image'] == filename]

        x_values = np.array(filtered_row.iloc[:,3:29:2].values).reshape((-1,1))
        y_values = np.array(filtered_row.iloc[:,4:29:2].values).reshape((-1,1))

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.concatenate([x_values,y_values],axis=1)
        
        # print(lm_pred)
        # print(xy_pairs)

        lm_targets = xy_pairs.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)

        lm_tars = []
        lm_preds = []

        for i in range(len(lm_targets)):
            if int(lm_targets[i][0]) != 0:
                lm_tars.append(lm_targets[i])
                lm_preds.append(lm_pred[i])

        lm_targets = np.array(lm_tars).reshape((-1,2))
        lm_pred = np.array(lm_preds).reshape((-1,2))

        mse = mean_squared_error(lm_targets, lm_pred)

        # print(filename)
        # print(mse)

        return mse


class pb_mse_metric_test(nn.Module):
    def __init__(self):
        super(pb_mse_metric_test, self).__init__()
    
    def forward(self,target,prediction,filename,params,name):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = '//data/scratch/r094879/data/'

        cumulative_sum = np.zeros(prediction.shape[2:])
        max_val = np.zeros(13)

        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            max_val[i] = prediction[0,i,int(lm_preds[0]),int(lm_preds[1])]
        #     cumulative_sum += prediction[0,i,:,:]
        #     plt.imshow(prediction[0,i,:,:], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(os.path.join("//data/scratch/r094879/data/results",name,'heatmaps',filename+'_'+str(i)+'.png'))
        #     plt.close()     

        # plt.imshow(cumulative_sum, cmap='gray')
        # plt.title("Cumulative Sum of All Slices")
        # plt.axis('off')
        # plt.savefig(os.path.join("//data/scratch/r094879/data/results",name,'heatmaps',filename+'.png'))
        # plt.close()
    
        # Use input image to resize predictions
        image_dir = params.image_dir
        target_dir = "annotations/"
        img = dcmread(os.path.join(root,image_dir,filename+".dcm"))    
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * float(img_size[1])/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * float(img_size[0])/float(params.input_size)

        # Get targets

        csv_file = os.path.join(root,'annotations/annotations.csv')
        csv_df = pd.read_csv(csv_file)

        filtered_row = csv_df[csv_df['image'] == filename]

        x_values = np.array(filtered_row.iloc[:,3:29:2].values).reshape((-1,1))
        y_values = np.array(filtered_row.iloc[:,4:29:2].values).reshape((-1,1))

        stats_df = pd.read_csv(os.path.join('//data/scratch/r094879/data/stats',name+'.csv'))

        stats_df.loc[stats_df["image"]==str(filename),"T4x"] = lm_pred[0,0]
        stats_df.loc[stats_df["image"]==str(filename),"T4y"] = lm_pred[0,1]
        stats_df.loc[stats_df["image"]==str(filename),"T4_val"] = max_val[0]
        stats_df.loc[stats_df["image"]==str(filename),"T5x"] = lm_pred[1,0]
        stats_df.loc[stats_df["image"]==str(filename),"T5y"] = lm_pred[1,1]
        stats_df.loc[stats_df["image"]==str(filename),"T5_val"] = max_val[1]
        stats_df.loc[stats_df["image"]==str(filename),"T6x"] = lm_pred[2,0]
        stats_df.loc[stats_df["image"]==str(filename),"T6y"] = lm_pred[2,1]
        stats_df.loc[stats_df["image"]==str(filename),"T6_val"] = max_val[2]
        stats_df.loc[stats_df["image"]==str(filename),"T7x"] = lm_pred[3,0]
        stats_df.loc[stats_df["image"]==str(filename),"T7y"] = lm_pred[3,1]
        stats_df.loc[stats_df["image"]==str(filename),"T7_val"] = max_val[3]
        stats_df.loc[stats_df["image"]==str(filename),"T8x"] = lm_pred[4,0]
        stats_df.loc[stats_df["image"]==str(filename),"T8y"] = lm_pred[4,1]
        stats_df.loc[stats_df["image"]==str(filename),"T8_val"] = max_val[4]
        stats_df.loc[stats_df["image"]==str(filename),"T9x"] = lm_pred[5,0]
        stats_df.loc[stats_df["image"]==str(filename),"T9y"] = lm_pred[5,1]
        stats_df.loc[stats_df["image"]==str(filename),"T9_val"] = max_val[5]
        stats_df.loc[stats_df["image"]==str(filename),"T10x"] = lm_pred[6,0]
        stats_df.loc[stats_df["image"]==str(filename),"T10y"] = lm_pred[6,1]
        stats_df.loc[stats_df["image"]==str(filename),"T10_val"] = max_val[6]
        stats_df.loc[stats_df["image"]==str(filename),"T11x"] = lm_pred[7,0]
        stats_df.loc[stats_df["image"]==str(filename),"T11y"] = lm_pred[7,1]
        stats_df.loc[stats_df["image"]==str(filename),"T11_val"] = max_val[7]
        stats_df.loc[stats_df["image"]==str(filename),"T12x"] = lm_pred[8,0]
        stats_df.loc[stats_df["image"]==str(filename),"T12y"] = lm_pred[8,1]
        stats_df.loc[stats_df["image"]==str(filename),"T12_val"] = max_val[8]
        stats_df.loc[stats_df["image"]==str(filename),"L1x"] = lm_pred[9,0]
        stats_df.loc[stats_df["image"]==str(filename),"L1y"] = lm_pred[9,1]
        stats_df.loc[stats_df["image"]==str(filename),"L1_val"] = max_val[9]
        stats_df.loc[stats_df["image"]==str(filename),"L2x"] = lm_pred[10,0]
        stats_df.loc[stats_df["image"]==str(filename),"L2y"] = lm_pred[10,1]
        stats_df.loc[stats_df["image"]==str(filename),"L2_val"] = max_val[10]
        stats_df.loc[stats_df["image"]==str(filename),"L3x"] = lm_pred[11,0]
        stats_df.loc[stats_df["image"]==str(filename),"L3y"] = lm_pred[11,1]
        stats_df.loc[stats_df["image"]==str(filename),"L3_val"] = max_val[11]
        stats_df.loc[stats_df["image"]==str(filename),"L4x"] = lm_pred[12,0]
        stats_df.loc[stats_df["image"]==str(filename),"L4y"] = lm_pred[12,1]
        stats_df.loc[stats_df["image"]==str(filename),"L4_val"] = max_val[12]

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.concatenate([x_values,y_values],axis=1)
        
        # print(lm_pred)
        # print(xy_pairs)

        lm_targets = xy_pairs.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)

        lm_tars = []
        lm_preds = []

        for i in range(len(lm_targets)):
            if int(lm_targets[i][0]) != 0:
                lm_tars.append(lm_targets[i])
                lm_preds.append(lm_pred[i])

        lm_targets = np.array(lm_tars).reshape((-1,2))
        lm_pred = np.array(lm_preds).reshape((-1,2))

        error_tar = []
        error_pred = []
        count = 0
        dist = (lm_targets[1,1]-lm_targets[0,1])/2
        print(dist)
        for i in range(len(lm_targets)):
            if abs(int(lm_preds[i][1])-int(lm_targets[i][1])) < dist:
                if abs(int(lm_preds[i][0])-int(lm_targets[i][0])) < dist:
                    count+=1
                    error_tar.append(lm_targets[i])
                    error_pred.append(lm_pred[i])
        
        if error_pred:            
            error_tar = np.array(error_tar).reshape((-1,2))
            error_pred = np.array(error_pred).reshape((-1,2))

            rmse = root_mean_squared_error(error_tar, error_pred)

            stats_df.loc[stats_df["image"]==str(filename),"RMSE"] = rmse   
        else:
            rmse = np.nan
            
                    
        id_acc = count/len(lm_targets)
        
        stats_df.loc[stats_df["image"]==str(filename),"correct"] = count
        stats_df.loc[stats_df["image"]==str(filename),"total"] = len(lm_targets)
        stats_df.loc[stats_df["image"]==str(filename),"id_acc"] = id_acc

        stats_df.to_csv(os.path.join('//data/scratch/r094879/data/stats',name+'.csv'),index=False)

        # print(filename)
        # print(mse)

        return rmse


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
        super(custom_weighted_loss2, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map*10).int()
        diff = (prediction - target)**2
        weighted = diff*(weight_map+1)
        loss = torch.mean(weighted)
        return loss


class custom_weighted_loss_2(nn.Module):
    # Custom weight loss that:
    #     1) ignore the layers that are immediately before/after layers that contain landmarks
    #     2) increase the error weighting in layers that contain no landmark at all 
    def __init__(self):
        super(custom_weighted_loss_2, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()
        weight_map = weight_map*9 + 1
        for k in range(weight_map.shape[0]):
            first = False
            last = False
            next = True
            for i in range(13):
                if 10 in weight_map[k,i,:,:]:
                    if not first:
                        first = True 
                        last = False
                        next = False
                        if i != 0:
                            weight_map[k,i-1,:,:] = weight_map[k,i-1,:,:]*0
                else:    
                    if first and not last:
                        first = False
                        last = True
                        next = True
                        weight_map[k,i,:,:] = weight_map[k,i,:,:]*0
                    elif next:
                        weight_map[k,i,:,:] = weight_map[k,i,:,:]*5
                        
        diff = (prediction - target)**2
        weighted = diff*weight_map
        loss = torch.mean(weighted)
        return loss


class custom_weighted_loss_3(nn.Module):
    # Custom weight loss that:
    #     1) ignore the layers that are immediately before/after layers that contain landmarks
    #     2) increase the error weighting in layers that contain no landmark at all 
    def __init__(self):
        super(custom_weighted_loss_3, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()

        weight_map_equal = np.zeros((1,1,weight_map.shape[2],weight_map.shape[2]))
        
        for k in range(weight_map.shape[0]):
            weight_map_equal = torch.sum(weight_map[k,:,:,:],0)
            # plt.imshow(weight_map_equal*255, cmap='gray')
            # plt.title("Weighted")
            # plt.axis('off')
            # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_weighted",str(k)+'.png'))
            # plt.close()
            
            first = False
            last = False
            next = True
            for i in range(13):
                if 1 in weight_map[k,i,:,:]:
                    weight_map[k,i,:,:] = weight_map_equal*9+1 
                    if not first:
                        first = True 
                        last = False
                        next = False
                        if i != 0:
                            weight_map[k,i-1,:,:] = weight_map_equal*9+1 
                else:    
                    if first and not last:
                        first = False
                        last = True
                        weight_map[k,i,:,:] = weight_map_equal*9+1 
                    elif last and not next:
                        next = True
                        weight_map[k,i,:,:] = (weight_map_equal+1)*5
                    elif next:
                        weight_map[k,i,:,:] = (weight_map_equal+1)*5
                        
        diff = (prediction - target)**2
        weighted = diff*weight_map
        loss = torch.mean(weighted)
        return loss