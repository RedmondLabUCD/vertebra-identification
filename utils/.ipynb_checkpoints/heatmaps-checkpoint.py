#import libraries 
import numpy as np
import pandas as pd 
import os
import math
import cv2
import matplotlib.pyplot as plt


def create_hm(landmarks,old_dim,new_dim,size=3):
    '''
    Resize landmarks to desired image dimensions and create heatmaps.
    '''
    lm = landmarks.copy()
    
    # Rescale landmark coordinates
    lm = resize_lm(lm,old_dim,new_dim)

    # Create heatmap
    hm = generate_hm(lm,new_dim,s=size)
                           
    return hm


def pb_create_hm_aug(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps AUG",
                     tl_dir="ROI LM Top-Lefts AUG",dim=128,size=3):
    '''
    Creates local heatmaps for Plan B isolate landmark ROIs.
    '''
    lm = landmarks.copy()
    lm = np.nan_to_num(landmarks)

    tls = pd.read_csv(os.path.join(current_dir,tl_dir,filename +".csv"))
    tls = np.asarray(tls).astype(float)

    for tl in tls:
        if int(lm[int(tl[0]-1),1]) != 0 and int(lm[int(tl[0]-1),1]) != 0:
            if tl[0] < 12: 
                lm_new = np.empty((11,2))
                lm_new[:] = np.nan

                for i in range(11):
                    lms = lm[i,:]
                    if lms[0]-tl[1] < dim and lms[0]-tl[1] >= 0 and lms[1]-tl[2] < dim and lms[1]-tl[2] >= 0:
                        lm_new[i,0] = lms[0]-tl[1]
                        lm_new[i,1] = lms[1]-tl[2]

                hm = generate_hm(lm_new,dim,s=size)
#                 np.save(os.path.join(current_dir,save_dir,
#                                          filename+"_r_"+str(int(tl[0]))+"_0"),hm)
                np.save(os.path.join(current_dir,save_dir,
                                         filename+"_r_"+str(int(tl[0]))+"_" + str(int(tl[3]))),hm)
                
#                 print("Right " + str(tl[0]) + " " + str(tl[3]))
#                 print(os.path.join(current_dir,"ROI LMs", filename +"_r_"+str(int(tl[0]))+"_" + str(int(tl[3])) +".png"))
#                 img = cv2.imread(os.path.join(current_dir,"ROI LMs AUG",
#                                               filename +"_r_"+str(int(tl[0]))+"_" + str(int(tl[3])) +".png"))
#                 fig = plt.figure(figsize=(20,6))
#                 plt.imshow(img[:,:,0],cmap="gray")
#                 plt.show()

#                 fig = plt.figure(figsize=(18,10))
#                 for j in range(12):
#                     ax = fig.add_subplot(2,6,j+1)
#                     ax.imshow(hm[:,:,j])
#                 plt.show()
                
            elif tl[0] >= 12: 
                lm_new = np.empty((11,2))
                lm_new[:] = np.nan

                for i in range(11,22):
                    lms = lm[i,:]
                    if lms[0]-tl[1] < dim and lms[0]-tl[1] >= 0 and lms[1]-tl[2] < dim and lms[1]-tl[2] >= 0:
                        lm_new[i-11,0] = -lms[0] + (tl[1] + (dim-1))
                        lm_new[i-11,1] = lms[1]-tl[2]
                        
                hm = generate_hm(lm_new,dim,s=size)
#                 np.save(os.path.join(current_dir,save_dir,
#                                          filename+"_l_"+str(int(tl[0]))),hm)
                np.save(os.path.join(current_dir,save_dir,
                                         filename+"_l_"+str(int(tl[0]))+"_" + str(int(tl[3]))),hm)

#                 print("Left " + str(tl[0]) + " " + str(tl[3]))
#                 img = cv2.imread(os.path.join(current_dir,"ROI LMs AUG",
#                                               filename +"_l_"+str(int(tl[0]))+"_" + str(int(tl[3])) +".png"))
#                 fig = plt.figure(figsize=(20,6))
#                 plt.imshow(img[:,:,0],cmap="gray")
#                 plt.show()

#                 fig = plt.figure(figsize=(18,10))
#                 for j in range(12):
#                     ax = fig.add_subplot(2,6,j+1)
#                     ax.imshow(hm[:,:,j])
#                 plt.show()

                
def gaussian_k(x0,y0,sigma,height,width):
        x = np.arange(0,height,1,float) ## (width,)
        y = np.arange(0,width,1,float)[:,np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))

    
def generate_hm(landmarks,dim,s=3):
        ''' 
        Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
            
        '''
        dim = int(dim)
        Nlandmarks = len(landmarks)
        hm = np.zeros((dim,dim,Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.isnan(landmarks[i]).any():
                hm[:,:,i] = gaussian_k(landmarks[i][0],landmarks[i][1],s,dim,dim)
            else:
                hm[:,:,i] = np.zeros((dim,dim), dtype = np.float32)
        return hm
    
