#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import itertools
import math
from PIL import Image, ImageDraw
import cv2 as cv
import random

from utils.feature_extraction import femhead_centre, get_contours


def create_ROI_mask(current_dir,filename):
    '''
    Uses the femoral head mask and landmark coordinates to define a region of interest
    mask centered around the femoral head centroid and including the key landmarks.
    '''
    
    centroid_r, centroid_l = femhead_centre(os.path.join(current_dir,"FemHead Masks",filename+".png"))
    contl, contr = get_contours(os.path.join(os.path.join(current_dir,"FemHead Masks",filename+".png")))
    
    img = cv.imread(os.path.join(current_dir,"Images",filename+".png"))
    
    # Create mask background of appropriate size 
    mask = Image.new(mode="RGB", size=(img.shape[1],img.shape[0]))
    draw = ImageDraw.Draw(mask)

    # Find maximum distance from centroid to max and min landmark coordinates
    x_dist = max((np.max(contr[:,1])-centroid_r[1]),(centroid_r[1]-np.min(contr[:,1])),
                 (np.max(contl[:,1])-centroid_l[1]),(centroid_l[1]-np.min(contl[:,1])))
    y_dist = max((np.max(contr[:,0])-centroid_r[0]),(centroid_r[0]-np.min(contr[:,0])),
                 (np.max(contl[:,0])-centroid_l[0]),(centroid_l[0]-np.min(contl[:,0])))
    
    dist = max(x_dist,y_dist)
    
    # define half length of the square ROI
    dist = round(1.5*dist)
    
    # Define and draw square ROIs
    r_rect = [(centroid_r[1] + dist, centroid_r[0] + dist),
              (centroid_r[1] - dist, centroid_r[0] - dist)]
    l_rect = [(centroid_l[1] + dist, centroid_l[0] + dist),
              (centroid_l[1]- dist, centroid_l[0] - dist)]
    
    draw.rectangle(r_rect,fill=(255,255,255),outline=(0,0,0))
    draw.rectangle(l_rect,fill=(255,255,255),outline=(0,0,0))

#     mask.show()
    mask.save(os.path.join(current_dir,"ROI Masks",filename+".png"))
    
    return os.path.join(current_dir,"ROI Masks",filename+".png")
    
    
def extract_ROI(current_dir,filename,mask_dir="ROI Masks",img_dir="Images",
                save_dir="ROI", mask_roi_dir=None):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    
    if mask_roi_dir is None:
        mask_roi_dir = current_dir

    contl, contr = get_contours(os.path.join(mask_roi_dir,mask_dir,filename+".png"))

    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))

    # Crop out the ROI for left and right
    cropped_img_r = img[int(np.min(contr[:,0])):int(np.max(contr[:,0])),
                        int(np.min(contr[:,1])):int(np.max(contr[:,1]))]
    cropped_img_l = img[int(np.min(contl[:,0])):int(np.max(contl[:,0])),
                        int(np.min(contl[:,1])):int(np.max(contl[:,1]))]
    
    # Flip left ROI 
    cropped_img_l = cv.flip(cropped_img_l,1)

    # Save the ROIs
    cv.imwrite(os.path.join(current_dir,save_dir,filename+"_r.png"),cropped_img_r)
    cv.imwrite(os.path.join(current_dir,save_dir,filename+"_l.png"),cropped_img_l)
    

def final_extract_ROI(current_dir,filename,mask_dir="ROI Masks",img_dir="Images",
                save_dir="ROI", mask_roi_dir=None):
    '''
    Created for final test.
    '''
    
    if mask_roi_dir is None:
        mask_roi_dir = current_dir

    centroid_r, centroid_l = femhead_centre(os.path.join(mask_roi_dir,mask_dir,filename+".png"))
    contl, contr = get_contours(os.path.join(mask_roi_dir,mask_dir,filename+".png"))

    # Find maximum distance from centroid to max and min landmark coordinates
    x_dist = max((np.max(contr[:,1])-centroid_r[1]),(centroid_r[1]-np.min(contr[:,1])),
                 (np.max(contl[:,1])-centroid_l[1]),(centroid_l[1]-np.min(contl[:,1])))
    y_dist = max((np.max(contr[:,0])-centroid_r[0]),(centroid_r[0]-np.min(contr[:,0])),
                 (np.max(contl[:,0])-centroid_l[0]),(centroid_l[0]-np.min(contl[:,0])))

    dist = max(x_dist,y_dist)
    
    # define half length of the square ROI
    dist = round(1.2*dist)
    
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))

    # Crop out the ROI for left and right
    cropped_img_r = img[int(centroid_r[0]-dist):int(centroid_r[0]+dist),
                        int(centroid_r[1]-dist):int(centroid_r[1]+dist)]
    cropped_img_l = img[int(centroid_l[0]-dist):int(centroid_l[0]+dist),
                        int(centroid_l[1]-dist):int(centroid_l[1]+dist)]
    
    # Flip left ROI 
    cropped_img_l = cv.flip(cropped_img_l,1)

    # Save the ROIs
    cv.imwrite(os.path.join(current_dir,save_dir,filename+"_r.png"),cropped_img_r)
    cv.imwrite(os.path.join(current_dir,save_dir,filename+"_l.png"),cropped_img_l)
    
    
def extract_ROI_from_lm_aug(current_dir,filename,landmarks,image_size,dim=128,
                            img_dir="Images",save_dir="ROI LMs AUG",tl_dir="ROI LM Top-Lefts AUG"):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(landmarks)
    
    # Define array to collect the centre coordinates, landmark number, 
    # and off-centre number for each coordinate
    tl = np.zeros((22*2,4))
    tl[:,0] = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,
               12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22]
    
    # All points will have an ROI where they are at the centre
    do_all = [dim/2,dim/2]
    
    # Each point will also have an ROI where the point is off-centre
    options = [[dim/4,3*dim/4],[dim/4,dim/4],[3*dim/4,dim/4],[3*dim/4,3*dim/4]]
    
    # Create ROI for points 1 to 11 (right hip)
    for i in range(11):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            index = random.randint(0,3)
            ops = [do_all,options[index]]
            for num,op in enumerate(ops):
                y_width = op[1]
                x_width = op[0]
                
                # Deal with exceptions where the point desired as the centre is too close to edge of image
                if int(lm[i,1]) < y_width:
                    lm[i,1] = y_width
                if int(lm[i,1]) > image_size[1] - y_width:
                    lm[i,1] = image_size[1] - y_width
                if int(lm[i,0]) < x_width:
                    lm[i,0] = x_width
                    
                # Define and save cropped ROI
                cropped_img_r = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                    int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
                cv.imwrite(os.path.join(current_dir,save_dir,
                                        filename+"_r_"+str(i+1)+"_" + str((index+1)*num) +".png"),cropped_img_r)
                
                # Collect the top-left coordinate of the ROI 
                tl[2*i+num] = [i+1,lm[i,0]-x_width,lm[i,1]-y_width,(index+1)*num]

    # Create ROI for point 12 to 22 (left hip)
    for i in range(11,22):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            index = random.randint(0,3)
            ops = [do_all,options[index]]
            for num,op in enumerate(ops):
                y_width = op[1]
                x_width = op[0]
                
                # Deal with exceptions where the point desired as the centre is too close to edge of image
                if int(lm[i,1]) < y_width:
                    lm[i,1] = y_width
                if int(lm[i,1]) > image_size[1] - y_width:
                    lm[i,1] = image_size[1] - y_width
                if int(lm[i,0]) > image_size[0] - (dim-x_width):
                    lm[i,0] = image_size[0] - (dim-x_width)
                    
                # Define and save cropped ROI
                cropped_img_l = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                    int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
                cropped_img_l = cv.flip(cropped_img_l,1) # flip left ROI to appear like right ones
                cv.imwrite(os.path.join(current_dir,save_dir,
                                        filename+"_l_"+str(i+1)+"_" + str((index+1)*num) +".png"),cropped_img_l)
                
                # Collect the centre coordinate of the ROI  
                tl[2*i+num] = [i+1,lm[i,0]-x_width,lm[i,1]-y_width,(index+1)*num]
                
    pd.DataFrame(tl).to_csv(os.path.join(current_dir,tl_dir,filename+".csv"),index=False)

    
def extract_ROI_from_lm_aug2(current_dir,filename,landmarks,image_size,dim=128,
                            img_dir="Images",save_dir="ROI LMs AUG2",tl_dir="ROI LM Top-Lefts AUG2"):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This time, the augmentation involves random placement of the landmark within the ROI.
    '''
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(landmarks)
    
    # Define array to collect the centre coordinates, landmark number, 
    # and off-centre number for each coordinate
    tl = np.zeros((18,4))
    tl[:,0] = [1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20]
    min_r = int(dim*0.1)
    max_r = int(dim*0.9)
    # Create ROI for points 1 to 9 (right hip)
    for i in range(9):
        y_width = random.randint(min_r,max_r)
        x_width = random.randint(min_r,max_r)
        
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < y_width:
                lm[i,1] = y_width
            if int(lm[i,1]) > image_size[1] - y_width:
                lm[i,1] = image_size[1] - y_width
            if int(lm[i,0]) < x_width:
                lm[i,0] = x_width

            # Define and save cropped ROI
            cropped_img_r = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_r_"+str(i+1)+"_0.png"),cropped_img_r)

            # Collect the top-left coordinate of the ROI 
            tl[i] = [i+1,lm[i,0]-x_width,lm[i,1]-y_width,0]

    # Create ROI for point 12 to 22 (left hip)
    for i in range(11,20):
        y_width = random.randint(min_r,max_r)
        x_width = random.randint(min_r,max_r)
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < y_width:
                lm[i,1] = y_width
            if int(lm[i,1]) > image_size[1] - y_width:
                lm[i,1] = image_size[1] - y_width
            if int(lm[i,0]) > image_size[0] - (dim-x_width):
                lm[i,0] = image_size[0] - (dim-x_width)

            # Define and save cropped ROI
            cropped_img_l = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
            cropped_img_l = cv.flip(cropped_img_l,1) # flip left ROI to appear like right ones
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_l_"+str(i+1) +"_0.png"),cropped_img_l)

            # Collect the centre coordinate of the ROI  
            tl[i-2] = [i+1,lm[i,0]-x_width,lm[i,1]-y_width,0]
                
    pd.DataFrame(tl).to_csv(os.path.join(current_dir,tl_dir,filename+".csv"),index=False)
      
        
def extract_ROI_from_lm(current_dir,filename,landmarks,image_size,dim=128,
                            img_dir="Images",save_dir="ROI LMs",tl_dir="ROI LM Top-Lefts",index=0):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(landmarks)
    
    # Define array to collect the centre coordinates, landmark number, 
    # and off-centre number for each coordinate
    tl = np.zeros((22,4))
    tl[:,0] = range(1,23)
    
    # All points will have an ROI where they are at the centre
    width = dim/2
    
    # Create ROI for points 1 to 11 (right hip)
    for i in range(11):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < width:
                lm[i,1] = width
            if int(lm[i,1]) > image_size[1] - width:
                lm[i,1] = image_size[1] - width
            if int(lm[i,0]) < width:
                lm[i,0] = width

            # Define and save cropped ROI
            cropped_img_r = img[int(lm[i,1]-width):int(lm[i,1]+width),
                                int(lm[i,0]-width):int(lm[i,0]+width)]
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_r_"+str(i+1)+"_" + str(index) +".png"),cropped_img_r)
            
#             fig = plt.figure(figsize=(20,6))
#             plt.imshow(cropped_img_r,cmap="gray")
#             plt.axis('off')
#             plt.show()

            # Collect the top-left coordinate of the ROI 
            tl[i] = [i+1,lm[i,0]-width,lm[i,1]-width,index]

    # Create ROI for point 12 to 22 (left hip)
    for i in range(11,22):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:   
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < width:
                lm[i,1] = width
            if int(lm[i,1]) > image_size[1] - width:
                lm[i,1] = image_size[1] - width
            if int(lm[i,0]) > image_size[0] - width:
                lm[i,0] = image_size[0] - width

            # Define and save cropped ROI
            cropped_img_l = img[int(lm[i,1]-width):int(lm[i,1]+width),
                                int(lm[i,0]-width):int(lm[i,0]+width)]
            cropped_img_l = cv.flip(cropped_img_l,1) # flip left ROI to appear like right ones
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_l_"+str(i+1)+"_" + str(index) +".png"),cropped_img_l)

#             fig = plt.figure(figsize=(20,6))
#             plt.imshow(cropped_img_l,cmap="gray")
#             plt.axis('off')
#             plt.show() 
            
            # Collect the centre coordinate of the ROI  
            tl[i] = [i+1,lm[i,0]-width,lm[i,1]-width,index]
                
    pd.DataFrame(tl).to_csv(os.path.join(current_dir,tl_dir,filename+".csv"),index=False)

    
def extract_ROI_from_pred_lm(current_dir,filename,landmarks,image_size,dim=128,
                            img_dir="Images",save_dir="ROI LMs",tl_dir="ROI LM Top-Lefts",index=0):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(landmarks)
    
    # Define array to collect the centre coordinates, landmark number, 
    # and off-centre number for each coordinate
    tl = np.zeros((18,4))
    tl[:,0] = [1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20]
    
    # All points will have an ROI where they are at the centre
    width = dim/2
    
    # Create ROI for points 1 to 11 (right hip)
    for i in range(9):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < width:
                lm[i,1] = width
            if int(lm[i,1]) > image_size[1] - width:
                lm[i,1] = image_size[1] - width
            if int(lm[i,0]) < width:
                lm[i,0] = width

            # Define and save cropped ROI
            cropped_img_r = img[int(lm[i,1]-width):int(lm[i,1]+width),
                                int(lm[i,0]-width):int(lm[i,0]+width)]
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_r_"+str(i+1)+"_" + str(index) +".png"),cropped_img_r)

            # Collect the top-left coordinate of the ROI 
            tl[i] = [i+1,lm[i,0]-width,lm[i,1]-width,index]

    # Create ROI for point 12 to 22 (left hip)
    for i in range(11,20):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:   
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < width:
                lm[i,1] = width
            if int(lm[i,1]) > image_size[1] - width:
                lm[i,1] = image_size[1] - width
            if int(lm[i,0]) > image_size[0] - width:
                lm[i,0] = image_size[0] - width

            # Define and save cropped ROI
            cropped_img_l = img[int(lm[i,1]-width):int(lm[i,1]+width),
                                int(lm[i,0]-width):int(lm[i,0]+width)]
            cropped_img_l = cv.flip(cropped_img_l,1) # flip left ROI to appear like right ones
            cv.imwrite(os.path.join(current_dir,save_dir,
                                    filename+"_l_"+str(i+1)+"_" + str(index) +".png"),cropped_img_l)
            
            tl[i-2] = [i+1,lm[i,0]-width,lm[i,1]-width,index]
                
    pd.DataFrame(tl).to_csv(os.path.join(current_dir,tl_dir,filename+".csv"),index=False)


# def extract_ROI_from_pred_lm(current_dir,filename,landmarks,image_size,dim=128,
#                             img_dir="Images",save_dir="ROI LMs",tl_dir="ROI LM Top-Lefts",index=0):
#     '''
#     Given the image and the ROI mask, the ROI section of the image is extracted and saved.
#     This same function is used to extract the ROI section of the femhead masks.
#     '''
#     # Open image to extract ROI from 
#     img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
#     lm = np.nan_to_num(landmarks)
    
#     # Define array to collect the centre coordinates, landmark number, 
#     # and off-centre number for each coordinate
#     tl = np.zeros((22,4))
#     tl[:,0] = range(1,23)
    
#     # All points will have an ROI where they are at the centre
#     width = dim/2
    
#     # Create ROI for points 1 to 11 (right hip)
#     for i in range(11):
#         if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
#             # Deal with exceptions where the point desired as the centre is too close to edge of image
#             if int(lm[i,1]) < width:
#                 lm[i,1] = width
#             if int(lm[i,1]) > image_size[1] - width:
#                 lm[i,1] = image_size[1] - width
#             if int(lm[i,0]) < width:
#                 lm[i,0] = width

#             # Define and save cropped ROI
#             cropped_img_r = img[int(lm[i,1]-width):int(lm[i,1]+width),
#                                 int(lm[i,0]-width):int(lm[i,0]+width)]
#             cv.imwrite(os.path.join(current_dir,save_dir,
#                                     filename+"_r_"+str(i+1)+"_" + str(index) +".png"),cropped_img_r)

#             # Collect the top-left coordinate of the ROI 
#             tl[i] = [i+1,lm[i,0]-width,lm[i,1]-width,index]

#     # Create ROI for point 12 to 22 (left hip)
#     for i in range(11,22):
#         if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:   
#             # Deal with exceptions where the point desired as the centre is too close to edge of image
#             if int(lm[i,1]) < width:
#                 lm[i,1] = width
#             if int(lm[i,1]) > image_size[1] - width:
#                 lm[i,1] = image_size[1] - width
#             if int(lm[i,0]) > image_size[0] - width:
#                 lm[i,0] = image_size[0] - width

#             # Define and save cropped ROI
#             cropped_img_l = img[int(lm[i,1]-width):int(lm[i,1]+width),
#                                 int(lm[i,0]-width):int(lm[i,0]+width)]
#             cropped_img_l = cv.flip(cropped_img_l,1) # flip left ROI to appear like right ones
#             cv.imwrite(os.path.join(current_dir,save_dir,
#                                     filename+"_l_"+str(i+1)+"_" + str(index) +".png"),cropped_img_l)
            
#             tl[i] = [i+1,lm[i,0]-width,lm[i,1]-width,index]
                
#     pd.DataFrame(tl).to_csv(os.path.join(current_dir,tl_dir,filename+".csv"),index=False)
    
    
def extract_ROI_for_femurs(current_dir,filename,landmarks,image_size,
                            img_dir="Images",save_dir="Test",index=0):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    
    if not os.path.exists(os.path.join(current_dir,save_dir)): os.makedirs(os.path.join(current_dir,save_dir)) 
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(landmarks)
    
    width = 100
    
    # Define and save cropped ROI
    
    if lm[10,:].all() != 0 and lm[9,:].all() != 0:
        if max(int(lm[10,1]),int(lm[9,1]))+width >= image_size[1]:
            max_r_y = image_size[1]-1
        else:
            max_r_y = max(int(lm[10,1]),int(lm[9,1]))+width
        if int(lm[10,0]-width) < 0:
            min_r_x = 0
        else:
            min_r_x = int(lm[10,0]-width)  
        min_r_y = (min(int(lm[10,1]),int(lm[9,1]))-width)
        cropped_img_r = img[min_r_y:max_r_y,
                            min_r_x:int(lm[9,0]+width)]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_r.png"),cropped_img_r)
    elif lm[10,:].all() != 0:
        if int(lm[10,1]+width) >= image_size[1]:
            max_r_y = image_size[1]-1
        else:
            max_r_y = int(lm[10,1]+width)       
        if int(lm[10,0]-width) < 0:
            min_r_x = 0
        else:
            min_r_x = int(lm[10,0]-width)
        min_r_y = int(lm[10,1]-width)               
        cropped_img_r = img[min_r_y:max_r_y,
                            min_r_x:int(lm[10,0]+2*width)]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_r.png"),cropped_img_r)
    elif lm[9,:].all() != 0:
        if int(lm[9,1]+width) >= image_size[1]:
            max_r_y = image_size[1] - 1
        else:
            max_r_y = int(lm[9,1]+width)      
        if int(lm[9,0]-2*width) < 0:
            min_r_x = 0
        else:
            min_r_x = int(lm[9,0]-2*width)               
        min_r_y = int(lm[9,1]-width)             
        cropped_img_r = img[min_r_y:max_r_y,
                            min_r_x:int(lm[9,0]+width)]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_r.png"),cropped_img_r)
    else:
        print(filename + " right")
        print(lm)
    
    if lm[21,:].all() != 0 and lm[20,:].all() != 0:
        if max(int(lm[21,1]),int(lm[20,1]))+width >= image_size[1]:
            max_l_y = image_size[1]-1
        else:
            max_l_y = max(int(lm[21,1]),int(lm[20,1]))+width
        if int(lm[21,0]+width) >= image_size[0]:
            max_l_x = image_size[0]-1
        else:
            max_l_x = int(lm[21,0]+width)
        min_l_y = (min(int(lm[21,1]),int(lm[20,1]))-width)
        min_l_x = int(lm[20,0]-width)
        cropped_img_l = img[min_l_y:max_l_y,
                            min_l_x:max_l_x]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_l.png"),cropped_img_l)
    elif lm[20,:].all() != 0:
        if int(lm[20,1]+width) >= image_size[1]:
            max_l_y = image_size[1]-1
        else:
            max_l_y = int(lm[20,1]+width)
        if int(lm[20,0]+2*width) >= image_size[0]:
            max_l_x = image_size[0]-1
        else:
            max_l_x = int(lm[20,0]+2*width)
        min_l_y = int(lm[20,1]-width)
        min_l_x = int(lm[20,0]-width)
        cropped_img_l = img[min_l_y:max_l_y,
                            min_l_x:max_l_x]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_l.png"),cropped_img_l)
    elif lm[21,:].all() != 0:
        if int(lm[21,1]+width) >= image_size[1]:
            max_l_y = image_size[1]-1
        else:
            max_l_y = int(lm[21,1]+width)
        if int(lm[21,0]+width) >= image_size[0]:
            max_l_x = image_size[0]-1
        else:
            max_l_x = int(lm[21,0]+width)
        min_l_y = int(lm[21,1]-width)
        min_l_x = int(lm[21,0]+2*width)
        cropped_img_r = img[min_l_y:max_l_y,
                            min_l_x:max_l_x]
        cv.imwrite(os.path.join(current_dir,save_dir,
                            filename+"_femur_l.png"),cropped_img_l)
    else:
        print(filename + " left")

    return min_r_x, min_r_y, min_l_x, min_l_y
    
            
def post_extract_ROI_from_lm(current_dir,filename,pred_lms,landmarks,image_size,
                             dim=128,img_dir="Images",save_dir="PB ROI LMs"):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''
    # Open image to extract ROI from 
    img = cv.imread(os.path.join(current_dir,img_dir,filename+".png"))
    
    lm = np.nan_to_num(pred_lms)
    
    op = [dim/2,dim/2]
    
    for i in range(11):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            y_width = op[1]
            x_width = op[0]
            if int(lm[i,1]) < y_width:
                lm[i,1] = y_width
            if int(lm[i,1]) > image_size[1] - y_width:
                lm[i,1] = image_size[1] - y_width
            if int(lm[i,0]) < x_width:
                lm[i,0] = x_width
                
            cropped_img_r = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
            cv.imwrite(os.path.join(current_dir,save_dir,filename+"_r_"+str(i+1)+".png"),cropped_img_r)

    for i in range(11,22):
        if int(lm[i,1]) != 0 and int(lm[i,0]) != 0:
            y_width = op[1]
            x_width = op[0]
            if int(lm[i,1]) < y_width:
                lm[i,1] = y_width
            if int(lm[i,1]) > image_size[1] - y_width:
                lm[i,1] = image_size[1] - y_width
            if int(lm[i,0]) > image_size[0] - (dim-x_width):
                lm[i,0] = image_size[0] - (dim-x_width)
            cropped_img_l = img[int(lm[i,1]-y_width):int(lm[i,1]+(dim-y_width)),
                                int(lm[i,0]-x_width):int(lm[i,0]+(dim-x_width))]
            cropped_img_l = cv.flip(cropped_img_l,1)
            cv.imwrite(os.path.join(current_dir,save_dir,filename+"_l_"+str(i+1) +".png"),cropped_img_l)
                

def resize_roi_lm(landmarks, contr, contl):
    '''
    Resize landmarks to fit axis of ROI (including the flipped left).
    '''
    
     # Alter the CSV values to match ROI
    landmarks[2:8,0] = landmarks[2:8,0] - np.ones((1,6))*np.min(contr[:,0])
    landmarks[2:8,1] = landmarks[2:8,1] - np.ones((1,6))*np.min(contr[:,1])
    landmarks[13:19,0] = -landmarks[13:19,0] + np.ones((1,6))*np.max(contl[:,0])
    landmarks[13:19,1] = landmarks[13:19,1] - np.ones((1,6))*np.min(contl[:,1])

    lm_r = landmarks[2:8,:] 
    lm_l = landmarks[13:19,:]
    
    return lm_r, lm_l

def resize_roi(lm, cont, left=False):
    '''
    Resize landmarks to fit axis of ROI (including the flipped left).
    '''
    
     # Alter the CSV values to match ROI
    if not left:
        lm[:,0] = lm[:,0] - np.ones((1,len(lm)))*np.min(cont[:,0])
    else:
        lm[:,0] = -lm[:,0] + np.ones((1,len(lm)))*np.max(cont[:,0])

    lm[:,1] = lm[:,1] - np.ones((1,len(lm)))*np.min(cont[:,1])
    
    return lm


def reverse_resize_roi_lm(lm, cont, left=False):
    '''
    Reverse of the above function: resize_roi_lm
    '''
    
     # Alter the CSV values to match ROI
    if not left:
        lm[:,0] = lm[:,0] + np.ones((1,len(lm)))*np.min(cont[:,0])
    else:
        lm[:,0] = -lm[:,0] + np.ones((1,len(lm)))*np.max(cont[:,0])
        
    lm[:,1] = lm[:,1] + np.ones((1,len(lm)))*np.min(cont[:,1])

    return lm

def roi_contour_dims(cont):
    '''
    Get the rectangular dimensions of a contour. Used for ROI mask contours.
    '''
                           
    x_dist = np.max(cont[:,0]) - np.min(cont[:,0]) + 1
    y_dist = np.max(cont[:,1]) - np.min(cont[:,1]) + 1
                           
    return [x_dist,y_dist]