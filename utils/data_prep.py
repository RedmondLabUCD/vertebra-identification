import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2
from utils.datasets import HipSegDataset
from numpy import fliplr
import math
from tqdm import tqdm
from PIL import Image,ImageEnhance
from scipy import ndimage
from skimage import io
from utils import datasets
from utils.landmark_prep import prep_landmarks


def mean_and_std(index, data_dir, params, AUG):
    '''
    Calculates mean and standard deviation of images to be 
    used in image normalization.
    Inspired by: towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    '''
    Dataset = getattr(datasets,"HipSegDataset")
    
    # Define basic transform (resize and make tensor)
    transform = transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                    transforms.ToTensor()])
    
    # Set up transforms for targets
    if "Masks" in params.target_dir:
        target_transform = transforms.Compose([transforms.Grayscale(),
                                               transforms.Resize((params.input_size,params.input_size)),
                                               transforms.ToTensor()
                                               ])
    else:
        target_transform = transforms.ToTensor()
        
    # Define validation dataset
    if index+1 > 10:
        val_index = 1
    else:
        val_index = index + 1

    # Define and load training dataset
    train_data = []
    for i in range(1,11):
        if i != index and i != val_index:
            if AUG:
                fold_data = Dataset(data_dir,i,params.image_dir+" AUG",params.target_dir+" AUG",
                                    target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
            else:
                fold_data = Dataset(data_dir,i,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                    input_tf=transform,output_tf=target_transform)
            train_data = ConcatDataset([train_data, fold_data])

    loader = DataLoader(train_data,batch_size=params.batch_size,shuffle=False)
    
    # Calculate mean and std for each batch 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    # Get mean and std across the batches
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    mean2 = [1.*mean[0],1.*mean[1],1.*mean[2]]
    
    std = std.cpu().numpy()
    std2 = [1.*std[0],1.*std[1],1.*std[2]]

    return mean2, std2


def final_mean_and_std(data_dir, params, AUG):
    '''
    Calculates mean and standard deviation of images to be 
    used in image normalization.
    Inspired by: towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    '''
    Dataset = getattr(datasets,"HipSegDataset")
    
    # Define basic transform (resize and make tensor)
    transform = transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                    transforms.ToTensor()])
    
    # Set up transforms for targets
    if "Masks" in params.target_dir:
        target_transform = transforms.Compose([transforms.Grayscale(),
                                               transforms.Resize((params.input_size,params.input_size)),
                                               transforms.ToTensor()
                                               ])
    else:
        target_transform = transforms.ToTensor()

    # Define and load training dataset
    train_data = []
    for i in range(1,10):
        if AUG:
            fold_data = Dataset(data_dir,i,params.image_dir+" AUG",params.target_dir+" AUG",
                                target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
        else:
            fold_data = Dataset(data_dir,i,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                input_tf=transform,output_tf=target_transform)
        train_data = ConcatDataset([train_data, fold_data])

    loader = DataLoader(train_data,batch_size=params.batch_size,shuffle=False)
    
    # Calculate mean and std for each batch 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    # Get mean and std across the batches
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    mean2 = [1.*mean[0],1.*mean[1],1.*mean[2]]
    
    std = std.cpu().numpy()
    std2 = [1.*std[0],1.*std[1],1.*std[2]]

    return mean2, std2


def apply_clahe(data_dir):
    '''
    Applies CLAHE to data to increase contrast.
    '''

    images = glob(os.path.join(data_dir,"*"))
    if not os.path.exists(data_dir+" CLAHE"): os.makedirs(data_dir+" CLAHE")
    
    for image in images:
        image_name = image.split("\\")[-1].split(".")[0]
        
        img = cv2.imread(image, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_img = clahe.apply(img)
        
        cv2.imwrite(os.path.join(data_dir+' CLAHE',image_name+'.png'),cl_img)
        

# def apply_clahe(data_dir):
#     '''
#     Applies CLAHE to data to increase contrast.
#     '''

#     images = glob(os.path.join(data_dir,"*.png"))
#     if not os.path.exists(data_dir+" CLAHE"): os.makedirs(data_dir+" CLAHE")
    
#     for image in images:
#         image_name = image.split("\\")[-1].split(".")[0]
#         img = cv2.imread(image, 0)
#         denoised = cv2.fastNlMeansDenoising(img,None,h=3,templateWindowSize=3,searchWindowSize=21)
#         clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2,2))
#         cl_img = clahe.apply(denoised)
# #         cl_img = cv2.fastNlMeansDenoising(cl_img,None,h=3,templateWindowSize=7,searchWindowSize=21)
        
#         cv2.imwrite(os.path.join(data_dir+' CLAHE',image_name+'.png'),cl_img)
        
        
# def apply_clahe_2(data_dir):
#     '''
#     Applies CLAHE to data to increase contrast.
#     '''

#     images = glob(os.path.join(data_dir,"*.png"))
#     if not os.path.exists(data_dir+" CLAHE2"): os.makedirs(data_dir+" CLAHE2")
    
#     for image in images:
#         image_name = image.split("\\")[-1].split(".")[0]
#         img = cv2.imread(image, 0)
#         denoised = cv2.bilateralFilter(img,9,75,75)
#         clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2,2))
#         cl_img = clahe.apply(denoised)
# #         cl_img = cv2.fastNlMeansDenoising(cl_img,None,h=3,templateWindowSize=7,searchWindowSize=21)
        
#         cv2.imwrite(os.path.join(data_dir+' CLAHE2',image_name+'.png'),cl_img)
        
        
def aug_femhead_data(data_dir,image_dir="Images",tar_dir="FemHead Masks"):
    '''
    Applies CLAHE, contrast reduction, and rotation to femhead data (i.e. repeat of ME Project)
    '''
    
    images = glob(os.path.join(data_dir,image_dir,"*.png"))
    masks = glob(os.path.join(data_dir,tar_dir,"*.png"))
    if not os.path.exists(os.path.join(data_dir,image_dir+" AUG")): 
        os.makedirs(os.path.join(data_dir,image_dir+" AUG"))
    if not os.path.exists(os.path.join(data_dir,tar_dir+" AUG")): 
        os.makedirs(os.path.join(data_dir,tar_dir+" AUG"))
        
    images.sort()
    masks.sort()        
        
    for image, mask in tqdm(zip(images, masks), total=len(images)):
        filename = image.split("\\")[-1].split(".")[0]

        # ________ORIGINALS________ 

        img = io.imread(image)
        tar_mask = io.imread(mask)
        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'.png'),img)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'.png'),tar_mask)

        # ________HORIZONTAL FLIP________

        flippedlr_image = fliplr(img) #nd array
        flippedlr_image = Image.fromarray(flippedlr_image) #ndarray to image
        flippedlr_filename = filename+'_flippedlr.png'
        flippedlr_image.save(os.path.join(data_dir,image_dir+" AUG",flippedlr_filename))

        flippedlr_tar_mask = fliplr(tar_mask) #nd array
        flippedlr_tar_mask = Image.fromarray(flippedlr_tar_mask) #ndarray to image
        flippedlr_tar_mask.save(os.path.join(data_dir,tar_dir+" AUG",flippedlr_filename))
        
        # ________ROTATION________ 

        # Find current angle
        img = cv2.imread(image) 
        msk = cv2.imread(mask) 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        
        angles = []

        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))
                if (abs(angle) < 45):
                    angles.append(angle)

        if angles == []:
            mean_angle = 0.0;
        else:
            mean_angle = np.mean(angles)

        # Rotate to 0 degrees
        if (mean_angle != 0): 
            img_rotated = ndimage.rotate(img,-mean_angle,reshape=False)
            mask_rotated = ndimage.rotate(msk,-mean_angle,reshape=False)

            cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_0.png'), img_rotated)
            cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_0.png'), mask_rotated)

            img = img_rotated
            msk = mask_rotated
        
        # Rotate to +7 degrees
        img_rotated_plus7 = ndimage.rotate(img,45,reshape=False)
        mask_rotated_plus7 = ndimage.rotate(msk,45,reshape=False)

        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_plus45.png'), img_rotated_plus7)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_plus45.png'),mask_rotated_plus7)

        # Rotate to -7 degrees
        img_rotated_minus7 = ndimage.rotate(img,-45,reshape=False)
        mask_rotated_minus7 = ndimage.rotate(msk,-45,reshape=False)

        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_minus45.png'),img_rotated_minus7)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_minus45.png'),mask_rotated_minus7)

        # ________ALTER CONTRAST________ 

        # Increase contrast with CLAHE
        img = cv2.imread(image, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_img = clahe.apply(img)
        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_clahe.png'),cl_img)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_clahe.png'),tar_mask)

        # Lower Contrast with Contrast enhancer
        img = Image.open(image)
        enhancer = ImageEnhance.Contrast(img)
        low_img = enhancer.enhance(0.5)
        low_img.save(os.path.join(data_dir,image_dir+" AUG",filename+'_low.png'))
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_low.png'),tar_mask)

        
def aug_femhead_roi_data(data_dir,image_dir="Images",tar_dir="FemHead Masks"):
    '''
    Applies CLAHE, contrast reduction, and rotation to femhead data (i.e. repeat of ME Project)
    '''
    
    images = glob(os.path.join(data_dir,image_dir,"*.png"))
    masks = glob(os.path.join(data_dir,tar_dir,"*.png"))
    if not os.path.exists(os.path.join(data_dir,image_dir+" AUG")): 
        os.makedirs(os.path.join(data_dir,image_dir+" AUG"))
    if not os.path.exists(os.path.join(data_dir,tar_dir+" AUG")): 
        os.makedirs(os.path.join(data_dir,tar_dir+" AUG"))
        
    images.sort()
    masks.sort()        
        
    for image, mask in tqdm(zip(images, masks), total=len(images)):
        filename = image.split("\\")[-1].split(".")[0]

        # ________ORIGINALS________ 

        img = io.imread(image)
        tar_mask = io.imread(mask)
        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'.png'),img)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'.png'),tar_mask)
        
        # ________ROTATION________ 

        # Find current angle
        img = cv2.imread(image) 
        msk = cv2.imread(mask) 
        
        # Rotate to +7 degrees
        img_rotated_plus45 = ndimage.rotate(img,45,reshape=False)
        mask_rotated_plus45 = ndimage.rotate(msk,45,reshape=False)

        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_plus45.png'), img_rotated_plus45)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_plus45.png'),mask_rotated_plus45)

        # Rotate to -7 degrees
        img_rotated_minus45 = ndimage.rotate(img,-45,reshape=False)
        mask_rotated_minus45 = ndimage.rotate(msk,-45,reshape=False)

        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_minus45.png'),img_rotated_minus45)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_minus45.png'),mask_rotated_minus45)

        # ________ALTER CONTRAST________ 

        # Increase contrast with CLAHE
        img = cv2.imread(image, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_img = clahe.apply(img)
        cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_clahe.png'),cl_img)
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_clahe.png'),tar_mask)

        # Lower Contrast with Contrast enhancer
        img = Image.open(image)
        enhancer = ImageEnhance.Contrast(img)
        low_img = enhancer.enhance(0.5)
        low_img.save(os.path.join(data_dir,image_dir+" AUG",filename+'_low.png'))
        cv2.imwrite(os.path.join(data_dir,tar_dir+" AUG",filename+'_low.png'),tar_mask)
        
        
def augment_lm_data(data_dir,image_dir="Images",tar_dir="CSVs"):
    
    images = glob(os.path.join(data_dir,image_dir,"*.png"))
    masks = glob(os.path.join(data_dir,tar_dir,"*.csv"))
    if not os.path.exists(os.path.join(data_dir,tar_dir+" AUG")): 
        os.makedirs(os.path.join(data_dir,tar_dir+" AUG"))

    images.sort()
    masks.sort() 
    
    for image, mask in tqdm(zip(images, masks),total=len(masks)):
        filename = mask.split("\\")[-1].split(".")[0]

        # ________ORIGINALS________ 

        # prep target annotations arrays 
        landmarks, image_size = prep_landmarks(filename,os.path.join(data_dir,tar_dir))
#         image_size = np.asarray(output.iloc[0,4:6]).astype(np.float)
        image_size_2 = image_size.reshape((1,2))
#         landmarks = np.asarray(output.iloc[:,1:3])
#         landmarks = landmarks.astype(np.float)
        landOriginal = landmarks.copy()
        landOriginal = np.append(image_size_2,landOriginal,axis=0)
        pd.DataFrame(landOriginal).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'.csv'),index=False)

        # ________HORIZONTAL FLIP________ 

        landmarks_flippedlr = landmarks.copy()
        for row in landmarks_flippedlr:
            row[0] = image_size[0] - row[0]
        landmarks_flippedlr = np.append(image_size_2,landmarks_flippedlr,axis=0)
        pd.DataFrame(landmarks_flippedlr).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_flippedlr.csv'),
                                                 index = False)

        # ________ROTATION________ 

        img = cv2.imread(image) 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        
        angles = []

        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))
                if (abs(angle) < 45):
                    angles.append(angle)

        if angles == []:
            mean_angle = 0.0
        else:
            mean_angle = np.mean(angles)

        if (mean_angle != 0): 
            img_rotated = ndimage.rotate(img,-mean_angle,reshape=False)
            cv2.imwrite(os.path.join(data_dir,image_dir+" AUG",filename+'_0.png'), img_rotated)
            
            rotated = rotate_csv(landmarks,image_size,-mean_angle)
            rotated = np.append(image_size_2,rotated,axis=0)
            pd.DataFrame(rotated).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_0.csv'),index=False)
            landmarks = rotated[1:,:]

        rotated = rotate_csv(landmarks,image_size,45)
        rotated = np.append(image_size_2,rotated,axis=0)
        pd.DataFrame(rotated).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_plus45.csv'),index=False)

        rotated = rotate_csv(landmarks,image_size,-45)
        rotated = np.append(image_size_2,rotated,axis=0)
        pd.DataFrame(rotated).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_minus45.csv'),index=False)

        # ________ALTER CONTRAST________ 

        # Increase contrast with CLAHE
        pd.DataFrame(landOriginal).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_clahe.csv'),
                                          index=False)

        # Lower Contrast with Contrast enhancer
        pd.DataFrame(landOriginal).to_csv(os.path.join(data_dir,tar_dir+' AUG',filename+'_low.csv'),index=False)
        

def rotate_csv(landmarks,image_size,angle):
    # angle = 180-angle
    center = image_size/2.0
    np.reshape(center,(2,1))
    c, s = math.cos(angle/180.*math.pi), math.sin(angle/180.*math.pi)
    R = np.array(((c,-s),(s,c)))

    rotated = []
    for row in landmarks:
        np.reshape(row,(2,1))
        relative = row-center
        A = np.dot(relative,R)
        rotated_coord = A+center
        np.reshape(rotated_coord,(1,2))
        rotated.append(rotated_coord)

    np.reshape(rotated,(22,2))

    return rotated 

    
# def fem_head_contrast(data_dir):
#     '''
#     Applies CLAHE to data to increase contrast.
#     '''

#     images = glob(os.path.join(data_dir,"Images","*.png"))
#     masks = glob(os.path.join(data_dir,"FemHead Masks","*.png"))
    
#     images.sort()
#     masks.sort()     
    
#     image_stats = pd.DataFrame(columns=['Image','Contrast'])
#     index = 0
    
#     for image, mask in tqdm(zip(images, masks), total=len(images)):
#         image_name = image.split("\\")[-1].split(".")[0]
#         image_stats.loc[index,'Image'] = image_name
        
#         # read in image and mask
#         img = cv2.imread(image, cv2.IMREAD_COLOR)
#         mask = cv2.imread(mask, cv2.IMREAD_COLOR)
        
#         # erode mask
#         kernel = np.ones((10,10),np.uint8)
#         mask_erode = cv2.erode(mask, kernel, iterations=1)
        
#         # dilate mask
#         mask_dilate = cv2.dilate(mask, kernel, iterations=1)
        
#         # extract parts just inside and outside femhead based on erosion/dilation masks
#         result_out = img.copy()
#         result_in = img.copy()
#         result_out[np.where(mask_dilate - mask == (0,0,0))] = 0
#         result_in[np.where(mask - mask_erode == (0,0,0))] = 0
        
#         # calculate avg pixel value in and out of femhead
#         mean_out = result_out[np.where(result_out > (0,0,0))].mean()
#         mean_in = result_in[np.where(result_in > (0,0,0))].mean()
        
#         # define local contrast as the difference between in and out of femhead values
#         contrast = mean_in-mean_out
#         image_stats.loc[index,'Contrast'] = contrast
        
#         index = index + 1
        
#     # save in csv file
#     image_stats.to_csv(os.path.join(data_dir,'contrast_dice_stats.csv'),index=False)
    
    
def doughnut_contrast(image_name):
    '''
    Applies CLAHE to data to increase contrast. REPLACE ABOVE SO IT WORKS FOR ONLY ONE IMAGE.
    '''
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Remove all of previous dataset so no replicas
    data_dir = os.path.join(root,"Dataset")
        
    # read in image and mask
    img = cv2.imread(os.path.join(data_dir,"Images",image_name+".png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(data_dir,"FemHead Masks",image_name+".png"), cv2.IMREAD_COLOR)
    
    cv2.imshow("hi", img)
    cv2.waitKey(0)

    # erode mask
    kernel = np.ones((10,10),np.uint8)
    mask_erode = cv2.erode(mask, kernel, iterations=1)
    
    cv2.imshow("hi", mask_erode)
    cv2.waitKey(0)

    # dilate mask
    mask_dilate = cv2.dilate(mask, kernel, iterations=1)
    
    cv2.imshow("hi", mask_dilate)
    cv2.waitKey(0)

    # extract parts just inside and outside femhead based on erosion/dilation masks
    result_out = img.copy()
    result_in = img.copy()
    result_out[np.where(mask_dilate - mask == (0,0,0))] = 0
    result_in[np.where(mask - mask_erode == (0,0,0))] = 0
    
    cv2.imshow("hi", result_out)
    cv2.waitKey(0)
    cv2.imshow("hi", result_in)
    cv2.waitKey(0)

    # calculate avg pixel value in and out of femhead
    mean_out = result_out[np.where(result_out > (0,0,0))].mean()
    mean_in = result_in[np.where(result_in > (0,0,0))].mean()

    # define local contrast as the difference between in and out of femhead values
    contrast = mean_in-mean_out
    
    return contrast
        
    
def rms_contrast(image_name,img_dir="Images"):
    '''
    Applies CLAHE to data to increase contrast. REPLACE ABOVE SO IT WORKS FOR ONLY ONE IMAGE.
    '''
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Remove all of previous dataset so no replicas
    data_dir = os.path.join(root,"Dataset")
        
    # read in image and mask
    if os.path.exists(os.path.join(data_dir,img_dir,image_name+".png")):
        img = cv2.imread(os.path.join(data_dir,img_dir,image_name+".png"), 0)
    else:
        img = cv2.imread(os.path.join(data_dir,"FINAL TEST",img_dir,image_name+".png"), 0)
    
#     contrast = img[img>3].std()
   
#     new = np.empty((0))
#     for row in img:
#         no_back = np.asarray([x for x in row if x > 3])
#         new = np.append(new,no_back,axis=0)

    contrast = np.std(img)
        
    return contrast

    
def size_by_black(image_name):
    '''
    Applies CLAHE to data to increase contrast. REPLACE ABOVE SO IT WORKS FOR ONLY ONE IMAGE.
    '''
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Remove all of previous dataset so no replicas
    data_dir = os.path.join(root,"Dataset")
        
    # read in image and mask
    img = cv2.imread(os.path.join(data_dir,"Images",image_name+".png"), cv2.IMREAD_COLOR)
    
    number_black = np.sum(img == 0)
    number_other = np.sum(img != 0)
    
    return number_black/(number_black+number_other)
        