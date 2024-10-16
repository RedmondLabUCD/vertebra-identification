#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import itertools
import math
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shutil
import random
import seaborn as sns
import scipy
import pyreadstat
from pydicom import dcmread
from utils.roi_functions import create_ROI_mask, extract_ROI, resize_roi_lm, extract_ROI_from_lm, extract_ROI_from_lm_aug, extract_ROI_from_lm_aug2
from utils.landmark_prep import prep_landmarks
from utils.heatmaps import create_hm, create_roi_hm, create_hm_w_back, pb_create_hm, pb_create_hm_aug
from utils.feature_extraction import extract_image_size
from utils.data_prep import aug_femhead_data, augment_lm_data, doughnut_contrast, size_by_black
    

def prep_data():

    df = pd.DataFrame(columns = ['image','id','group','T4x','T4y','T5x','T5y','T6x','T6y','T7x','T7y','T8x','T8y','T9x','T9y',
                             'T10x','T10y','T11x','T11y','T12x','T12y','L1x','L1y','L2x','L2y','L3x','L3y','L4x','L4y'])

    filenames = [os.path.normpath(file).split(os.path.sep)[-1][:-4]
                     for file in glob('//data/scratch/r094879/data/images/*.dcm')]

    df['image'] = filenames

    # Read the Excel file
    mappings_file = '//data/scratch/r094879/data/annotations/mappings.csv' 
    df_x = pd.read_csv(mappings_file)

    # Loop through each row in the Excel file and process
    for index, row in df_x.iterrows():
        create_data_file(row,df)

    df2 = df.replace('', np.nan, regex=True)
    df2.to_csv('//data/scratch/r094879/data/annotations/annotations.csv',index=False)


def create_data_file(row,df):

    vertebra_list = ['T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4']

    # Extract ID and group
    id = row['id']
    group = row['group']

    # Open corresponding SPSS file
    spss_file = f"/data/scratch/r094879/data/annotations/{group}_mergedABQQM_Ling_20140128.sav"
    df_spss, meta = pyreadstat.read_sav(spss_file)

    # Find the row in the SPSS file corresponding to the ID
    spss_row = df_spss[df_spss['ergoid'] == id]
    if spss_row.empty:
        print(f"No matching row found for ID {id} in group {group}")
        return

    # For each vertebra, get image name and x, y coordinates
    for vertebra in vertebra_list:

        x = spss_row['e1_17971.'+str(vertebra)].values[0]
        y = spss_row['e1_17972.'+str(vertebra)].values[0]
        img = spss_row['e1_17962.'+str(vertebra)].values[0]

        df.loc[df["image"]==img,str(vertebra)+'x'] = x
        df.loc[df["image"]==img,str(vertebra)+'y'] = y
        df.loc[df["image"]==img,'id'] = id
        df.loc[df["image"]==img,'group'] = group

        # Skip if any value is missing
        if pd.isna(img) or pd.isna(x) or pd.isna(y):
            print(f"Missing data for vertebra {vertebra} in ID {id}. Skipping...")
            continue
        df.to_csv('//data/scratch/r094879/data/annotations/annotations.csv',index=False)


def plot_images_with_points():

    csv_file = '//data/scratch/r094879/data/annotations/annotations.csv' 
    df_x = pd.read_csv(csv_file)

    dicom_dir = '//data/scratch/r094879/data/images'
    output_dir = '//data/scratch/r094879/data/images_with_points'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        image_name = row['image']  # Get the DICOM image name from the 'image' column
        dicom_file_path = os.path.join(dicom_dir, image_name)

        # Read the DICOM file
        dicom_image = pydicom.dcmread(dicom_file_path)

        # Extract pixel array from the DICOM file
        pixel_array = dicom_image.pixel_array

        # Get the x and y values for each vertebra
        x_values = row.iloc[3:29:2].values 
        y_values = row.iloc[4:29:2].values

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.array(list(zip(x_values, y_values)))
        xy_pairs = xy_pairs[~np.isnan(xy_pairs).any(axis=1)]

        # Split back into x and y arrays after removing NaN pairs
        if len(xy_pairs) > 0:  # Proceed only if we have valid points to plot
            x_values, y_values = xy_pairs[:, 0], xy_pairs[:, 1]
    
            # Plot the DICOM image
            plt.imshow(pixel_array, cmap='gray')
    
            # Plot the x and y points on the image
            plt.scatter(x_values, y_values, c='red', s=40, marker='o')
    
            # Save the image as a PNG file
            output_file_name = f"{image_name}_annotated.png"
            output_file_path = os.path.join(output_dir, output_file_name)
            plt.savefig(output_file_path)
    
            # Clear the plot for the next iteration
            plt.clf()
    
            print(f"Saved {output_file_path}")
        else:
            print(f"No valid points to plot for {dicom_image_name}")

    print("All images have been processed and saved as PNG files.")
            

def create_dataset():
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Remove all of previous dataset so no replicas
    save_dir = os.path.join(root,"Dataset")
    
    if not os.path.exists(os.path.join(save_dir,"Images AUG")):
        os.makedirs(os.path.join(save_dir,"Images AUG"))
    if not os.path.exists(os.path.join(save_dir,"ROI")):
        os.makedirs(os.path.join(save_dir,"ROI"))

    csv_filename_set = {os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                        for file in glob(os.path.join(save_dir,fold_name,"CSVs","*.csv"))}

    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                 for file in glob(os.path.join(save_dir,fold_name,"Images","*.png"))]
    
    current_dir = os.path.join(save_dir,fold_name)
#         aug_femhead_data(current_dir)
#         augment_lm_data(current_dir)

#         aug_csv_filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
#                             for file in glob(os.path.join(save_dir,fold_name,"CSVs AUG","*.csv"))]

    # Go through each image and create a LM mask
    for filename in filenames:

        # Create ROI Mask and extract ROI of images and femhead masks
        image_size = extract_image_size(os.path.join(current_dir,"Images",filename+".png"))
#             roi_mask = create_ROI_mask(current_dir,filename)
        extract_ROI(current_dir,filename)
        extract_ROI(current_dir,filename,img_dir="FemHead Masks",save_dir="ROI FemHead Masks")

#             if filename in csv_filename_set:
#                 landmarks, image_size = prep_landmarks(filename,os.path.join(current_dir,"CSVs"))
#                 hm = create_hm_w_back(landmarks,image_size,new_dim=256.0,size=15)
#                 np.save(os.path.join(current_dir,"LM Heatmaps Back",filename),hm)
            
# #                 hm = create_hm_w_back(landmarks,image_size,new_dim=256.0,size=30)
# #                 np.save(os.path.join(current_dir,"LM Heatmaps30",filename),hm)

#                 # Create ROI Mask, extract ROI of images and femhead masks, and create ROI heatmaps
            
#                 # Step 2: Basis
#                 extract_ROI_from_lm(current_dir,filename,landmarks,image_size,dim=128,
#                             save_dir="ROI LMs",tl_dir="ROI LM Top-Lefts")
#                 pb_create_hm(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps",
#                      tl_dir="ROI LM Top-Lefts",dim=128,size=5)
            
#                 # Step 2: Data augmentation
#                 extract_ROI_from_lm_aug(current_dir,filename,landmarks,image_size,dim=128,
#                             save_dir="ROI LMs AUG",tl_dir="ROI LM Top-Lefts AUG")
#                 pb_create_hm_aug(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps AUG",
#                      tl_dir="ROI LM Top-Lefts AUG",dim=128,size=5)
            
#                 # Step 2: Double size
#                 extract_ROI_from_lm(current_dir,filename,landmarks,image_size,dim=256,
#                             save_dir="ROI LMs Double",tl_dir="ROI LM Top-Lefts Double")
#                 pb_create_hm(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps Double",
#                      tl_dir="ROI LM Top-Lefts Double",dim=256,size=5)
            
#                 # Step 2: Data Augmentation 2
#                 extract_ROI_from_lm_aug2(current_dir,filename,landmarks,image_size,dim=128,
#                             save_dir="ROI LMs AUG2",tl_dir="ROI LM Top-Lefts AUG2")
#                 pb_create_hm(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps AUG2",
#                      tl_dir="ROI LM Top-Lefts AUG2",dim=128,size=5)
            
#                 # Step 2: Data Augmentation 2 + Double Size
#                 extract_ROI_from_lm_aug2(current_dir,filename,landmarks,image_size,dim=256,
#                             save_dir="ROI LMs Double AUG2",tl_dir="ROI LM Top-Lefts Double AUG2")
#                 pb_create_hm(current_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps Double AUG2",
#                      tl_dir="ROI LM Top-Lefts Double AUG2",dim=256,size=5)
            
#         for filename in aug_csv_filenames:   
#             landmarks = pd.read_csv(os.path.join(save_dir,fold_name,"CSVs AUG",filename+'.csv'))
#             landmarks = np.asarray(landmarks).astype(float).reshape((-1,2))
#             image_size = landmarks[0,:]
#             landmarks = landmarks[1:,:]

#             hm = create_hm(landmarks,image_size,new_dim=256.0,size=15)
#             np.save(os.path.join(current_dir,"LM Heatmaps AUG",filename),hm)


def split_data(k=10):
    '''
    Splits up the pre-processed images, femoral head masks, and landmark CSV files 
    into 10 folds. Since there might not be femoral head masks or landmark files for 
    each image, the data is split by the smallest data type first and the remainder 
    in each of the other types is split up after.
    
    '''
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Remove all of previous dataset so no replicas
    save_dir = os.path.join(root,"Dataset")
    
    images = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0] 
              for file in glob(os.path.join(save_dir,"Images","*.png"))]
    masks = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0] 
             for file in glob(os.path.join(save_dir,"FemHead Masks","*.png"))]
    csvs = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0] 
            for file in glob(os.path.join(save_dir,"CSVs","*.csv"))]
    
    images_set = set(images)
    masks_set = set(masks)
    csvs_set = set(csvs)
    
    data = sorted([images,masks,csvs],key=len)

    # Split data into datasets based on smallest type first
    for i in range(len(data)):
        split = 1/k
        split_size = int(split * len(data[i]))
        remain_size = len(data[i]) - k*split_size
        data[i] = shuffle(data[i])
        fold_sizes = np.ones(k)*split_size
        
        for r in range(remain_size):
            fold_sizes[r] = fold_sizes[r]+1
        
        start = 0
        # Create the folds
        for n in range(k):
            fold = data[i][int(start):int(start+fold_sizes[n])]
            
            fold_name = "Fold " + str(n+1)
            if not os.path.exists(os.path.join(save_dir,fold_name)):
                os.makedirs(os.path.join(save_dir,fold_name))
                os.makedirs(os.path.join(save_dir,fold_name,"Images"))
                os.makedirs(os.path.join(save_dir,fold_name,"FemHead Masks"))
                os.makedirs(os.path.join(save_dir,fold_name,"CSVs"))

            for file in fold:
                if os.path.exists(os.path.join(save_dir,"FemHead Masks",file+".png")):
                    shutil.copy(os.path.join(save_dir,"FemHead Masks",file+".png"),
                                os.path.join(save_dir,fold_name,"FemHead Masks",file+".png"))
                    masks.remove(file)
                if os.path.exists(os.path.join(save_dir,"Images",file+".png")):
                    shutil.copy(os.path.join(save_dir,"Images",file+".png"),
                                os.path.join(save_dir,fold_name,"Images",file+".png"))
                    images.remove(file)
                if os.path.exists(os.path.join(save_dir,"CSVs",file+".csv")):
                    shutil.copy(os.path.join(save_dir,"CSVs",file+".csv"),
                                os.path.join(save_dir,fold_name,"CSVs",file+".csv"))
                    csvs.remove(file)
                    
            start = start + fold_sizes[n]
