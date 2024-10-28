#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
from PIL import Image
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
from utils.heatmaps import create_hm
from utils.feature_extraction import extract_image_size
    

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
    variable_names = {'RSI_1':'e1','RSI_2':'e2','RSI_3':'e3','RSI_4':'e4','RSII_2':'e4','RSIII_1':'ej'}

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

        x = spss_row[variable_names[group]+'_17971.'+str(vertebra)].values[0]
        y = spss_row[variable_names[group]+'_17972.'+str(vertebra)].values[0]
        img = spss_row[variable_names[group]+'_17962.'+str(vertebra)].values[0]

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
    df = pd.read_csv(csv_file)

    dicom_dir = '//data/scratch/r094879/data/images'
    output_dir = '//data/scratch/r094879/data/images_with_points'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        image_name = row['image']  # Get the DICOM image name from the 'image' column
        dicom_file_path = os.path.join(dicom_dir, image_name+'.dcm')

        # Read the DICOM file
        dicom_image = dcmread(dicom_file_path)

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
            print(f"No valid points to plot for {image_name}")

    print("All images have been processed and saved as PNG files.")
            

def create_dataset():

    csv_file = '//data/scratch/r094879/data/annotations/annotations.csv' 
    df = pd.read_csv(csv_file)

    dicom_dir = '//data/scratch/r094879/data/images'
    output_dir = '//data/scratch/r094879/data/heatmaps'
    output_dir_2 = '//data/scratch/r094879/data/imgs'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir_2):
        os.makedirs(output_dir_2)

    for index, row in df.iterrows():
        image_name = row['image']  # Get the DICOM image name from the 'image' column
        dicom_file_path = os.path.join(dicom_dir, image_name+'.dcm')

        # Read the DICOM file
        dicom_image = dcmread(dicom_file_path)

        # Extract pixel array from the DICOM file
        pixel_array = dicom_image.pixel_array

        # Get the x and y values for each vertebra
        x_values = row.iloc[3:29:2].values 
        y_values = row.iloc[4:29:2].values

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.array(list(zip(x_values, y_values)))

        hm = create_hm(xy_pairs,pixel_array.shape,new_dim=256.0,size=15)
        np.save(os.path.join(output_dir,image_name),hm)

        image = Image.fromarray(pixel_array)
        image.save(os.path.join(output_dir_2,image_name+'.png'))


def view_heatmaps():
    
    file_path = '//data/scratch/r094879/data/heatmaps/1.2.392.200036.9125.9.0.68100090.749932288.3927965275.npy'

    data = np.load(file_path)

    save_path = '//data/scratch/r094879/data/data_check'

    # Ensure the output directories exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Initialize an array to store the sums of each slice
    cumulative_sum = np.zeros(data.shape[0:2], dtype=data.dtype)
    print(data.shape)
    print(data.shape[0:2])
    print(data.shape[2])
    
    # Iterate through each slice in the 3D array
    for i in range(data.shape[2]):
        slice_data = data[i]
        
        # Plot the slice
        plt.imshow(slice_data, cmap='gray')
        
        # Save each slice as a .png
        plt.savefig(f"//data/scratch/r094879/data/data_check/slice_{i}.png")
        plt.close()  # Close the plot to free memory
        
        # Calculate the sum of the slice
        cumulative_sum += slice_data
    
    # Plot and save the cumulative sum as a .png
    plt.imshow(cumulative_sum, cmap='gray')
    plt.title("Cumulative Sum of All Slices")
    plt.savefig("//data/scratch/r094879/data/data_check/cumulative_sum.png")
    plt.close()


