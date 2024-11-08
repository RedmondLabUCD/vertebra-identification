#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
from PIL import Image
import PIL
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
import cv2 as cv
from pydicom.pixel_data_handlers.util import apply_voi_lut
# import h5py
    

def prep_data():

    # df = pd.DataFrame(columns = ['image','id','group','T4x','T4y','T5x','T5y','T6x','T6y','T7x','T7y','T8x','T8y','T9x','T9y',
    #                          'T10x','T10y','T11x','T11y','T12x','T12y','L1x','L1y','L2x','L2y','L3x','L3y','L4x','L4y'])

    # filenames = [os.path.normpath(file).split(os.path.sep)[-1][:-4]
    #                  for file in glob('//data/scratch/r094879/data/images/*.dcm')]

    # df['image'] = filenames

    # # Read the Excel file
    # mappings_file = '//data/scratch/r094879/data/annotations/mappings.csv' 
    # df_x = pd.read_csv(mappings_file)

    # # Loop through each row in the Excel file and process
    # for index, row in df_x.iterrows():
    #     create_data_file(row,df)

    # df2 = df.replace('', np.nan, regex=True)

    annotations_file = '//data/scratch/r094879/data/annotations/annotations.csv' 
    df2 = pd.read_csv(annotations_file)
    
    imgs_to_delete = df2['image'][df2['group'].isna()].tolist()

    count = 0
    for img_to_delete in imgs_to_delete:
        os.remove(os.path.join('//data/scratch/r094879/data/images',img_to_delete+'.dcm'))
        count = count + 1
    print(count)
    df2 = df2.dropna(subset=['group'])
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


def plot_images_with_points_256():

    csv_file = '//data/scratch/r094879/data/annotations/annotations.csv' 
    df = pd.read_csv(csv_file)

    img_dir = '//data/scratch/r094879/data/imgs'
    hm_dir = '//data/scratch/r094879/data/heatmaps'
    output_dir = '//data/scratch/r094879/data/images_with_points'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        image_name = row['image']  # Get the DICOM image name from the 'image' column
        output_file_path = os.path.join(output_dir,image_name+'.png')
        img_file_path = os.path.join(img_dir,image_name+'.png')

        lm_pred = np.zeros((13,2))
        hm = np.load(os.path.join(hm_dir, image_name+'.npy'))
        print(hm.shape)

        image_dir = params.image_dir
        target_dir = "annotations/"
        img = dcmread(os.path.join(root,image_dir,filename+".dcm"))
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)
        
        for i in range(13):
            lm_preds = np.unravel_index(hm[:,:,i].argmax(),(256,256))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,1] = lm_preds[1]
            lm_pred[i,0] = lm_preds[0]

        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        print('prediction')
        print(lm_pred)

        csv_file = os.path.join(root,'annotations/annotations.csv')
        csv_df = pd.read_csv(csv_file)

        filtered_row = csv_df[csv_df['image'] == filename]

        x_values = np.array(filtered_row.iloc[:,3:29:2].values).reshape((-1,1))
        y_values = np.array(filtered_row.iloc[:,4:29:2].values).reshape((-1,1))

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.concatenate([x_values,y_values],axis=1)
        print('target')
        print(xy_pairs)

        img = Image.open(img_file_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(lm_pred[:,0], lm_pred[:,1], c='red', s=5, marker='o')
        plt.savefig(output_file_path)
        plt.close(fig)
            

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
        img = dicom_image.pixel_array
        # img = img.astype(float)
        # img = (img-img.min())/(img.max()-img.min())*255.0
        # img = img.astype(np.uint8)

        # img_pil = Image.fromarray(img)
        # final_img = img_pil.resize((256,256))

        # final_img.save(os.path.join(output_dir_3,'test.png'))

        
        # resized_image = cv.resize(img, (256,256))
        # cv.imwrite(os.path.join(output_dir_2,image_name+'.png'), resized_image)

        # img = (img-img.min())/(img.max()-img.min())*255.0

        # # Resize to 256x256
        # img_pil = Image.fromarray(img)
        # final_img = img_pil.resize((256,256))
    
        # Convert to numpy array and add a channel dimension
        # img_array = np.array(img_resized)[..., np.newaxis]
        # final_img = Image.fromarray(img_resized)
        # final_img.save(os.path.join(output_dir_2,image_name+'.png'))
        
        
        # img = apply_voi_lut(dicom_image.pixel_array, dicom_image, index=0)
        # img = lin_stretch_img(img, 0.1, 99.9)  # Apply "linear stretching" (lower percentile 0.1 goes to 0, and percentile 99.9 to 255).

        # if dicom_image[0x0028, 0x0004].value == 'MONOCHROME1':
        #     img = 255-img 

        # cv.imwrite(os.path.join(output_dir_2,image_name+'.png'), img)
        
        # # Normalisation
        # img = (img - img.min())/(img.max() - img.min()) 
        # if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        #     img = 1 - img # some images are inverted
        # # img = cv2.resize(img, (self.size,self.size))
        # image_2 = (img * 255).astype(np.float32)
        # scaled_image = np.uint8(image_2)
        # final_image = Image.fromarray(scaled_image)
        # final_image.save(os.path.join(output_dir_2,image_name+'.png'))

        # # Extract pixel array from the DICOM file and convert to .png
        # pixel_array = dicom_image.pixel_array
        # # img = cv.bitwise_not(pixel_array)
        # # img.save(os.path.join(output_dir_2,image_name+'.png'))
        
        # scaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        # scaled_image = np.uint8(scaled_image)
        # final_image = Image.fromarray(scaled_image)
        # final_image.save(os.path.join(output_dir_2,image_name+'.png'))

        # Get the x and y values for each vertebra
        x_values = row.iloc[3:29:2].values 
        y_values = row.iloc[4:29:2].values

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.array(list(zip(x_values, y_values)))

        hm = create_hm(xy_pairs,(img.shape[0],img.shape[1]),new_dim=256.0,size=5)
        np.save(os.path.join(output_dir,image_name),hm)


def view_heatmaps():
    
    file_path = '//data/scratch/r094879/data/heatmaps/1.2.392.200036.9125.9.0.68100090.749932288.3927965275.npy'

    data = np.load(file_path)

    save_path = '//data/scratch/r094879/data/data_check'

    # Ensure the output directories exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Initialize an array to store the sums of each slice
    cumulative_sum = np.zeros(data.shape[0:2], dtype=data.dtype)
    
    # Iterate through each slice in the 3D array
    for i in range(data.shape[2]):
        slice_data = data[:,:,i]
        
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



# output_filename_prefix = 'volume_data'

# def write_tensor_label_hdf5(mat_filelist):
#     """ We will use constant label (class 0) for the test data """
#     tensor_filenames = [line.rstrip() for line in open(mat_filelist, 'r')]
#     img_size=256

#     N = len(tensor_filenames)

#     # =============================================================================
#     # Specify what data and label to write, CHNAGE this according to your needs...
#     data_dim = [img_size,img_size,1]
#     hm_dim = [img_size,img_size,13]
#     data_dtype = 'uint8'
#     label_dtype = 'uint8'
#     # =============================================================================
    
#     h5_batch_size = N
    
#     # set batch buffer
#     batch_data_dim = [N] + data_dim
#     batch_hm_dim = [N] + hm_dim
#     h5_batch_data = np.zeros(batch_data_dim)
#     h5_batch_hm = np.zeros(batch_hm_dim)
    
#     for k in range(N):
#         mat = sio.loadmat(tensor_filenames[k])
#         d = mat[mat.keys()[0]]
#         l = labels[k]

#         h5_batch_data[k, ...] = d
#         h5_batch_hm[k, ...] = l
        
#         if (k+1)%h5_batch_size == 0 or k==N-1:
#             print '[%s] %d/%d' % (datetime.datetime.now(), k+1, N)
#             print 'batch data shape: ', h5_batch_data.shape
#             h5_filename = output_filename_prefix+str(k/h5_batch_size)+'.h5'
#             print h5_filename
#             print np.shape(h5_batch_data)
#             print np.shape(h5_batch_label)
#             begidx = 0
#             endidx = min(h5_batch_size, (k%h5_batch_size)+1) 
#             print h5_filename, data_dtype, label_dtype
#             save_h5(h5_filename, h5_batch_data[begidx:endidx,:,:,:,:], h5_batch_label[begidx:endidx,:], data_dtype, label_dtype) 


# write_tensor_label_hdf5('mat_filelist.txt', 26, 3)
# (d,l) = load_h5(output_filename_prefix+'0.h5')
# print d.shape
# print l.shape

# def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
#     h5_fout = h5py.File(h5_filename)
#     h5_fout.create_dataset(
#             'images', data=data,
#             compression='gzip', compression_opts=4,
#             dtype=data_dtype,
#     )
#     h5_fout.create_dataset(
#             'heatmaps', data=label,
#             compression='gzip', compression_opts=1,
#             dtype=label_dtype,
#     )
#     h5_fout.close()
