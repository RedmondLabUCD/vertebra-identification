import numpy as np
import pandas as pd 
import os
import math
import itertools
import skimage
from glob import glob
from sklearn.utils import shuffle
from torchvision.datasets.utils import list_files
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import image

from utils.feature_extraction import get_contours, femhead_centre
from utils.process_predictions import pixel_to_mm
from utils.landmark_prep import prep_landmarks

def metric_calculation(rater,msk_dir=os.path.join("Dataset","FemHead Masks"),
                       save_dir=os.path.join("Results","Statistics",'rater_metrics.csv'),
                       femur_dir=os.path.join("Inter-Rater",'final_roisin_femur_roi_mins.csv')):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    msk_dir = os.path.join(root,msk_dir)
    csv_dir = os.path.join(root,"Inter-Rater")
    df1 = pd.read_csv(os.path.join(csv_dir,rater+".csv"),header=None)  

    filenames = pd.unique(df1[3].values.ravel())
    
#     df = pd.DataFrame(columns = ['Image','Rater','Version'])
#     df.to_csv(os.path.join(root,save_dir),index=False)

    for filename in filenames:
        filename = filename[:-4]
        print(filename)
        metric_calc(filename,1,rater,csv_dir,femur_dir=femur_dir)
        metric_calc(filename,2,rater,csv_dir,femur_dir=femur_dir)
        metric_calc(filename,3,rater,csv_dir,femur_dir=femur_dir)

        
def metric_calc(filename,version,rater,csv_path,msk_dir=os.path.join("Dataset","FINAL TEST","FemHead Masks"),
                       img_path=os.path.join("Dataset","Images"),
                       save_dir=os.path.join("Results","Statistics",'rater_metrics.csv'),
                       femur_dir=os.path.join("Results","Statistics",'femur_metrics.csv')):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df = pd.read_csv(os.path.join(root,save_dir)) 
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    img_dir = os.path.join(root,img_path) 
    msk_dir = os.path.join(root,msk_dir)
    lms, image_size = get_landmarks(filename,version,rater,csv_path)
    lms = np.asarray(lms).astype(float).reshape((-1,2))

    if not np.isnan(lms[5,0]) and not np.isnan(lms[16,0]):

        # Define Hilgenreiners slope and centre
        hilg_slope = (lms[5,1]-lms[16,1])/(lms[5,0]-lms[16,0])
        hilg_centre = [abs(lms[5,0]-lms[16,0])/2,abs(lms[5,1]-lms[16,1])/2]

        # Define length between points 3-4
        len_3_4 = math.dist(lms[2,:],lms[3,:])
        len_3_4 = pixel_to_mm(filename,len_3_4)

        # Define length between points 4-5
        len_4_5 = math.dist(lms[3,:],lms[4,:])
        len_4_5 = pixel_to_mm(filename,len_4_5)

        # Define length between points 14-15
        len_14_15 = math.dist(lms[13,:],lms[14,:])
        len_14_15 = pixel_to_mm(filename,len_14_15)

        # Define length between points 15-16
        len_15_16 = math.dist(lms[14,:],lms[15,:])
        len_15_16 = pixel_to_mm(filename,len_15_16)

        if os.path.exists(os.path.join(msk_dir,filename+".png")):
            # Define width of femoral head
            mask = cv.imread(os.path.join(msk_dir,filename+".png"))
            contl, contr = get_contours(os.path.join(msk_dir,filename+".png"))
            halfway = round((lms[16,0]-lms[5,0])/2+lms[5,0])

            if contr is not None and np.mean(contr[:,1]) < halfway:
                r_width, r_width_line = femhead_width(contr,mask,os.path.join(msk_dir,filename+".png"),
                                                  lms,image_size,left=False)
            else:
                r_width = np.nan
                r_width_line = np.nan

            if contl is not None and np.mean(contl[:,1]) > halfway:
                l_width, l_width_line = femhead_width(contl,mask,os.path.join(msk_dir,filename+".png"),
                                                  lms,image_size)
            else:
                l_width = np.nan
                l_width_line = np.nan

            l_width = pixel_to_mm(filename,l_width)
            r_width = pixel_to_mm(filename,r_width)
                     
            # Define Perkin's line
            perkins_slope = -1/hilg_slope
            perkins_pt_r = lms[2,:]
            perkins_pt_l = lms[13,:]  

            # Determine percentage coverage
            cl = np.expand_dims(contl.astype(np.float32), 1)
            cl = cv.UMat(cl)
            l_area = cv.contourArea(cl)

            cr = np.expand_dims(contr.astype(np.float32), 1)
            cr = cv.UMat(cr)
            r_area = cv.contourArea(cr)

            p_mask = perkins_mask(image_size,perkins_slope,perkins_pt_r,perkins_pt_l)
            covered_mask = mask_combine_covered(image_size,mask,p_mask) 
            uncovered_mask = mask_combine_uncovered(image_size,mask,p_mask) 
            cv.imwrite(os.path.join(root,"Dataset","Inter_Comb_Mask",filename+"_"+str(version)+".png"),covered_mask)

            comb_contl, comb_contr = get_contours(os.path.join(root,"Dataset",
                                                               "Inter_Comb_Mask",filename+"_"+str(version)+".png"))

            if comb_contl is not None:
                comb_cl = np.expand_dims(comb_contl.astype(np.float32), 1)
                comb_cl = cv.UMat(comb_cl)
                comb_l_area = cv.contourArea(comb_cl)
                l_pct = comb_l_area/l_area
            else:
                l_pct = 0

            if comb_contr is not None:
                comb_cr = np.expand_dims(comb_contr.astype(np.float32), 1)
                comb_cr = cv.UMat(comb_cr)
                comb_r_area = cv.contourArea(comb_cr)
                r_pct = comb_r_area/r_area
            else:
                r_pct = 0

        else:
            covered_mask = None
            r_pct = np.nan
            l_pct = np.nan
            r_width_line = np.nan
            l_width_line = np.nan

        # Define up/down angle
        r_ud_slope = (lms[2,1]-lms[3,1])/(lms[2,0]-lms[3,0])
        l_ud_slope = (lms[14,1]-lms[13,1])/(lms[14,0]-lms[13,0])
        r_ud_angle = math.degrees(math.atan(r_ud_slope)-math.atan(hilg_slope))
        l_ud_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_ud_slope))
        
        # Define up/down angle
        r_aia_slope = (lms[2,1]-lms[4,1])/(lms[2,0]-lms[4,0])
        l_aia_slope = (lms[15,1]-lms[13,1])/(lms[15,0]-lms[13,0])
        r_aia_angle = math.degrees(math.atan(r_aia_slope)-math.atan(hilg_slope))
        l_aia_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_aia_slope))
        
        femur_df = pd.read_csv(os.path.join(root,femur_dir))
    
        r_m_f = femur_df.loc[(femur_df["Image"]==int(filename)) & (femur_df["Version"]==int(version)),"r_m"]
        l_m_f = femur_df.loc[(femur_df["Image"]==int(filename)) & (femur_df["Version"]==int(version)),"l_m"]
        
        # Define femoral angle
        r_fem_angle = math.degrees(math.atan(r_m_f)-math.atan(hilg_slope))
        l_fem_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_m_f))

        if r_fem_angle < 0:
            r_fem_angle = r_fem_angle + 90
        else:
            r_fem_angle = r_fem_angle - 90
            
        if l_fem_angle < 0:
            l_fem_angle = l_fem_angle + 90
        else:
            l_fem_angle = l_fem_angle - 90
        
        new_row = pd.DataFrame({'Image':filename,'Version':int(version),'Rater':rater,
                                'R_lat_sourcil':len_3_4,'R_med_sourcil':len_4_5,'L_lat_sourcil':len_14_15,
                                'L_med_sourcil':len_15_16,'L_fem_width':l_width,'R_fem_width':r_width,
                                'R_rat_3_4_width':len_3_4/r_width,'L_rat_14_15_width':len_14_15/l_width,
                                'R_rat_4_5_width':len_4_5/r_width,'L_rat_15_16_width':len_15_16/l_width,
                                'R_rat_3_4_5_width':(len_3_4+len_4_5)/r_width,
                                'L_rat_14_15_16_width':(len_14_15+len_15_16)/l_width,'R_coverage':r_pct,
                                'L_coverage':l_pct,'R_sourcil_angle':r_ud_angle,'L_sourcil_angle':l_ud_angle,
                                'R_AIA':r_aia_angle,'L_AIA':l_aia_angle,'R_fem_angle':r_fem_angle,
                                'L_fem_angle':l_fem_angle},index=[0])

        df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
        df.to_csv(os.path.join(root,save_dir),index=False)

        
def get_landmarks(filename,version,rater,csv_path):
                  
    data = pd.read_csv(os.path.join(csv_path,rater+'.csv'), header=None)
    
    data = data.loc[(data[3]==str(filename)+".png") & (data[6]==version)]
    data = data.reset_index()

    image_size = np.asarray(data.loc[0,4:6])

    # Get all the landmarks (insert NaN where a landmark does not exist)
    landmarks = pd.DataFrame(columns=[1,2])
    row = 0

    for num in range(1,23):
        if row >= len(data):
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: np.nan, 2: np.nan}])])
        elif str(num) in str(data.loc[row,0]):
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: data.loc[row,1], 2: data.loc[row,2]}])])
            row+=1
        else:
            # print(csv + ' is missing a key point!')
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: np.nan, 2: np.nan}])])

    landmarks = landmarks.reset_index(drop=True)
    landmarks = np.asarray(landmarks).astype(float)
                  
    return landmarks, image_size

                          
def perkins_mask(image_size,perkins_slope,perkins_pt_r,perkins_pt_l):
                 
    p_mask = np.zeros((image_size[1],image_size[0],3), np.uint8)
                 
    for i in range(image_size[0]):
        y_exp_r = get_y(perkins_slope,perkins_pt_r,i)
        y_exp_l = get_y(perkins_slope,perkins_pt_l,i)
        
        for j in range(image_size[1]):
            if perkins_slope < 0 and y_exp_r < j < y_exp_l:
                p_mask[j,i] = (255,255,255)
            elif perkins_slope >= 0 and y_exp_r > j > y_exp_l:
                p_mask[j,i] = (255,255,255)
                          
    return p_mask


def mask_combine_covered(image_size,mask,p_mask):
                 
    comb_mask = np.zeros((image_size[1],image_size[0],3), np.uint8)
                 
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if (mask[j,i] == (255,255,255)).all() and (p_mask[j,i] == (255,255,255)).all():
                comb_mask[j,i]=(255,255,255)
                          
    return comb_mask


def mask_combine_line(image_size,mask,p_mask):
                 
    comb_mask = np.zeros((image_size[1],image_size[0],1), np.uint8)
                 
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if (mask[j,i] == (255,255,255)).all() and (p_mask[j,i] == 255):
                comb_mask[j,i]=255
                          
    return comb_mask


def mask_combine_uncovered(image_size,mask,p_mask):
                 
    comb_mask = np.zeros((image_size[1],image_size[0],3), np.uint8)
                 
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if (mask[j,i] == (255,255,255)).all():
                comb_mask[j,i] = (255,255,255)
            if (p_mask[j,i] == (255,255,255)).all():
                comb_mask[j,i]=(255,255,255)
                          
    return comb_mask

    
# def femhead_width(cont,mask,msk_path,lms,image_size,left=True):
    
#     temp_mask = mask.copy()
#     halfway = round((lms[16,0]-lms[5,0])/2+lms[5,0])

#     if left:
#         pt1 = lms[18,:]
#         pt2 = lms[17,:]
#         __, line_point = femhead_centre(msk_path)
#         for i in range(halfway):
#             temp_mask[:,i]=(0,0,0)
#     else:
#         pt1 = lms[6,:]
#         pt2 = lms[7,:]
#         line_point, __ = femhead_centre(msk_path)
#         for i in range(halfway,image_size[0]):
#             temp_mask[:,i]=(0,0,0)
    
#     line_point = [line_point[1],line_point[0]]
#     slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

#     line_mask = np.zeros((image_size[1],image_size[0],1), np.uint8)

#     for i in range(image_size[0]):
#         y_exp = round(get_y(slope,line_point,i))
#         if 0 <= y_exp < image_size[1]:
#             line_mask[y_exp,i] = 255

#     comb_mask = mask_combine_line(image_size,temp_mask,line_mask) 

# #     plt.imshow(mask)
# #     plt.show()
# #     plt.imshow(temp_mask)
# #     plt.show()
# #     plt.imshow(line_mask)
# #     plt.show()
# #     plt.imshow(comb_mask)
# #     plt.show()

#     line = np.where(comb_mask==255)
#     line = [[line[1][np.argmin(line[1])],line[0][np.argmin(line[1])]],
#             [line[1][np.argmax(line[1])],line[0][np.argmax(line[1])]]]
#     line = np.asarray(line).reshape((-1,2))

#     width = math.dist(line[0,:], line[1,:])

#     return width, line


def femhead_width(cont,mask,msk_path,lms,image_size,left=True,p=2):
    
    temp_mask = mask.copy()
    halfway = round((lms[14+p,0]-lms[5,0])/2+lms[5,0])

    if left:
        pt1 = lms[16+p,:]
        pt2 = lms[15+p,:]
        __, line_point = femhead_centre(msk_path)
        for i in range(halfway):
            temp_mask[:,i]=(0,0,0)
    else:
        pt1 = lms[6,:]
        pt2 = lms[7,:]
        line_point, __ = femhead_centre(msk_path)
        for i in range(halfway,image_size[0]):
            temp_mask[:,i]=(0,0,0)
    
    line_point = [line_point[1],line_point[0]]
    slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    max_width = 0

#     for j in range(50):
    line_mask = np.zeros((image_size[1],image_size[0],1), np.uint8)

    for i in range(image_size[0]):
        y_exp = round(get_y(slope,line_point,i))
        if 0 <= y_exp < image_size[1]:
            line_mask[y_exp,i] = 255

    comb_mask = mask_combine_line(image_size,temp_mask,line_mask) 

    line = np.where(comb_mask==255)
    line = [[line[1][np.argmin(line[1])],line[0][np.argmin(line[1])]],
            [line[1][np.argmax(line[1])],line[0][np.argmax(line[1])]]]
    line = np.asarray(line).reshape((-1,2))

    width = math.dist(line[0,:], line[1,:])
    max_line = line
    max_width = width

    return max_width, max_line
    

def get_y(slope,point,x):
    return slope*(x-point[0])+point[1]
        
    
def get_x(slope,point,y):
    return (y-point[1])/slope+point[0]