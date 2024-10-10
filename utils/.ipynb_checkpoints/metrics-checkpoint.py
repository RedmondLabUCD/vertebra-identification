#import libraries 
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
from matplotlib.lines import Line2D

from utils.feature_extraction import get_contours, femhead_centre
from utils.process_predictions import pixel_to_mm
from utils.landmark_prep import prep_landmarks

def metric_calculation(msk_dir=os.path.join("Dataset","FemHead Masks"),
                       csv_dir = os.path.join("Dataset","CSVs"),
                       img_dir=os.path.join("Dataset","Images"),
                       pred_csv_dir=None,
                       save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                       femur_dir=os.path.join("Dataset","FINAL TEST","pred_femur_roi_mins.csv"),
                       extra=""):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    msk_dir = os.path.join(root,msk_dir)
    csv_dir_full = os.path.join(root,csv_dir)
    if pred_csv_dir is not None:
        pred_csv_dir = os.path.join(root,pred_csv_dir)
    
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(csv_dir_full,"*.csv"))]
    
    df = pd.DataFrame(columns = ['Image','R_lat_sourcil','R_med_sourcil','R_sourcil_angle','R_AIA','R_fem_angle',
                                 'R_fem_width','R_coverage','R_coverage_new','r_rat_3_4_width','r_rat_4_5_width','r_rat_3_4_5_width',
                                 'L_lat_sourcil','L_med_sourcil','L_sourcil_angle','L_AIA','L_fem_angle',
                                 'L_fem_width','L_coverage','L_coverage_new','l_rat_14_15_width','l_rat_15_16_width',
                                 'l_rat_14_15_16_width','hilg_slope','r_width_line_1_x','l_width_line_1_x',
                                 'r_width_line_1_y','l_width_line_1_y','r_width_line_2_x','l_width_line_2_x',
                                 'r_width_line_2_y','l_width_line_2_y'])

    df['Image'] = filenames
    df.to_csv(os.path.join(root,save_dir),index=False)   
    
#     rater_set = {33926936,33930903,33935029,34097900,34132760,34139539,34146428,34192818,34776553,34777653}
    for filename in filenames:
#         if filename in rater_set:
        metric_calc(filename,csv_dir=csv_dir,img_dir=img_dir,msk_dir=msk_dir,pred_csv_dir=pred_csv_dir,
                   save_dir=save_dir,femur_dir=femur_dir,extra=extra)
    
    
def metric_display(msk_dir=os.path.join("Dataset","FemHead Masks"),
                   csv_dir = os.path.join("Dataset","CSVs"),
                   img_dir=os.path.join("Dataset","Images"),
                   pred_csv_dir=None,
                   save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                   femur_dir=os.path.join("Dataset","FINAL TEST","pred_femur_roi_mins.csv"),
                   extra=""):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    msk_dir = os.path.join(root,msk_dir)
    csv_dir_full = os.path.join(root,csv_dir)
    if pred_csv_dir is not None:
        pred_csv_dir = os.path.join(root,pred_csv_dir)
    
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(csv_dir_full,"*.csv"))]
    
    if not os.path.exists(os.path.join(root,"Results","Metrics Images"+extra)):
        os.makedirs(os.path.join(root,"Results","Metrics Images"+extra))
    
    for filename in filenames:
        display_measure(filename,pred_csv_dir=pred_csv_dir,msk_dir=msk_dir,img_dir=img_dir,csv_dir=csv_dir,
                        save_dir=save_dir,femur_dir=femur_dir,extra=extra)

        
def metric_calc(filename,msk_dir=os.path.join("Dataset","FemHead Masks"),
                       pred_csv_dir=None,img_dir=os.path.join("Dataset","Images"),
                       csv_dir=os.path.join("Dataset","CSVs"),
                       save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                       femur_dir=os.path.join("Dataset","FINAL TEST","pred_femur_roi_mins.csv"),
                       extra=""):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
    csv_dir_full = os.path.join(root,csv_dir)
                
    print(filename)
    if not os.path.exists(os.path.join(root,"Dataset","Comb_Mask"+extra)):
        os.makedirs(os.path.join(root,"Dataset","Comb_Mask"+extra))
    lms_gt, image_size = prep_landmarks(filename,csv_dir_full)

    df = pd.read_csv(os.path.join(root,save_dir))
    femur_df = pd.read_csv(os.path.join(root,femur_dir))
    
    r_m_f = femur_df.loc[df["Image"]==int(filename),"r_m"]
    l_m_f = femur_df.loc[df["Image"]==int(filename),"l_m"]
    
    if pred_csv_dir is not None:
        lms = pd.read_csv(os.path.join(pred_csv_dir,filename+".csv"))
        p = 0
    else:
        lms = lms_gt
        p = 2
    lms = np.asarray(lms).astype(float).reshape((-1,2))

    if not np.isnan(lms[5,0]) and not np.isnan(lms[14+p,0]):

        # Define Hilgenreiners slope and centre
        hilg_slope = (lms[5,1]-lms[14+p,1])/(lms[5,0]-lms[14+p,0])
        hilg_centre = [abs(lms[5,0]-lms[14+p,0])/2,abs(lms[5,1]-lms[14+p,1])/2]
        df.loc[df["Image"]==int(filename),"hilg_slope"] = hilg_slope

        # Define length between points 3-4
        len_3_4 = math.dist(lms[2,:],lms[3,:])
        len_3_4 = pixel_to_mm(filename,len_3_4)
        df.loc[df["Image"]==int(filename),"R_lat_sourcil"] = len_3_4 

        # Define length between points 4-5
        len_4_5 = math.dist(lms[3,:],lms[4,:])
        len_4_5 = pixel_to_mm(filename,len_4_5)
        df.loc[df["Image"]==int(filename),"R_med_sourcil"] = len_4_5 

        # Define length between points 14-15
        len_14_15 = math.dist(lms[11+p,:],lms[12+p,:])
        len_14_15 = pixel_to_mm(filename,len_14_15)
        df.loc[df["Image"]==int(filename),"L_lat_sourcil"] = len_14_15

        # Define length between points 15-16
        len_15_16 = math.dist(lms[12+p,:],lms[13+p,:])
        len_15_16 = pixel_to_mm(filename,len_15_16)
        df.loc[df["Image"]==int(filename),"L_med_sourcil"] = len_15_16

        if os.path.exists(os.path.join(msk_dir,filename+".png")):
            # Define width of femoral head
            mask = cv.imread(os.path.join(msk_dir,filename+".png"))
            contl, contr = get_contours(os.path.join(msk_dir,filename+".png"))

            halfway = round((lms[14+p,0]-lms[5,0])/2+lms[5,0])

            if contr is not None and np.mean(contr[:,1]) < halfway:
                r_width, r_width_line = femhead_width(contr,mask,os.path.join(msk_dir,filename+".png"),
                                                  lms,image_size,left=False,p=p)
                df.loc[df["Image"]==int(filename),"r_width_line_1_x"] = r_width_line[0,0]
                df.loc[df["Image"]==int(filename),"r_width_line_1_y"] = r_width_line[0,1]
                df.loc[df["Image"]==int(filename),"r_width_line_2_x"] = r_width_line[1,0]
                df.loc[df["Image"]==int(filename),"r_width_line_2_y"] = r_width_line[1,1]
            else:
                r_width = np.nan
                r_width_line = np.nan
                df.loc[df["Image"]==int(filename),"r_width_line_1_x"] = np.nan
                df.loc[df["Image"]==int(filename),"r_width_line_1_y"] = np.nan
                df.loc[df["Image"]==int(filename),"r_width_line_2_x"] = np.nan
                df.loc[df["Image"]==int(filename),"r_width_line_2_y"] = np.nan

            if contl is not None and np.mean(contl[:,1]) > halfway:
                l_width, l_width_line = femhead_width(contl,mask,os.path.join(msk_dir,filename+".png"),
                                                  lms,image_size,p=p)
                df.loc[df["Image"]==int(filename),"l_width_line_1_x"] = l_width_line[0,0]
                df.loc[df["Image"]==int(filename),"l_width_line_1_y"] = l_width_line[0,1]
                df.loc[df["Image"]==int(filename),"l_width_line_2_x"] = l_width_line[1,0]
                df.loc[df["Image"]==int(filename),"l_width_line_2_y"] = l_width_line[1,1]
            else:
                l_width = np.nan
                l_width_line = np.nan
                df.loc[df["Image"]==int(filename),"l_width_line_1_x"] = np.nan
                df.loc[df["Image"]==int(filename),"l_width_line_1_y"] = np.nan
                df.loc[df["Image"]==int(filename),"l_width_line_2_x"] = np.nan
                df.loc[df["Image"]==int(filename),"l_width_line_2_y"] = np.nan

            l_width = pixel_to_mm(filename,l_width)
            df.loc[df["Image"]==int(filename),"L_fem_width"] = l_width
            r_width = pixel_to_mm(filename,r_width)
            df.loc[df["Image"]==int(filename),"R_fem_width"] = r_width

            # Define ratios
            df.loc[df["Image"]==int(filename),"r_rat_3_4_width"] = len_3_4/r_width
            df.loc[df["Image"]==int(filename),"l_rat_14_15_width"] = len_14_15/l_width
            df.loc[df["Image"]==int(filename),"r_rat_4_5_width"] = len_4_5/r_width
            df.loc[df["Image"]==int(filename),"l_rat_15_16_width"] = len_15_16/l_width
            df.loc[df["Image"]==int(filename),"r_rat_3_4_5_width"] = (len_3_4+len_4_5)/r_width
            df.loc[df["Image"]==int(filename),"l_rat_14_15_16_width"] = (len_14_15+len_15_16)/l_width

            # Define Perkin's line
            perkins_slope = -1/hilg_slope
            perkins_pt_r = lms[2,:]
            perkins_pt_l = lms[11+p,:]  

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

#             plt.imshow(mask)
#             plt.show()
#             plt.imshow(p_mask)
#             plt.show()
#             plt.imshow(covered_mask)
#             plt.show()
#             plt.imshow(uncovered_mask,cmap='gray')
#             plt.show()

            cv.imwrite(os.path.join(root,"Dataset","Comb_Mask"+extra,filename+".png"),covered_mask)
    #         plt.imsave(os.path.join(root,"Dataset","Comb_Mask",filename+".png"),covered_mask)

            comb_contl, comb_contr = get_contours(os.path.join(root,"Dataset",
                                                               "Comb_Mask"+extra,filename+".png"))

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
            
        df.loc[df["Image"]==int(filename),"R_coverage"] = r_pct
        df.loc[df["Image"]==int(filename),"L_coverage"] = l_pct

#         # Define landmarks adjusted to new coordinate system
#         lms_adj = np.ones((22,2))*np.nan
#         for i in range(22):
#             lms_adj[i,0] = lms[i,0]-hilg_centre[0]
#             lms_adj[i,1] = -(lms[i,1]-hilg_centre[1])
#             lms_adj[i,0] = pixel_to_mm(filename,lms_adj[i,0])
#             lms_adj[i,1] = pixel_to_mm(filename,lms_adj[i,1])
#             df.loc[df["Image"]==int(filename),"LM"+str(i+1)+"_adj_x"] = lms_adj[i,0]
#             df.loc[df["Image"]==int(filename),"LM"+str(i+1)+"_adj_y"] = lms_adj[i,1]

        # Define up/down angle
        r_ud_slope = (lms[2,1]-lms[3,1])/(lms[2,0]-lms[3,0])
        l_ud_slope = (lms[12+p,1]-lms[11+p,1])/(lms[12+p,0]-lms[11+p,0])
        r_ud_angle = math.degrees(math.atan(r_ud_slope)-math.atan(hilg_slope))
        l_ud_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_ud_slope))
        df.loc[df["Image"]==int(filename),"R_sourcil_angle"] = r_ud_angle 
        df.loc[df["Image"]==int(filename),"L_sourcil_angle"] = l_ud_angle
        
        #Define AIA
        r_aia_slope = (lms[2,1]-lms[4,1])/(lms[2,0]-lms[4,0])
        l_aia_slope = (lms[13+p,1]-lms[11+p,1])/(lms[13+p,0]-lms[11+p,0])
        r_aia_angle = math.degrees(math.atan(r_aia_slope)-math.atan(hilg_slope))
        l_aia_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_aia_slope))
        df.loc[df["Image"]==int(filename),"R_AIA"] = r_aia_angle 
        df.loc[df["Image"]==int(filename),"L_AIA"] = l_aia_angle

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
            
        df.loc[df["Image"]==int(filename),"R_fem_angle"] = r_fem_angle 
        df.loc[df["Image"]==int(filename),"L_fem_angle"] = l_fem_angle

        rxy, lxy = femhead_centre(os.path.join(root,msk_dir,filename+".png"))  
        contr_x, contr_y = rotate(rxy, contr, math.radians(r_fem_angle))
        contl_x, contl_y = rotate(lxy, contl, -math.radians(l_fem_angle))
        
        contr_new = np.stack((contr_x, contr_y), axis=1)
        contl_new = np.stack((contl_x, contl_y), axis=1)

        contr_r = np.asarray(contr)
        contr_r = np.reshape(contr_r,(-1,2))
        contr_r = np.stack((contr_r[:,1], contr_r[:,0]), axis=1)
        contr_r = np.reshape(contr_r,(-1,2))

        contl_l = np.asarray(contl)
        contl_l = np.reshape(contl_l,(-1,2))
        contl_l = np.stack((contl_l[:,1], contl_l[:,0]), axis=1)
        contl_l = np.reshape(contl_l,(-1,2))

        straight_femur_mask_array = np.zeros((image_size[1],image_size[0],3), np.uint8)
        cv.imwrite('sfm_img'+str(filename)+'.png', straight_femur_mask_array)
        sfm_img2 = cv.imread('sfm_img'+str(filename)+'.png')
        cv.drawContours(sfm_img2, [contr_new.astype(int)], 0, (255, 255, 255), thickness=-1)
        cv.drawContours(sfm_img2, [contl_new.astype(int)], 0, (255, 255, 255), thickness=-1)
        cv.imwrite('sfm_img'+str(filename)+'.png', sfm_img2)

        contl_new, contr_new = get_contours('sfm_img'+str(filename)+'.png')
        sfm_img2 = cv.imread('sfm_img'+str(filename)+'.png')

         # Determine percentage coverage
        cl = np.expand_dims(contl_new.astype(np.float32), 1)
        cl = cv.UMat(cl)
        l_area_n = cv.contourArea(cl)

        cr = np.expand_dims(contr_new.astype(np.float32), 1)
        cr = cv.UMat(cr)
        r_area_n = cv.contourArea(cr)

        p_mask = perkins_mask(image_size,perkins_slope,perkins_pt_r,perkins_pt_l)
        covered_mask = mask_combine_covered(image_size,sfm_img2,p_mask) 
        cv.imwrite(os.path.join(root,"Dataset","Comb_Mask_rotate",filename+".png"),covered_mask)
        comb_contl_n, comb_contr_n = get_contours(os.path.join(root,"Dataset","Comb_Mask_rotate",filename+".png"))

        # plt.imshow(p_mask)
        # plt.show()
        plt.imshow(covered_mask)
        plt.show()

        if comb_contl_n is not None:
            comb_cl = np.expand_dims(comb_contl_n.astype(np.float32), 1)
            comb_cl = cv.UMat(comb_cl)
            comb_l_area = cv.contourArea(comb_cl)
            l_pct_new = comb_l_area/l_area_n
        else:
            l_pct_new = 0

        if comb_contr_n is not None:
            comb_cr = np.expand_dims(comb_contr_n.astype(np.float32), 1)
            comb_cr = cv.UMat(comb_cr)
            comb_r_area = cv.contourArea(comb_cr)
            r_pct_new = comb_r_area/r_area_n
        else:
            r_pct_new = 0

        print("New percentage:")
        print(str(r_pct) + " -> " + str(r_pct_new))
        print(str(l_pct) + " -> " + str(l_pct_new))

        df.loc[df["Image"]==int(filename),"R_coverage_new"] = r_pct_new
        df.loc[df["Image"]==int(filename),"L_coverage_new"] = l_pct_new
        
        sfm_img3 = cv.imread('sfm_img'+str(filename)+'.png')
        cv.drawContours(sfm_img3, [contr_r.astype(int)], 0, (255, 0, 0), 3)
        cv.drawContours(sfm_img3, [contl_l.astype(int)], 0, (255, 0, 0), 3)
        plt.imshow(sfm_img3)
        plt.show()
        
        df.to_csv(os.path.join(root,save_dir),index=False)

#         display_measure(filename,pred_csv_dir=pred_csv_dir,img_dir=img_dir,
#                        csv_dir=csv_dir,msk_dir=msk_dir,
#                        save_dir=save_dir,
#                        extra=extra)

                          
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


def display_measure(filename,pred_csv_dir=None,img_dir=os.path.join("Dataset","Images"),
                       csv_dir=os.path.join("Dataset","CSVs"),
                       msk_dir=os.path.join("Dataset","FemHead Masks"),
                       save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                       femur_dir=os.path.join("Dataset","FINAL TEST","pred_femur_roi_mins.csv"),
                       extra=""):
    
    print(filename)
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    img = image.imread(os.path.join(root,img_dir,filename+".png"))[:,:,:3]
    covered_mask = cv.imread(os.path.join(root,"Dataset","Comb_Mask"+extra,filename+".png"))
    
    if not os.path.exists(os.path.join(root,"Results","Metrics Images"+extra)):
        os.makedirs(os.path.join(root,"Results","Metrics Images"+extra))
      
    csv_dir = os.path.join(root,csv_dir) 
    lms_gt, image_size = prep_landmarks(filename,csv_dir)
    
    if pred_csv_dir is not None:
        lms = pd.read_csv(os.path.join(root,pred_csv_dir,filename+".csv"))
        p = 0
    else:
        lms = lms_gt
        p = 2
    lms = np.asarray(lms).astype(float).reshape((-1,2))
    
    df = pd.read_csv(os.path.join(root,save_dir))
    
    hilg_slope = df.loc[df["Image"]==int(filename),"hilg_slope"].values[0] 
    r_sourcil_ang = df.loc[df["Image"]==int(filename),"R_sourcil_angle"].values[0]
    l_sourcil_ang = df.loc[df["Image"]==int(filename),"L_sourcil_angle"].values[0]
    r_aia = df.loc[df["Image"]==int(filename),"R_AIA"].values[0]
    l_aia = df.loc[df["Image"]==int(filename),"L_AIA"].values[0]
    r_fem_ang = df.loc[df["Image"]==int(filename),"R_fem_angle"].values[0]
    l_fem_ang = df.loc[df["Image"]==int(filename),"L_fem_angle"].values[0]
    r_pct = df.loc[df["Image"]==int(filename),"R_coverage"].values[0]
    l_pct = df.loc[df["Image"]==int(filename),"L_coverage"].values[0]
    r_1x = df.loc[df["Image"]==int(filename),"r_width_line_1_x"].values[0]
    l_1x = df.loc[df["Image"]==int(filename),"l_width_line_1_x"].values[0]
    r_1y = df.loc[df["Image"]==int(filename),"r_width_line_1_y"].values[0]
    l_1y = df.loc[df["Image"]==int(filename),"l_width_line_1_y"].values[0] 
    r_2x = df.loc[df["Image"]==int(filename),"r_width_line_2_x"].values[0] 
    l_2x = df.loc[df["Image"]==int(filename),"l_width_line_2_x"].values[0] 
    r_2y = df.loc[df["Image"]==int(filename),"r_width_line_2_y"].values[0] 
    l_2y = df.loc[df["Image"]==int(filename),"l_width_line_2_y"].values[0] 
    
    r_width_line = [[r_1x,r_1y],[r_2x,r_2y]]
    r_width_line = np.reshape(r_width_line,(2,2))
    l_width_line = [[l_1x,l_1y],[l_2x,l_2y]]
    l_width_line = np.reshape(l_width_line,(2,2))

    femur_df = pd.read_csv(os.path.join(root,femur_dir))
                
    r_m_f = femur_df.loc[df["Image"]==int(filename),"r_m"]
    r_x_f = femur_df.loc[df["Image"]==int(filename),"r_x"]
    r_y_f = femur_df.loc[df["Image"]==int(filename),"r_y"]
    l_m_f = femur_df.loc[df["Image"]==int(filename),"l_m"]
    l_x_f = femur_df.loc[df["Image"]==int(filename),"l_x"]
    l_y_f = femur_df.loc[df["Image"]==int(filename),"l_y"]
                
    if not np.isnan(lms[5,0]) and not np.isnan(lms[14+p,0]):
    
        fig = plt.figure(figsize=(30,10))
#         fig, ax = plt.subplots(figsize=(30,10))
        plt.axis('off')
        plt.imshow(img)
        
        gt_point = Line2D([], [], color='y', marker='.', linestyle='None',
                          markersize=15, label='Ground Truth')
        tar_point = Line2D([], [], color='b', marker='.', linestyle='None',
                          markersize=15, label='System Estimate')
        plt.legend(handles=[gt_point,tar_point],fontsize=15)

        # plot ground truth landmarks
        plt.scatter(lms_gt[:,0],lms_gt[:,1],s=75,marker='.',c='y')

        # plot landmarks
        plt.scatter(lms[:,0],lms[:,1],s=60,marker='.',c='b')

        # plot hilgenreiners 
        plt.plot([0,image_size[0]-1],[get_y(hilg_slope,lms[5,:],0),
                                      get_y(hilg_slope,lms[5,:],image_size[0]-1)],'b--')

        # plot perkins
        plt.plot([get_x(-1/hilg_slope,lms[2,:],0),get_x(-1/hilg_slope,lms[2,:],
                 image_size[1]-1)],[0,image_size[1]-1],'b--')
        plt.plot([get_x(-1/hilg_slope,lms[11+p,:],0),get_x(-1/hilg_slope,lms[11+p,:],
                 image_size[1]-1)],[0,image_size[1]-1],'b--')

        # plot 3-4 length
        plt.plot(lms[2:4,0],lms[2:4,1],c='m')
        plt.plot(lms[11+p:13+p,0],lms[11+p:13+p,1],c='m')
        plt.plot(lms[3:5,0],lms[3:5,1],c='m')
        plt.plot(lms[12+p:14+p,0],lms[12+p:14+p,1],c='m')
        # plot 3-4 angle
        # plt.plot([lms[3,0],lms[2,0]],[lms[3,1],get_y(hilg_slope,lms[3,:],lms[2,0])],'b--')
        # plt.plot([lms[12+p,0],lms[11+p,0]],[lms[12+p,1],get_y(hilg_slope,lms[12+p,:],
        #                                                       lms[11+p,0])],'b--')
                
        plt.plot([get_x(r_m_f,[r_x_f,r_y_f],0),get_x(r_m_f,[r_x_f,r_y_f],image_size[1]-1)],
                 [0,image_size[1]-1],'m--')
        plt.plot([get_x(l_m_f,[l_x_f,l_y_f],0),get_x(l_m_f,[l_x_f,l_y_f],image_size[1]-1)],
                 [0,image_size[1]-1],'m--')

        if os.path.exists(os.path.join(root,msk_dir,filename+".png")):
            # Highlight amount of femoral head covered
            yellow = np.full_like(img,(255,255,0))
            blend = 0.5
            img_yellow = cv.addWeighted(img, blend, yellow, 1-blend, 0)
            result = np.where(covered_mask==(255,255,255), img_yellow, img)
            plt.imshow(result) 

            if r_width_line is not np.nan:
                plt.plot(r_width_line[:,0],r_width_line[:,1],c='m')
            if l_width_line is not np.nan:
                plt.plot(l_width_line[:,0],l_width_line[:,1],c='m')

            rxy, lxy = femhead_centre(os.path.join(root,"Dataset","Comb_Mask"+extra,
                                                   filename+".png"))
            if rxy is not None:
                plt.text(rxy[1]-20, rxy[0], str(round(r_pct*100))+"$\%$",
                         fontsize=15, color='k', weight='bold')
            if lxy is not None:
                plt.text(lxy[1]-40, lxy[0], str(round(l_pct*100))+"$\%$",
                         fontsize=15, color='k', weight='bold')

        # Add text
        plt.text(100, image_size[1]-100, 
                 "$\\alpha ^R_{sourcil}$: " + str(round(r_sourcil_ang,1))+"$^\circ$\n" + 
                 "$AIA^R$: " + str(round(r_aia,1))+"$^\circ$\n" + 
                 "$\\alpha ^R_{femur}$: " + str(round(r_fem_ang,1))+"$^\circ$", 
                 fontsize=15, color='w', weight='bold')
        plt.text(image_size[0]-500, image_size[1]-100, 
                 "$\\alpha ^L_{sourcil}$: " + str(round(l_sourcil_ang,1))+"$^\circ$\n" + 
                 "$AIA^L$: " + str(round(l_aia,1))+"$^\circ$\n" + 
                 "$\\alpha ^L_{femur}$: " + str(round(l_fem_ang,1))+"$^\circ$", 
                 fontsize=15, color='w', weight='bold')
        
        plt.savefig(os.path.join(root,"Results","Metrics Images"+extra,filename+".png"),
                    bbox_inches='tight')
#         plt.close()
        plt.show()


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox = origin[0]*np.ones(len(point[:,0]))
    oy = origin[1]*np.ones(len(point[:,1]))

    qx = ox + math.cos(angle) * (point[:,0] - ox) - math.sin(angle) * (point[:,1] - oy)
    qy = oy + math.sin(angle) * (point[:,0] - ox) + math.cos(angle) * (point[:,1] - oy)
    
    return qy, qx


def femhead_width(cont,mask,msk_path,lms,image_size,left=True,p=0):
    
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
#     max_width = 0

# #     for j in range(50):
#     line_mask = np.zeros((image_size[1],image_size[0],1), np.uint8)

#     for i in range(image_size[0]):
#         y_exp = round(get_y(slope,line_point,i))
#         if 0 <= y_exp < image_size[1]:
#             line_mask[y_exp,i] = 255

#     comb_mask = mask_combine_line(image_size,temp_mask,line_mask) 

# #         plt.imshow(mask)
# #         plt.show()
# #         plt.imshow(temp_mask)
# #         plt.show()
# #         plt.imshow(line_mask)
# #         plt.show()
# #         plt.imshow(comb_mask)
# #         plt.show()

#     line = np.where(comb_mask==255)
#     line = [[line[1][np.argmin(line[1])],line[0][np.argmin(line[1])]],
#             [line[1][np.argmax(line[1])],line[0][np.argmax(line[1])]]]
#     line = np.asarray(line).reshape((-1,2))

#     width = math.dist(line[0,:], line[1,:])
#     max_line = line
#     max_width = width

# #         if width >= max_width:
# #             max_width = width
# #             max_line = line
# #         else:
# #             break

# #         line_point[1] = line_point[1]+1

#     return max_width, max_line
    

def get_y(slope,point,x):
    return slope*(x-point[0])+point[1]
        
    
def get_x(slope,point,y):
    return (y-point[1])/slope+point[0]