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

from utils.feature_extraction import get_contours, femhead_centre
from utils.process_predictions import pixel_to_mm
from utils.landmark_prep import prep_landmarks

def metric_calculation(msk_dir=os.path.join("Dataset","FemHead Masks"),
                       pred_csv_dir=None,
                       save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                       extra=""):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    msk_dir = os.path.join(root,msk_dir)
    csv_dir = os.path.join(root,"Dataset","CSVs")
    if pred_csv_dir is not None:
        pred_csv_dir = os.path.join(root,pred_csv_dir)
    
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(csv_dir,"*.csv"))]
    
    df = pd.DataFrame(columns = ['Image','r_len_3_4','r_len_4_5','r_ang_3_4','r_femhead_width','r_pct_coverage','r_rat_3_4_width',
                                 'r_rat_4_5_width','r_rat_3_4_5_width','l_len_14_15','l_len_15_16','l_ang_14_15',
                                 'l_femhead_width','l_pct_coverage','l_rat_14_15_width','l_rat_15_16_width',
                                 'l_rat_14_15_16_width'])

    df['Image'] = filenames
    df.to_csv(os.path.join(root,save_dir),index=False)   
    
#     rater_set = {33926936,33930903,33935029,34097900,34132760,34139539,34146428,34192818,34776553,34777653}
    for filename in filenames:
#         if filename in rater_set:
            metric_calc(filename,msk_dir=msk_dir,pred_csv_dir=pred_csv_dir,
                       save_dir=save_dir,extra=extra)
        
        
def metric_calc(filename,msk_dir=os.path.join("Dataset","FemHead Masks"),
                       pred_csv_dir=None,img_dir=os.path.join("Dataset","Images"),
                       csv_dir=os.path.join("Dataset","CSVs"),
                       save_dir=os.path.join("Results","Statistics",'metrics.csv'),
                       extra=""):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    img_dir = os.path.join(root,img_dir)    
    csv_dir = os.path.join(root,csv_dir) 
    print(filename)
    lms_gt, image_size = prep_landmarks(filename,csv_dir)

    df = pd.read_csv(os.path.join(root,save_dir))
    
    if pred_csv_dir is not None:
        lms = pd.read_csv(os.path.join(pred_csv_dir,filename+".csv"))
    else:
        lms = lms_gt
    lms = np.asarray(lms).astype(float).reshape((-1,2))

    if not np.isnan(lms[5,0]) and not np.isnan(lms[16,0]):

        # Define Hilgenreiners slope and centre
        hilg_slope = (lms[5,1]-lms[16,1])/(lms[5,0]-lms[16,0])
        hilg_centre = [abs(lms[5,0]-lms[16,0])/2,abs(lms[5,1]-lms[16,1])/2]

        # Define length between points 3-4
        len_3_4 = math.dist(lms[2,:],lms[3,:])
        len_3_4 = pixel_to_mm(filename,len_3_4)
        df.loc[df["Image"]==int(filename),"r_len_3_4"] = len_3_4 

        # Define length between points 4-5
        len_4_5 = math.dist(lms[3,:],lms[4,:])
        len_4_5 = pixel_to_mm(filename,len_4_5)
        df.loc[df["Image"]==int(filename),"r_len_4_5"] = len_4_5 

        # Define length between points 14-15
        len_14_15 = math.dist(lms[13,:],lms[14,:])
        len_14_15 = pixel_to_mm(filename,len_14_15)
        df.loc[df["Image"]==int(filename),"l_len_14_15"] = len_14_15

        # Define length between points 15-16
        len_15_16 = math.dist(lms[14,:],lms[15,:])
        len_15_16 = pixel_to_mm(filename,len_15_16)
        df.loc[df["Image"]==int(filename),"l_len_15_16"] = len_15_16

#         if os.path.exists(os.path.join(msk_dir,filename+".png")):
#             # Define width of femoral head
#             mask = cv.imread(os.path.join(msk_dir,filename+".png"))
#             contl, contr = get_contours(os.path.join(msk_dir,filename+".png"))

#             halfway = round((lms[14,0]-lms[5,0])/2+lms[5,0])

#             if contr is not None and np.mean(contr) < halfway:
#                 r_width, r_width_line = femhead_width(contr,mask,os.path.join(msk_dir,filename+".png"),
#                                                   lms,image_size,left=False)
#             else:
#                 r_width = np.nan
#                 r_width_line = np.nan

#             if contl is not None and np.mean(contl) > halfway:
#                 l_width, l_width_line = femhead_width(contl,mask,os.path.join(msk_dir,filename+".png"),
#                                                   lms,image_size)
#             else:
#                 l_width = np.nan
#                 l_width_line = np.nan

#             l_width = pixel_to_mm(filename,l_width)
#             df.loc[df["Image"]==int(filename),"l_femhead_width"] = l_width
#             r_width = pixel_to_mm(filename,r_width)
#             df.loc[df["Image"]==int(filename),"r_femhead_width"] = r_width

#             # Define ratios
#             df.loc[df["Image"]==int(filename),"r_rat_3_4_width"] = len_3_4/r_width
#             df.loc[df["Image"]==int(filename),"l_rat_14_15_width"] = len_14_15/l_width
#             df.loc[df["Image"]==int(filename),"r_rat_4_5_width"] = len_4_5/r_width
#             df.loc[df["Image"]==int(filename),"l_rat_15_16_width"] = len_15_16/l_width
#             df.loc[df["Image"]==int(filename),"r_rat_3_4_5_width"] = (len_3_4+len_4_5)/r_width
#             df.loc[df["Image"]==int(filename),"l_rat_14_15_16_width"] = (len_14_15+len_15_16)/l_width

        # Define Perkin's line
        perkins_slope = -1/hilg_slope
        perkins_pt_r = lms[2,:]
        perkins_pt_l = lms[13,:]  

#             # Determine percentage coverage
#             cl = np.expand_dims(contl.astype(np.float32), 1)
#             cl = cv.UMat(cl)
#             l_area = cv.contourArea(cl)

#             cr = np.expand_dims(contr.astype(np.float32), 1)
#             cr = cv.UMat(cr)
#             r_area = cv.contourArea(cr)

#             p_mask = perkins_mask(image_size,perkins_slope,perkins_pt_r,perkins_pt_l)
#             covered_mask = mask_combine_covered(image_size,mask,p_mask) 
#             uncovered_mask = mask_combine_uncovered(image_size,mask,p_mask) 

# #             plt.imshow(mask)
# #             plt.show()
# #             plt.imshow(p_mask)
# #             plt.show()
# #             plt.imshow(covered_mask)
# #             plt.show()
# #             plt.imshow(uncovered_mask,cmap='gray')
# #             plt.show()

#             cv.imwrite(os.path.join(root,"Dataset","Comb_Mask"+extra,filename+".png"),covered_mask)
#     #         plt.imsave(os.path.join(root,"Dataset","Comb_Mask",filename+".png"),covered_mask)

#             comb_contl, comb_contr = get_contours(os.path.join(root,"Dataset",
#                                                                "Comb_Mask"+extra,filename+".png"))

#             if comb_contl is not None:
#                 comb_cl = np.expand_dims(comb_contl.astype(np.float32), 1)
#                 comb_cl = cv.UMat(comb_cl)
#                 comb_l_area = cv.contourArea(comb_cl)
#                 l_pct = comb_l_area/l_area
#             else:
#                 l_pct = 0

#             if comb_contr is not None:
#                 comb_cr = np.expand_dims(comb_contr.astype(np.float32), 1)
#                 comb_cr = cv.UMat(comb_cr)
#                 comb_r_area = cv.contourArea(comb_cr)
#                 r_pct = comb_r_area/r_area
#             else:
#                 r_pct = 0

#             df.loc[df["Image"]==int(filename),"r_pct_coverage"] = r_pct
#             df.loc[df["Image"]==int(filename),"l_pct_coverage"] = l_pct

#         else:
#             covered_mask = None
#             r_pct = np.nan
#             l_pct = np.nan
#             r_width_line = np.nan
#             l_width_line = np.nan

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
        l_ud_slope = (lms[14,1]-lms[13,1])/(lms[14,0]-lms[13,0])
        r_ud_angle = math.degrees(math.atan(r_ud_slope)-math.atan(hilg_slope))
        l_ud_angle = math.degrees(math.atan(hilg_slope)-math.atan(l_ud_slope))
        df.loc[df["Image"]==int(filename),"r_ang_3_4"] = r_ud_angle 
        df.loc[df["Image"]==int(filename),"l_ang_14_15"] = l_ud_angle

        df.to_csv(os.path.join(root,save_dir),index=False)

#         display_measure(filename,img_dir,lms,lms_gt,hilg_slope,image_size,covered_mask,r_ud_angle,l_ud_angle,r_pct,
#                     l_pct,r_width_line,l_width_line,extra,msk_dir)

                          
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


# def display_measure(filename,img_dir,lms,hilg_slope,image_size,r_angle,l_angle,
#                        extra,msk_dir):
    
#     root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#     img = image.imread(os.path.join(img_dir,filename+".png"))[:,:,:3]
    
#     fig = plt.figure(figsize=(30,10))
#     plt.axis('off')
#     plt.imshow(img)
    
#     # plot landmarks
#     plt.scatter(lms[:,0],lms[:,1],s=10,marker='.',c='r')
    
#     # plot hilgenreiners 
#     plt.plot([0,image_size[0]-1],[get_y(hilg_slope,lms[5,:],0),get_y(hilg_slope,lms[5,:],image_size[0]-1)],'b--')
    
#     # plot perkins
#     plt.plot([get_x(-1/hilg_slope,lms[2,:],0),get_x(-1/hilg_slope,lms[2,:],image_size[1]-1)],[0,image_size[1]-1],'b--')
#     plt.plot([get_x(-1/hilg_slope,lms[13,:],0),get_x(-1/hilg_slope,lms[13,:],image_size[1]-1)],[0,image_size[1]-1],'b--')
    
#     # plot 3-4 length
#     plt.plot(lms[2:4,0],lms[2:4,1],c='m')
#     plt.plot(lms[13:15,0],lms[13:15,1],c='m')
#     plt.plot(lms[3:5,0],lms[3:5,1],c='m')
#     plt.plot(lms[14:16,0],lms[14:16,1],c='m')
#         # plot 3-4 angle
#     plt.plot([lms[3,0],lms[2,0]],[lms[3,1],get_y(hilg_slope,lms[3,:],lms[2,0])],'b--')
#     plt.plot([lms[14,0],lms[13,0]],[lms[14,1],get_y(hilg_slope,lms[14,:],lms[13,0])],'b--')
    
#     if os.path.exists(os.path.join(msk_dir,filename+".png")):
#         # Highlight amount of femoral head covered
#         yellow = np.full_like(img,(255,255,0))
#         blend = 0.5
#         img_yellow = cv.addWeighted(img, blend, yellow, 1-blend, 0)
#         result = np.where(covered_mask==(255,255,255), img_yellow, img)
#         plt.imshow(result) 
        
#         if r_width_line is not np.nan:
#             plt.plot(r_width_line[:,0],r_width_line[:,1],c='m')
#         if l_width_line is not np.nan:
#             plt.plot(l_width_line[:,0],l_width_line[:,1],c='m')
            
#         rxy, lxy = femhead_centre(os.path.join(root,"Dataset","Comb_Mask"+extra,filename+".png"))
#         if rxy is not None:
#             plt.text(rxy[1]-20, rxy[0], str(round(r_pct*100))+"$\%$", fontsize=10, color='k', weight='bold')
#         if lxy is not None:
#             plt.text(lxy[1]-40, lxy[0], str(round(l_pct*100))+"$\%$", fontsize=10, color='k', weight='bold')
    
#     # Add text
#     plt.text(lms[2,0], lms[2,1], str(round(r_angle,1))+"$^\circ$", fontsize=10, color='w', weight='bold')
#     plt.text(lms[14,0], lms[13,1], str(round(l_angle,1))+"$^\circ$", fontsize=10, color='w', weight='bold')

#     plt.savefig(os.path.join(root,"Results","Metrics Images"+extra,filename+".png"),bbox_inches='tight')
#     plt.close()
# #     plt.show()


def display_measure(filename,img_dir,lms,lms_gt,hilg_slope,image_size,covered_mask,r_angle,l_angle,r_pct,
                    l_pct,r_width_line,l_width_line,extra,msk_dir):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    img = image.imread(os.path.join(img_dir,filename+".png"))[:,:,:3]
    
    fig = plt.figure(figsize=(30,10))
    plt.axis('off')
    plt.imshow(img)
    
    # plot ground truth landmarks
    plt.scatter(lms_gt[:,0],lms_gt[:,1],s=10,marker='.',c='c')
    
    # plot landmarks
    plt.scatter(lms[:,0],lms[:,1],s=10,marker='.',c='r')
    
    # plot hilgenreiners 
    plt.plot([0,image_size[0]-1],[get_y(hilg_slope,lms[5,:],0),get_y(hilg_slope,lms[5,:],image_size[0]-1)],'b--')
    
    # plot perkins
    plt.plot([get_x(-1/hilg_slope,lms[2,:],0),get_x(-1/hilg_slope,lms[2,:],image_size[1]-1)],[0,image_size[1]-1],'b--')
    plt.plot([get_x(-1/hilg_slope,lms[11,:],0),get_x(-1/hilg_slope,lms[11,:],image_size[1]-1)],[0,image_size[1]-1],'b--')
    
    # plot 3-4 length
    plt.plot(lms[2:4,0],lms[2:4,1],c='m')
    plt.plot(lms[11:13,0],lms[11:13,1],c='m')
    plt.plot(lms[3:5,0],lms[3:5,1],c='m')
    plt.plot(lms[12:14,0],lms[12:14,1],c='m')
        # plot 3-4 angle
    plt.plot([lms[3,0],lms[2,0]],[lms[3,1],get_y(hilg_slope,lms[3,:],lms[2,0])],'b--')
    plt.plot([lms[12,0],lms[11,0]],[lms[12,1],get_y(hilg_slope,lms[12,:],lms[11,0])],'b--')
    
    if os.path.exists(os.path.join(msk_dir,filename+".png")):
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
            
        rxy, lxy = femhead_centre(os.path.join(root,"Dataset","Comb_Mask"+extra,filename+".png"))
        if rxy is not None:
            plt.text(rxy[1]-20, rxy[0], str(round(r_pct*100))+"$\%$", fontsize=10, color='k', weight='bold')
        if lxy is not None:
            plt.text(lxy[1]-40, lxy[0], str(round(l_pct*100))+"$\%$", fontsize=10, color='k', weight='bold')
    
    # Add text
    plt.text(lms[2,0], lms[2,1], str(round(r_angle,1))+"$^\circ$", fontsize=10, color='w', weight='bold')
    plt.text(lms[12,0], lms[11,1], str(round(l_angle,1))+"$^\circ$", fontsize=10, color='w', weight='bold')

    plt.savefig(os.path.join(root,"Results","Metrics Images"+extra,filename+".png"),bbox_inches='tight')
    plt.close()
#     plt.show()


def femhead_width(cont,mask,msk_path,lms,image_size,left=True):
    
    temp_mask = mask.copy()
    halfway = round((lms[14,0]-lms[5,0])/2+lms[5,0])

    if left:
        pt1 = lms[18-2,:]
        pt2 = lms[17-2,:]
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

#         plt.imshow(mask)
#         plt.show()
#         plt.imshow(temp_mask)
#         plt.show()
#         plt.imshow(line_mask)
#         plt.show()
#         plt.imshow(comb_mask)
#         plt.show()

    line = np.where(comb_mask==255)
    line = [[line[1][np.argmin(line[1])],line[0][np.argmin(line[1])]],
            [line[1][np.argmax(line[1])],line[0][np.argmax(line[1])]]]
    line = np.asarray(line).reshape((-1,2))

    width = math.dist(line[0,:], line[1,:])
    max_line = line
    max_width = width

#         if width >= max_width:
#             max_width = width
#             max_line = line
#         else:
#             break

#         line_point[1] = line_point[1]+1

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