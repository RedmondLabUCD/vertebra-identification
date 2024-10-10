import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import gca
from matplotlib.axes import Axes
from utils.landmark_prep import prep_landmarks
import matplotlib.patches as patches

def point_in_rectangle(bl, tr, p) :
    if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
        return True
    else :
          return False


def plot_roi_boxes():
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    pred_dir = os.path.join(root,"Results","UNet_LM_CL","Predicted CSVs")
#     pred_roi_dir = os.path.join(root,"Results","UNet_ROI_LM","Predicted_AUG2 CSVs")
    tar_dir = os.path.join(root,"Dataset","CSVs")
    img_dir = os.path.join(root,"Dataset","Images")
    save_dir = os.path.join(root,"Results","LM_CL_Coarse Assessment")

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                         for file in glob(os.path.join(img_dir,"*.png"))]

    for filename in filenames:
        img = Image.open(os.path.join(img_dir,filename+".png"))

        targets, __ = prep_landmarks(filename,tar_dir)
        targets = targets.reshape((-1,2))
        targets = np.nan_to_num(targets)

        preds = pd.read_csv(os.path.join(pred_dir,filename+".csv"))
        preds = np.asarray(preds).astype(float)

#         preds_roi = pd.read_csv(os.path.join(pred_roi_dir,filename+".csv"))
#         preds_roi = np.asarray(preds_roi).astype(float)

        # Define figure and legends
        labels=["Ground Truth","Model Prediction","Outside 128x128 Box","Inside 128x128 Box"]
        custom_lines = [Line2D([0], [0], color='b', lw=1),
                        Line2D([0], [0], color='y', lw=1),
                        Line2D([0], [0], color='r', lw=1),
                        Line2D([0], [0], color='g', lw=1)]

        fig, ax = plt.subplots(figsize=(5,5),dpi=120)
        ax.legend(custom_lines,labels,prop={"size":8},borderpad=0.4)
        plt.axis('off')
        plt.imshow(img, cmap='gray')

        for index,pred in enumerate(preds):
            if point_in_rectangle((pred[0]-64,pred[1]-64),(pred[0]+64,pred[1]+64),targets[index,:]):
                rect = patches.Rectangle((pred[0]-64, pred[1]-64), 128, 128, linewidth=1, edgecolor='g', facecolor='none')
            else:
                rect = patches.Rectangle((pred[0]-64, pred[1]-64), 128, 128, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.scatter(targets[:,0], targets[:,1], s=5, marker='.', c='b')
        plt.scatter(preds[:,0], preds[:,1], s=5, marker='.', c='y')

        plt.savefig(os.path.join(save_dir,filename+".png"),dpi=120*5)
        plt.show()
        plt.close()
        
        
def calculate_percentage_box(model_name="Attn_UNet_LM_CL",size=128,AUG=False):

    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if AUG:
        pred_dir = os.path.join(root,"Results",model_name,"Predicted_AUG CSVs")
    else:
        pred_dir = os.path.join(root,"Results",model_name,"Predicted CSVs")
    tar_dir = os.path.join(root,"Dataset","CSVs")
    img_dir = os.path.join(root,"Dataset","Images")
    # tar_dir = os.path.join(root,"Dataset","FINAL TEST","CSVs")
    # img_dir = os.path.join(root,"Dataset","FINAL TEST","Images")

    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                         for file in glob(os.path.join(tar_dir,"*.csv"))]
    count_yes = 0
    count_no = 0
    count_lower_yes = 0
    count_lower_no = 0
    dim = size

    for filename in filenames:
        targets, __ = prep_landmarks(filename,tar_dir)
        targets = targets.reshape((-1,2))
        targets = np.nan_to_num(targets)

        preds = pd.read_csv(os.path.join(pred_dir,filename+".csv"))
        preds = np.asarray(preds).astype(float)

        for index,pred in enumerate(preds):
            if point_in_rectangle((pred[0]-(dim/2),pred[1]-(dim/2)),(pred[0]+(dim/2),pred[1]+(dim/2)),targets[index,:]):
                if index == 10 or index == 11 or index == 21 or index == 22:
                    count_lower_yes = count_lower_yes+1
                count_yes = count_yes+1
            else:
                if index == 10 or index == 11 or index == 21 or index == 22:
                    count_lower_no = count_lower_no+1
                count_no = count_no+1
                
    print('Total within box: ' + str(count_yes/(count_yes+count_no)*100))
    print('Excluding femur: ' + str((count_yes-count_lower_yes)/((count_yes-count_lower_yes)+(count_no-count_lower_no))*100))
    print('Only femur: ' + str(count_lower_yes/(count_lower_yes+count_lower_no)*100))
