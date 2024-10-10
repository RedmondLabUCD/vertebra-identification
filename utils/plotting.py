import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import os
from utils.landmark_prep import prep_landmarks
from utils.process_predictions import pixel_to_mm
from scipy.stats import kde
import math

def plot_training(train_losses, train_accs, val_losses, val_accs, model_name="", return_fig=True):
    '''
    Plot losses and accuracy over the training process.

    train_losses (list): List of training losses over training.
    train_accs (list): List of training accuracies over training.
    val_losses (list): List of validation losses over training.
    val_accs (list):List of validation accuracies over training.
    model_name (str): Name of model as a string. 
    return_fig (Boolean): Whether to return figure or not. 
    '''
    epochs = [range(1, len(train_losses)+1, 1)]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    ax1.plot(val_losses, label='Validation loss')
    ax1.plot(train_losses, label="Training loss")
    ax1.set_title('Loss over training for {}'.format(model_name), fontsize=20)
    ax1.set_xlabel("epoch",fontsize=18)
    ax1.set_ylabel("loss",fontsize=18)
    ax1.legend()

    ax2.plot(val_accs, label='Validation accuracy')
    ax2.plot(train_accs, label='Training accuracy')
    ax2.set_title('Accuracy over training for {}'.format(model_name), fontsize=20)
    ax2.set_xlabel("epoch",fontsize=18)
    ax2.set_ylabel("accuracy",fontsize=18)
    ax2.legend()

    fig.tight_layout() 
    if return_fig:
        return fig
 
    
def visualize_age_count(csv_file):
    
    agen = pd.read_csv(csv_file)
#     agen = df[(df.dataset=="Train")]
    labels = sorted(np.asarray(agen['age'].unique()))
    male = agen[agen.sex=="M"]
    female = agen[agen.sex=='F']
    plotdata = pd.DataFrame({
        "Female":female["age"].value_counts().sort_index(ascending=True),
        "Male":male["age"].value_counts().sort_index(ascending=True)
        }, 
        index=labels
    )
    
    plotdata.plot(kind='bar',stacked=True,figsize=(15,5))
    plt.xlabel("Ages",fontsize=16)
    plt.ylabel("Count",fontsize=16)
    plt.title("Age and Gender Distribution",fontsize=20)
    

def age_dice_plot(csv_file):
    
    agen = pd.read_csv(csv_file)
    labels = sorted(np.asarray(agen['age'].unique()))
    male = agen[(agen.sex=='M')]
    female = agen[agen.sex=='F']
    plotdata = pd.DataFrame({
        "Female":female.groupby('age').UNet.mean(),
        "Male":male.groupby('age').UNet.mean()
        }, 
        index=labels
    )
    
    plotdata.plot(kind='bar',figsize=(15,5))
    plt.ylim([0.97, 1.0])
    plt.xlabel("Ages",fontsize=16)
    plt.ylabel("Dice",fontsize=16)
    plt.title("Age and Gender vs Dice Score",fontsize=20)
    
    
def age_count_dice_plot(csv_file):
    
    agen = pd.read_csv(csv_file)
    labels = sorted(np.asarray(agen['age'].unique()))
    male = agen[(agen.sex=='M')]
    female = agen[agen.sex=='F']
    plotdata = pd.DataFrame({
        "Female":(female.groupby('age').UNet.mean())/(female["age"].value_counts().sort_index(ascending=True)),
        "Male":(male.groupby('age').UNet.mean())/(male["age"].value_counts().sort_index(ascending=True)),
        }, 
        index=labels
    )
    
    plotdata.plot(kind='bar',figsize=(15,5))
#     plt.ylim([0.97, 1.0])
    plt.xlabel("Ages",fontsize=16)
    plt.ylabel("Dice / Count",fontsize=16)
    plt.title("Age and Gender vs Dice Score Divided by Data Count",fontsize=20)
    
    
def dataset_age_dice_plot(csv_file,subdir="Test"):
    
    df = pd.read_csv(csv_file)
    agen = df[(df.dataset==subdir)]
    print(agen.shape)
    labels = sorted(np.asarray(agen['age'].unique()))
    male = agen[(agen.sex=='M')]
    female = agen[agen.sex=='F']
    plotdata = pd.DataFrame({
        "Female":female.groupby('age').CENetAUG.mean(),
        "Male":male.groupby('age').CENetAUG.mean()
        }, 
        index=labels
    )
    
    plotdata.plot(kind='bar',figsize=(15,5))
    plt.ylim([0.8, 1.0])
    plt.xlabel("Ages",fontsize=16)
    plt.ylabel("Dice",fontsize=16)
    plt.title("Age and Gender vs Dice Score for CENet AUG",fontsize=20)
    

def gender_dataset_plot(csv_file):
    
    agen = pd.read_csv(csv_file)
    labels = sorted(np.asarray(agen['sex'].unique()))
    train = agen[(agen.dataset=='Train')]
    val = agen[agen.dataset=='Val']
    test = agen[agen.dataset=='Test']
    plotdata = pd.DataFrame({
        "Train":train.groupby('sex').UNet.mean(),
        "Validation":val.groupby('sex').UNet.mean(),
        "Test":test.groupby('sex').UNet.mean()
        }, 
        index=labels
    )
    
    plotdata.plot(kind='bar',figsize=(15,5))
    plt.ylim([0.98, 1.0])
    plt.xlabel("Gender",fontsize=16)
    plt.ylabel("Dice",fontsize=16)
    plt.title("Gender vs Dice Score",fontsize=20)
    
    
def contrast_dice_plot(csv_file,col="UNet",return_fig=True):
    
    contrast_stats = pd.read_csv(csv_file)
    
    X = contrast_stats["Contrast"].to_numpy()
    Y = contrast_stats[col].to_numpy()
    
    y_index = np.where(Y <= 0.5)
    
    X = np.delete(X,y_index[0])
    Y = np.delete(Y,y_index[0])
    
    theta = np.polyfit(X, Y, 1)
    y_line = theta[1] + theta[0] * X

    print(pearsonr(X,Y))
    
    fig = plt.figure(figsize=(10,5))
    plt.scatter(X,Y)
    plt.plot(X, y_line, 'r')
    plt.xlabel("Femoral Head Border Contrast Difference", fontsize=16)
    plt.ylabel("Dice Score", fontsize=16)
    plt.title("Dice vs Contrast for " + col + " Femoral Head Segmentation", fontsize=20)
    plt.legend()
    fig.tight_layout() 
    
    if return_fig:
        return fig
    
    
def lm_error_plot(csv_file,csv_file_2=None,return_fig=True):
    
    lm_errors = pd.read_csv(csv_file)
    print(lm_errors.isna().sum())
    
    if csv_file_2 != None:
        lm_errors_2 = pd.read_csv(csv_file_2)
        lm_errors.rename(columns = {'LM1':'1A','LM2':'2A','LM3':'3A','LM4':'4A','LM5':'5A',
                                    'LM6':'6A','LM7':'7A','LM8':'8A','LM9':'9A','LM10':'10A',
                                    'LM11':'11A','LM12':'12A','LM13':'13A','LM14':'14A','LM15':'15A',
                                    'LM16':'16A','LM17':'17A','LM18':'18A','LM19':'19A','LM20':'20A',
                                    'LM21':'21A','LM22':'22A'}, inplace = True)
        for i in range(22):
            extracted_col = lm_errors_2["LM"+str(22-i)]
            lm_errors.insert(22-i+1,str(22-i)+"B",extracted_col)

    lm_errors = lm_errors.iloc[:,1:]
    fig = lm_errors.plot(kind='box',figsize=(100,20))
    plt.title("Landmark Errors of Final Model", fontsize=45)
    plt.ylabel("Error (mm)",fontsize=30)
    plt.xticks(fontsize=30)
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    plt.savefig(os.path.join(root,"Results","Plots","lm_error_dist.png"))
        
    print(lm_errors.isna().sum())
    
#     if return_fig:
#         return fig


def lm_dist_calc(csv_name="lm_cl_aug2_dists.csv",pred_name="Predicted_CL_AUG2 Pred CSVs"):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    target_dir = os.path.join(root,"Dataset","FINAL TEST","CSVs")
    pred_dir = os.path.join(root,"Results","Final_UNet_ROI_LM_CL",pred_name)
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(target_dir,"*.csv"))]
    
    df = pd.DataFrame(columns = ['Image','LM1X','LM1Y','LM2X','LM2Y','LM3X','LM3Y','LM4X','LM4Y','LM5X','LM5Y','LM6X','LM6Y',
                                 'LM7X','LM7Y','LM8X','LM8Y','LM9X','LM9Y','LM12X','LM12Y','LM13X','LM13Y','LM14X','LM14Y',
                                 'LM15X','LM15Y','LM16X','LM16Y','LM17X','LM17Y','LM18X','LM18Y','LM19X','LM19Y','LM20X','LM20Y'])
    
    df.to_csv(os.path.join(root,"Results","Statistics",csv_name),index=False)   
    df = pd.read_csv(os.path.join(root,"Results","Statistics",csv_name))

    dist_x = np.zeros(len(filenames)*18)
    dist_y = np.zeros(len(filenames)*18)
    
    for index,filename in enumerate(filenames):
        lms_tar, image_size = prep_landmarks(filename,target_dir)
        lms_tar = lms_tar.reshape((-1,2))
        
        hilg_slope = (lms_tar[5,1]-lms_tar[14,1])/(lms_tar[5,0]-lms_tar[14,0])
        hilg_centre = [abs(lms_tar[5,0]-lms_tar[14,0])/2,abs(lms_tar[5,1]-lms_tar[14,1])/2]
        
#         lms_tar = np.nan_to_num(lms_tar)
        lms_tar = np.delete(lms_tar, 21, 0)
        lms_tar = np.delete(lms_tar, 20, 0)
        lms_tar = np.delete(lms_tar, 10, 0)
        lms_tar = np.delete(lms_tar, 9, 0)

        lms_pred = pd.read_csv(os.path.join(pred_dir,filename+".csv"))
        lms_pred = np.asarray(lms_pred).astype(float).reshape((-1,2))

        dist_x[index*18:index*18+18] = lms_tar[:,0] - lms_pred[:,0]
        dist_y[index*18:index*18+18] = lms_tar[:,1] - lms_pred[:,1]
        
        lms_pre_new = np.zeros((18,2))
        lms_tar_new = np.zeros((18,2))
        for i in range(18):
            lms_pred[i,0] = lms_pred[i,0]-hilg_centre[0]
            lms_pred[i,1] = -(lms_pred[i,1]-hilg_centre[1])
            
            angle = math.atan(hilg_slope)

            lms_pre_new[i,0] = hilg_centre[0]+math.cos(angle)*(lms_pred[i,0]-hilg_centre[1])-math.sin(angle)*(lms_pred[i,1]-hilg_centre[1])
            lms_pre_new[i,1] = hilg_centre[1]+math.sin(angle)*(lms_pred[i,0]-hilg_centre[0])+math.cos(angle)*(lms_pred[i,1]-hilg_centre[1])

            lms_tar[i,0] = lms_tar[i,0]-hilg_centre[0]
            lms_tar[i,1] = -(lms_tar[i,1]-hilg_centre[1])

            lms_tar_new[i,0] = hilg_centre[0]+math.cos(angle)*(lms_tar[i,0]-hilg_centre[1])-math.sin(angle)*(lms_tar[i,1]-hilg_centre[1])
            lms_tar_new[i,1] = hilg_centre[1]+math.sin(angle)*(lms_tar[i,0]-hilg_centre[0])+math.cos(angle)*(lms_tar[i,1]-hilg_centre[1])

        new_row = np.zeros(37)
        new_row[0] = filename
        
        for i in range(18):
            new_row[i*2+1] = pixel_to_mm(filename,lms_tar_new[i,0]-lms_pre_new[i,0])
            new_row[i*2+2] = pixel_to_mm(filename,lms_tar_new[i,1]-lms_pre_new[i,1])
            
        df.loc[len(df.index)] = new_row

    dist_x = dist_x[~np.isnan(dist_x)]
    dist_y = dist_y[~np.isnan(dist_y)]
    print(dist_x.mean())
    print(dist_x.std())
    print(dist_y.mean())
    print(dist_y.std())
    
    df.to_csv(os.path.join(root,"Results","Statistics",csv_name),index=False)  
    

def lm_dist_plot(stat_file_name="lm_dists.csv",return_fig=False):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df = pd.read_csv(os.path.join(root,"Results","Statistics",stat_file_name))
    
#     nbins=300
#     x = df.iloc[:,1].values
#     y = df.iloc[:,2].values
#     x_n = x[np.logical_not(np.isnan(x))]
#     y_n = y[np.logical_not(np.isnan(x))]
#     k = kde.gaussian_kde([x_n, y_n])
#     xi, yi = np.mgrid[x_n.min():x_n.max():nbins*1j,
#                       y_n.min():y_n.max():nbins*1j]
#     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    fig, axs = plt.subplots(3, 3)
#     axs[0, 0].scatter(df.iloc[:,1].values, df.iloc[:,2].values, s=1)

#     for i in range(9):
#         x = df.iloc[:,2*i+1].values
#         y = df.iloc[:,2*i+2].values
#         x_n = x[np.logical_not(np.isnan(x))]
#         y_n = y[np.logical_not(np.isnan(x))]
#         k = kde.gaussian_kde([x_n, y_n])
#         xi, yi = np.mgrid[x_n.min():x_n.max():nbins*1j,
#                           y_n.min():y_n.max():nbins*1j]
#         zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
#         if i < 3:
#             ax_x = 0
#         elif i < 6:
#             ax_x = 1
#         else:
#             ax_x = 2
        
#         k = i - ax_x*3
#         if k == 0:
#             ax_y = 0
#         elif k==1:
#             ax_y = 1
#         else:
#             ax_y = 2
        
#         axs[ax_x, ax_y].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
#         axs[ax_x, ax_y].set_title('R LM '+str(i+1),fontsize=10)
                        
#     axs[0, 0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    axs[0, 0].scatter(df.iloc[:,1].values, df.iloc[:,2].values, s=1)
    x1 = df.iloc[:,1].values[~np.isnan(df.iloc[:,1].values)]
    y1 = df.iloc[:,2].values[~np.isnan(df.iloc[:,2].values)]
    axs[0, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
#     axs[0, 0].scatter(np.median(x1), np.median(y1),s=20,color='r')
    axs[0, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[0, 0].set_title('R LM 1',fontsize=10)
    axs[0, 1].scatter(df.iloc[:,3].values, df.iloc[:,4].values, s=1)
    x1 = df.iloc[:,3].values[~np.isnan(df.iloc[:,3].values)]
    y1 = df.iloc[:,4].values[~np.isnan(df.iloc[:,4].values)]
    axs[0, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[0, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[0, 1].set_title('R LM 2',fontsize=10)
    axs[0, 2].scatter(df.iloc[:,5].values, df.iloc[:,6].values, s=1)
    x1 = df.iloc[:,5].values[~np.isnan(df.iloc[:,5].values)]
    y1 = df.iloc[:,6].values[~np.isnan(df.iloc[:,6].values)]
    axs[0, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[0, 2].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[0, 2].set_title('R LM 3',fontsize=10)
    axs[1, 0].scatter(df.iloc[:,7].values, df.iloc[:,8].values, s=1)
    x1 = df.iloc[:,7].values[~np.isnan(df.iloc[:,7].values)]
    y1 = df.iloc[:,8].values[~np.isnan(df.iloc[:,8].values)]
    axs[1, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1)  
    axs[1, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[1, 0].set_title('R LM 4',fontsize=10)
    axs[1, 1].scatter(df.iloc[:,9].values, df.iloc[:,10].values, s=1)
    x1 = df.iloc[:,9].values[~np.isnan(df.iloc[:,9].values)]
    y1 = df.iloc[:,10].values[~np.isnan(df.iloc[:,10].values)]
    axs[1, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[1, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[1, 1].set_title('R LM 5',fontsize=10)
    axs[1, 2].scatter(df.iloc[:,11].values, df.iloc[:,12].values, s=1)
    x1 = df.iloc[:,11].values[~np.isnan(df.iloc[:,11].values)]
    y1 = df.iloc[:,12].values[~np.isnan(df.iloc[:,12].values)]
    axs[1, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[1, 2].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[1, 2].set_title('R LM 6',fontsize=10)
    axs[2, 0].scatter(df.iloc[:,13].values, df.iloc[:,14].values, s=1)
    x1 = df.iloc[:,13].values[~np.isnan(df.iloc[:,13].values)]
    y1 = df.iloc[:,14].values[~np.isnan(df.iloc[:,14].values)]
    axs[2, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[2, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[2, 0].set_title('R LM 7',fontsize=10)
    axs[2, 1].scatter(df.iloc[:,15].values, df.iloc[:,16].values, s=1)
    x1 = df.iloc[:,15].values[~np.isnan(df.iloc[:,15].values)]
    y1 = df.iloc[:,16].values[~np.isnan(df.iloc[:,16].values)]
    axs[2, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[2, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[2, 1].set_title('R LM 8',fontsize=10)
    axs[2, 2].scatter(df.iloc[:,17].values, df.iloc[:,18].values, s=1)
    x1 = df.iloc[:,17].values[~np.isnan(df.iloc[:,17].values)]
    y1 = df.iloc[:,18].values[~np.isnan(df.iloc[:,18].values)]
    axs[2, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs[2, 2].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs[2, 2].set_title('R LM 9',fontsize=10)

    for ax in axs.flat:
        ax.set_xlabel('x-error [mm]', fontsize=8)
        ax.set_ylabel('y-error [mm]', fontsize=8)
        ax.set_xbound(-20,20)
        ax.set_ybound(-20,20)
        ax.set_xticks([-20,-10,0,10,20], labels=["-20","-10","0","10","20"], fontsize=8)
        ax.set_yticks([-20,-10,0,10,20], labels=["-20","-10","0","10","20"], fontsize=8)
        ax.grid()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(root,"Results",'right.png'), dpi=300, bbox_inches='tight')
        
    fig2, axs2 = plt.subplots(3, 3)
    axs2[0, 0].scatter(df.iloc[:,19].values, df.iloc[:,20].values, s=1)
    x1 = df.iloc[:,19].values[~np.isnan(df.iloc[:,19].values)]
    y1 = df.iloc[:,20].values[~np.isnan(df.iloc[:,20].values)]
    axs2[0, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[0, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[0, 0].set_title('L LM 12',fontsize=10)
    axs2[0, 1].scatter(df.iloc[:,21].values, df.iloc[:,22].values, s=1)
    x1 = df.iloc[:,21].values[~np.isnan(df.iloc[:,21].values)]
    y1 = df.iloc[:,22].values[~np.isnan(df.iloc[:,22].values)]
    axs2[0, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[0, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[0, 1].set_title('L LM 13',fontsize=10)
    axs2[0, 2].scatter(df.iloc[:,23].values, df.iloc[:,24].values, s=1)
    x1 = df.iloc[:,23].values[~np.isnan(df.iloc[:,23].values)]
    y1 = df.iloc[:,24].values[~np.isnan(df.iloc[:,24].values)]
    axs2[0, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[0, 2].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[0, 2].set_title('L LM 14',fontsize=10)
    axs2[1, 0].scatter(df.iloc[:,25].values, df.iloc[:,26].values, s=1)
    x1 = df.iloc[:,25].values[~np.isnan(df.iloc[:,25].values)]
    y1 = df.iloc[:,26].values[~np.isnan(df.iloc[:,26].values)]
    axs2[1, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[1, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[1, 0].set_title('L LM 15',fontsize=10)
    axs2[1, 1].scatter(df.iloc[:,27].values, df.iloc[:,28].values, s=1)
    x1 = df.iloc[:,27].values[~np.isnan(df.iloc[:,27].values)]
    y1 = df.iloc[:,28].values[~np.isnan(df.iloc[:,28].values)]
    axs2[1, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[1, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[1, 1].set_title('L LM 16',fontsize=10)
    axs2[1, 2].scatter(df.iloc[:,29].values, df.iloc[:,30].values, s=1)
    x1 = df.iloc[:,29].values[~np.isnan(df.iloc[:,29].values)]
    y1 = df.iloc[:,30].values[~np.isnan(df.iloc[:,30].values)]
    axs2[1, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[1, 2].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[1, 2].set_title('L LM 17',fontsize=10)
    axs2[2, 0].scatter(df.iloc[:,31].values, df.iloc[:,32].values, s=1)
    x1 = df.iloc[:,31].values[~np.isnan(df.iloc[:,31].values)]
    y1 = df.iloc[:,32].values[~np.isnan(df.iloc[:,32].values)]
    axs2[2, 0].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[2, 0].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[2, 0].set_title('L LM 18',fontsize=10)
    axs2[2, 1].scatter(df.iloc[:,33].values, df.iloc[:,34].values, s=1)
    x1 = df.iloc[:,33].values[~np.isnan(df.iloc[:,33].values)]
    y1 = df.iloc[:,34].values[~np.isnan(df.iloc[:,34].values)]
    axs2[2, 1].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[2, 1].text(-19.5,16,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[2, 1].set_title('L LM 19',fontsize=10)
    axs2[2, 2].scatter(df.iloc[:,35].values, df.iloc[:,36].values, s=1)
    x1 = df.iloc[:,35].values[~np.isnan(df.iloc[:,35].values)]
    y1 = df.iloc[:,36].values[~np.isnan(df.iloc[:,36].values)]
    axs2[2, 2].quiver(0,0,np.median(x1),np.median(y1),color='r',units='xy',scale=1) 
    axs2[2, 2].text(-19.5,-18,"median: ({0:.1f}, {1:.1f})".format(np.median(x1),np.median(y1)), bbox=dict(facecolor='none',edgecolor='none',alpha=0.5), fontsize=8)
    axs2[2, 2].set_title('L LM 20',fontsize=10)

    for ax2 in axs2.flat:
        ax2.set_xlabel('x-error [mm]', fontsize=8)
        ax2.set_ylabel('y-error [mm]', fontsize=8)
        ax2.set_xbound(-20,20)
        ax2.set_ybound(-20,20)
        ax2.set_xticks([-20,-10,0,10,20], labels=["-20","-10","0","10","20"], fontsize=8)
        ax2.set_yticks([-20,-10,0,10,20], labels=["-20","-10","0","10","20"], fontsize=8)
        ax2.grid()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax2 in axs2.flat:
        ax2.label_outer()

    plt.savefig(os.path.join(root,"Results",'left.png'), dpi=300, bbox_inches='tight')
    
    
    if return_fig:
        return fig
    
    
#     lm_errors = pd.read_csv(csv_file)
#     print(lm_errors.isna().sum())
    
#     if csv_file_2 != None:
#         lm_errors_2 = pd.read_csv(csv_file_2)
#         lm_errors.rename(columns = {'LM1':'1A','LM2':'2A','LM3':'3A','LM4':'4A','LM5':'5A',
#                                     'LM6':'6A','LM7':'7A','LM8':'8A','LM9':'9A','LM10':'10A',
#                                     'LM11':'11A','LM12':'12A','LM13':'13A','LM14':'14A','LM15':'15A',
#                                     'LM16':'16A','LM17':'17A','LM18':'18A','LM19':'19A','LM20':'20A',
#                                     'LM21':'21A','LM22':'22A'}, inplace = True)
#         for i in range(22):
#             extracted_col = lm_errors_2["LM"+str(22-i)]
#             lm_errors.insert(22-i+1,str(22-i)+"B",extracted_col)

#     lm_errors = lm_errors.iloc[:,1:]
#     fig = lm_errors.plot(kind='box',figsize=(50,10))
#     plt.title("Landmark Errors: A = w/o ROI, B = w/ ROI", fontsize=45)
#     plt.ylabel("Error (mm)",fontsize=30)
#     plt.xticks(fontsize=30)
#     root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#     plt.savefig(os.path.join(root,"Results","Plots","lm_error_dist.png"))
        
#     print(lm_errors.isna().sum())
    
# #     if return_fig:
# #         return fig