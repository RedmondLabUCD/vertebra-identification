# Vertebral Centroid Localisation and Classification in Sagittal Spinal Radiographs

This repository contains the official implementation of:

> **C. Gartland, F. Koromani, J. Healy, G. Roshchupkin, F. Rivadeneira, and S. J. Redmond**  
> *Tailored Loss Functions to Improve Vertebra Centroid Localisation and Classification in Sagittal Spinal Radiographs*  
> Proc. 33rd Artificial Intelligence and Cognitive Science (AICS), 2025 (2026, in press)

## Overview

Radiographic analysis of the spine typically requires manual identification of vertebrae, which is:
- Time-consuming  
- Subject to inter- and intra-observer variability  
- Difficult to scale to large datasets  

This work proposes an automated system that:
- Detects vertebral centroids  
- Assigns vertebral labels (e.g., T4–L4)  
- Handles real-world dataset challenges, including:
  - Partial annotations  
  - Missing vertebrae  
  - Variable field-of-view  

## Method

The model follows a heatmap-based landmark detection approach using a U-Net.

### Pipeline

1. **Input**
   - Sagittal spinal radiograph  

2. **Model**
   - U-Net predicts multi-channel heatmaps
   - Each channel corresponds to a vertebra  

3. **Output**
   - Peak detection → vertebra centroid  
   - Channel index → vertebra label  

## Tailored Loss Functions

A key contribution of this work is the design of loss functions that handle incomplete labels.

### Standard heatmap loss (baseline)

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{j=1}^{L} \sum_{i=1}^{N} (y_{ij} - \hat{y}_{ij})^2
$$

### Problem
- Penalises predictions for unlabelled vertebrae
- Suppresses valid detections  

### Proposed approach

- Ignore missing annotations per image  
- Handle vertebrae never labelled in dataset  
- Apply weighting based on annotation availability

## 1. Clone Repository

    git clone [https://github.com/RedmondLabUCD/vertebra-identification.git](https://github.com/RedmondLabUCD/vertebra-identification.git)

## 2. Data Preparation

The dataset used in this work is unfortunately not available publicly. However, all code for data preparation is provided to create the heatmaps and extract the images from original DICOM files. 

Split data into folds:

    split_data()

Create all data files needed (regions of interest, heatmaps, etc.):

    create_dataset()

## 3. Train model and test on reserved test dataset.

To train and test the final coarse and fine models, use:

    %run final_training.py <model_name>
    %run final_test.py <model_name>

Process results of coarse model and calculate accuracy:

    final_lm_preds_postprocess(model_name=<model_name>)
    calculate_percentage_box(model_name==<model_name>,size=<ROI width in pixels>)
