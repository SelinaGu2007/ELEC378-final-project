# ELEC 378 Final Project: Butterfly and Moth Classification

Team Name: Model Makers

This repository contains the code for our ELEC 378 final project. The task is to classify butterfly and moth images into 100 species categories. We implemented both classical machine learning baselines and CNN-based models, and our best final model was a modified EfficientNet-B0 pipeline.

## Project Overview

The project compares two main types of approaches:

1. **Classical feature-based models**
   - Raw-pixel SVM baseline
   - Edge-feature SVM baseline
   - HOG + SVM
   - SIFT + Bag of Visual Words + kernel SVM

2. **CNN-based learned-feature models**
   - ResNet-style CNN baseline
   - Early EfficientNet-style CNN baseline
   - Final modified EfficientNet-B0 model

The purpose of including both types of models was to compare hand-engineered features with learned convolutional features on a fine-grained butterfly and moth classification task.

## Data and Rules

All models were trained only on the provided Kaggle dataset.

All CNN and EfficientNet models were trained from scratch.

### Classical Model

The classical-model code includes raw-pixel, edge-feature, HOG, and SIFT-based SVM pipelines. Among these, our main non-neural-network model was the **SIFT + Bag of Visual Words + kernel SVM** pipeline. This model extracts local SIFT descriptors, converts them into fixed-length BoVW histograms, and trains an SVM classifier. The code also includes kernel and hyperparameter comparisons used to select the final SIFT/SVM setting.

### CNN-Based Models

The CNN code includes ResNet and EfficientNet models trained from scratch. These models use image resizing, normalization, data augmentation, and cross-entropy-based training for 100-class classification. We tested multiple CNN variants and found that the EfficientNet-style model performed better.

### Best Model

The best final model is the **modified EfficientNet-B0 pipeline**. It was trained from scratch without pretrained weights and selected using validation performance. The final prediction code uses the best validation checkpoint and test-time augmentation to generate the Kaggle submission.

Final Kaggle private score: **0.97200**

## Summary

The main finding of the project is that learned CNN features substantially outperformed hand-engineered features. Classical models such as SIFT + BoVW + SVM were useful baselines, but the modified EfficientNet-B0 pipeline achieved the best final performance because it could learn color, texture, local markings, and global wing structure directly from the image data.
