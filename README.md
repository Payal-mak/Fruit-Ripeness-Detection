# Fruit Ripeness Detection Using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fruit-ripeness-detection-5shejsoa7lxw7x3ppqsbzu.streamlit.app/?embed_options=light_theme)  

A lightweight, accurate fruit ripeness classifier that distinguishes between **unripe**, **ripe**, and **rotten** fruits using only traditional machine learning techniques and hand-crafted features.

## 🚀 Project Highlights
- **99.14% Test Accuracy** using LightGBM (best model)
- Real-time predictions on fruit photo (Apple, Banana, Orange)
- Interactive web demo built with Streamlit

## 📊 Results Summary

| Rank | Model                        | Accuracy | Precision | Recall | F1-Score |
|------|------------------------------|----------|-----------|--------|----------|
| 1    | **LightGBM**                 | **99.14%** | 99.15%    | 99.14% | 99.15%   |
| 2    | XGBoost                      | 99.09%   | 99.09%    | 99.09% | 99.09%   |
| 3    | Random Forest                | 98.32%   | 98.32%    | 98.32% | 98.31%   |
| 4    | Decision Tree                | 90.69%   | 90.66%    | 90.69% | 90.66%   |
| 5    | KNN                          | 86.79%   | 86.84%    | 86.79% | 86.75%   |

**Best Model**: LightGBM with **99.14%** accuracy 🏆


## 🛠️ Methodology

### 1. Dataset
- Source: [Kaggle - Fruit Ripeness: Unripe, Ripe and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
- Classes: `unripe`, `ripe`, `rotten` (apples, bananas, oranges, etc.)
- Thousands of images in train/test splits

### 2. Feature Engineering (Classical Approach)
Hand-crafted features extracted from each image:
- **Color Features**:
  - Statistics (mean, std, percentiles) in HSV channels
  - Normalized histograms in HSV and RGB (32 bins each)
- **Texture Features**:
  - Uniform Local Binary Patterns (LBP) histogram
- Total: ~300–400 highly discriminative features per image

### 3. Models Evaluated
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM - RBF & Poly)
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- Logistic Regression (Multinomial)

### 4. Deployment
Interactive web app built with **Streamlit**:
- Upload any fruit photo
- Real-time feature extraction and prediction
- Displays class (Unripe/Ripe/Rotten) with confidence score

## 🚀 Live Demo
Try the deployed app here:  
[Fruit-Ripeness.streamlit.app](https://fruit-ripeness-detection-5shejsoa7lxw7x3ppqsbzu.streamlit.app/?embed_options=light_theme)  

## Run the Streamlit App
streamlit run app.py
