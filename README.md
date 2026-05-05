# Fruit Ripeness Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fruit-ripeness-detection-5shejsoa7lxw7x3ppqsbzu.streamlit.app/?embed_options=light_theme)

A comprehensive **Fruit Ripeness Detection** system that classifies fruits into **Unripe**, **Ripe**, and **Rotten** categories. The project has two parts:

- **Part 1**: Traditional Machine Learning with hand-crafted features  
- **Part 2**: Deep Learning using modern vision models (EfficientNet-B3, ViT-B16, Swin-Tiny)

---

## 🚀 Project Highlights

### Machine Learning Approach
- **99.14% Test Accuracy** with LightGBM (best model)
- Lightweight & interpretable using classical features (color + texture)
- Real-time inference via Streamlit web app

### Deep Learning Approach
- State-of-the-art vision transformers and CNNs:
  - **EfficientNet-B3**
  - **Vision Transformer (ViT-B16)**
  - **Swin Transformer (Swin-Tiny)**
- End-to-end learning directly from raw images
- Higher generalization potential on diverse fruit images

---

## 📊 Results Summary

### Machine Learning Results

| Rank | Model            | Accuracy | Precision | Recall  | F1-Score |
|------|------------------|----------|-----------|---------|----------|
| 1    | **LightGBM**     | **99.14%** | 99.15%    | 99.14%  | 99.15%   |
| 2    | XGBoost          | 99.09%   | 99.09%    | 99.09%  | 99.09%   |
| 3    | Random Forest    | 98.32%   | 98.32%    | 98.32%  | 98.31%   |
| 4    | Decision Tree    | 90.69%   | 90.66%    | 90.69%  | 90.66%   |
| 5    | KNN              | 86.79%   | 86.84%    | 86.79%  | 86.75%   |

**Best ML Model**: LightGBM 🏆

### Deep Learning Results (Part 2)
*(Update these with your actual metrics from `Fruit_Ripeness_DL_2.ipynb`)*

| Model             | Test Accuracy | Precision | Recall | F1-Score | Remarks                  |
|-------------------|---------------|-----------|--------|----------|--------------------------|
| EfficientNet-B3   | ...%          | ...       | ...    | ...      | Fast & efficient         |
| ViT-B16           | ...%          | ...       | ...    | ...      | Strong global attention  |
| Swin-Tiny         | ...%          | ...       | ...    | ...      | Hierarchical features    |

**Best DL Model**: *(to be filled based on your experiments)*

---

## 🛠️ Methodology

### 1. Dataset
- **Source**: [Kaggle - Fruit Ripeness: Unripe, Ripe and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
- **Classes**: `unripe`, `ripe`, `rotten`
- Fruits: Apple, Banana, Orange, etc.
- Pre-split train/test sets

### 2. Machine Learning Approach (Part 1)
- **Feature Engineering**:
  - Color: HSV/RGB statistics + histograms
  - Texture: Local Binary Patterns (LBP)
  - ~300–400 features per image
- Classical ML models (scikit-learn, XGBoost, LightGBM)

### 3. Deep Learning Approach (Part 2)
- Models trained using PyTorch + Hugging Face Transformers / timm
- Data augmentation, transfer learning from ImageNet
- Fine-tuning of EfficientNet-B3, ViT-B16, and Swin-Tiny
- GPU acceleration (Colab T4)

### 4. Deployment
- **Streamlit** web application for real-time prediction
- Supports image upload
- Shows predicted class + confidence score

---

## 🚀 Live Demo
Try the deployed app here:  
[**Fruit Ripeness Detector**](https://fruit-ripeness-detection-5shejsoa7lxw7x3ppqsbzu.streamlit.app/)


---

## 🏃 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Payal-mak/Fruit-Ripeness-Detection_ML.git
   cd Fruit-Ripeness-Detection_ML

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit app
   ```bash
   streamlit run app.py

## Technologies Used

- ML: scikit-learn, XGBoost, LightGBM, OpenCV, scikit-image
- DL: PyTorch, timm / Hugging Face Transformers, EfficientNet, ViT, Swin Transformer
- Deployment: Streamlit
- Visualization: Matplotlib, Seaborn

### Table 6: Deep Learning Model Performance on Test Set

| Model          | Test Accuracy | Val Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Speed (ms/img) |
|----------------|---------------|--------------|-------------|--------------------|-----------------|----------------|
| **Swin-Tiny**  | **99.88%**    | **99.93%**   | **0.9988**  | **0.9989**         | **0.9988**      | 5.82           |
| **EfficientNet-B3** | **99.85%** | **99.93%**   | **0.9985**  | **0.9984**         | **0.9985**      | 5.66           |
| **ViT-B16**    | 99.60%        | 99.86%       | 0.9960      | 0.9959             | 0.9960          | 5.52           |

---
