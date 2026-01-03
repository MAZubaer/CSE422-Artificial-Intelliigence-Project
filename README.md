# CSE422-Artificial-Intelliigence-Project
Mushroom Edibility Classification: A Machine Learning Approach
## 🍄 Mushroom Edibility Classification: A Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A comprehensive machine learning project for accurately classifying mushrooms as **edible or poisonous** using morphological and ecological features. This repository includes **supervised and unsupervised learning models**, detailed data preprocessing, and performance evaluation to support safe foraging and mycological safety.

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 📖 Project Overview

Mushroom poisoning is a serious public health concern, especially in regions where wild mushrooms are consumed. This project leverages **machine learning** to automate and improve the accuracy of mushroom edibility classification, reducing reliance on subjective human judgment and minimizing health risks.

**Goal:** Develop a reliable ML model that can classify mushrooms as edible or poisonous with high accuracy using a dataset of morphological and ecological features.

## 📊 Dataset

- **Source:** Publicly available mushroom dataset
- **Instances:** 61,069 rows
- **Features:** 21 columns (mixed numerical and categorical)
- **Target:** Binary classification (`e` for edible, `p` for poisonous)
- **Key Features:** 
  - Numerical: `cap-diameter`, `stem-height`, `stem-width`
  - Categorical: `cap-shape`, `cap-color`, `gill-color`, `habitat`, `season`, etc.

## 🧠 Methodology

### 1. **Data Preprocessing**
- Removed features with ≥40% missing values
- Applied log transformation to handle skewness in numerical features
- Grouped rare categorical values (<5% frequency) into "Other"
- Used **One-Hot Encoding** for categorical variables and **StandardScaler** for numerical features
- Split data 80/20 with **stratified sampling** to maintain class distribution

### 2. **Supervised Learning Models**
We implemented and evaluated the following classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian & Bernoulli Naive Bayes
- XGBoost Classifier
- Neural Network (TensorFlow/Keras)

### 3. **Unsupervised Learning**
- Applied **K-Means Clustering** to log-transformed numerical features
- Used the **elbow method** to determine optimal clusters (k=2)
- Visualized clusters to explore natural groupings in the data

## 📈 Results

### Model Performance (Supervised)

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| SVM                 | 1.00     | 1.00      | 1.00   | 1.00     | 1.00 |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     | 1.00 |
| XGBoost             | 1.00     | 1.00      | 1.00   | 1.00     | 1.00 |
| Neural Network      | 0.998    | 0.998     | 0.998  | 0.998    | 1.00 |
| Decision Tree       | 0.99     | 0.99      | 0.99   | 0.99     | 0.99 |
| KNN                 | 1.00     | 1.00      | 1.00   | 1.00     | 1.00 |
| Logistic Regression | 0.71     | 0.71      | 0.71   | 0.71     | 0.70 |
| Gaussian NB         | 0.70     | 0.70      | 0.70   | 0.70     | 0.70 |

### Clustering Results (Unsupervised)
- K-Means revealed **two distinct clusters**, aligning with the binary nature of the classification problem
- Cluster visualizations confirmed meaningful separations in feature space

### Visualizations
- Confusion matrices for each model
- ROC curves and AUC scores
- Training/validation loss and accuracy curves (Neural Network)
- Clustering plots (2D feature projections)

## 🔧 Key Features

- **End-to-end ML pipeline** from data cleaning to model deployment-ready code
- **Comparison of 8+ classification algorithms** with detailed evaluation metrics
- **Hyperparameter tuning** for Neural Network and SVM
- **Unsupervised learning insights** with K-Means clustering
- **Comprehensive visualizations** for model interpretation
- **Stratified train-test split** to handle class imbalance
- **Log transformation and scaling** to normalize skewed distributions
