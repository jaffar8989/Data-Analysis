# Customer Churn Prediction - Machine Learning Project

## Overview

This project presents a comprehensive machine learning workflow for predicting customer churn in the telecommunications industry. Using a synthetic dataset of 5,000 customers, the notebook demonstrates data generation, exploratory analysis, feature engineering, model training, evaluation, and actionable business insights.

## Objectives

- Build an accurate predictive model to identify customers likely to churn
- Analyze key drivers of churn and segment high-risk groups
- Provide business recommendations for customer retention
- Compare multiple classification algorithms and select the best performer

## Prerequisites

- Python 3.8 or higher
- Required libraries:  
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn  
- Install dependencies with:  
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

## How to Run

1. Clone or download this repository.
2. Open the `Customer Churn Prediction.ipynb` notebook in Jupyter or VS Code.
3. Run each cell in order to execute all steps from data generation to business recommendations.

## Dataset Setup

A synthetic telecom dataset is generated with the following columns:

- `customerID` – Unique customer identifier  
- `gender`, `SeniorCitizen`, `Partner`, `Dependents` – Demographics  
- `tenure` – Months with the company  
- `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` – Service features  
- `Contract`, `PaperlessBilling`, `PaymentMethod` – Account and billing  
- `MonthlyCharges`, `TotalCharges` – Financial metrics  
- `Churn` – Target variable (Yes/No)

## Tasks Covered

### 1. Data Generation & Exploration

- Simulate a realistic telecom customer dataset
- Display dataset shape, columns, data types, and missing values
- Analyze churn distribution and numerical feature statistics

### 2. Data Visualization

- Visualize churn distribution and segment churn rates
- Explore relationships between churn and contract type, tenure, monthly charges, internet service, and senior citizen status

### 3. Data Preprocessing

- Encode categorical variables using label encoding
- Handle missing values and convert data types
- Scale numerical features for model compatibility

### 4. Feature Engineering

- Create new features: `ChargesPerTenure`, `HighCharges`, `LongTenure`
- Enhance predictive power with engineered variables

### 5. Model Training & Evaluation

- Train Logistic Regression, Random Forest, Gradient Boosting, and SVM classifiers
- Evaluate models using accuracy, precision, recall, F1-score, and AUC
- Perform cross-validation for robust performance estimates

### 6. Model Comparison & Selection

- Compare model metrics in a summary table
- Select the best model based on AUC and overall performance

### 7. Feature Importance & ROC Analysis

- Analyze feature importances for tree-based models
- Plot ROC curves for all models

### 8. Business Insights & Recommendations

- Segment customers by contract, tenure, and charges to identify high-risk groups
- Provide actionable recommendations for retention strategies

## Output Highlights

- Churn rate: **31.96%** (1,598 out of 5,000 customers)
- Best model: **Gradient Boosting** (AUC: 0.93, Accuracy: 88%, Precision: 83%, Recall: 74%)
- Key churn drivers: Month-to-month contracts, short tenure, high monthly charges, fiber optic service, senior citizens
- Estimated annual revenue protection: **$294,000** (saving ~350 customers at $70/month)

## Notes

- All code is modular, well-commented, and organized by analysis stage
- Random seed ensures reproducibility of results
- Business recommendations are based on actual segment churn rates from the analysis

---

**Author:** Jaffar Hasan  
**Date:** July 3,