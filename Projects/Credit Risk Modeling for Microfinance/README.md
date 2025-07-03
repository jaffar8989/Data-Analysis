# Credit Risk Modeling for Microfinance - Machine Learning Project

## Overview

This project implements a full machine learning workflow for credit risk modeling in the microfinance sector, with a focus on Shariah-compliant lending. Using a synthetic dataset of 5,000 microfinance clients, the system demonstrates data generation, SQL-based analytics, feature engineering, model training, evaluation, and actionable business recommendations.

## Objectives

- Build an accurate predictive model to identify high-risk microfinance clients
- Analyze key drivers of loan default and segment high-risk groups
- Provide Shariah-compliant recommendations for risk mitigation and portfolio optimization
- Demonstrate SQL analytics on the generated dataset

## Prerequisites

- Python 3.8 or higher
- Required libraries:  
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - sqlite3 (standard library)
- Install dependencies with:  
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

## How to Run

1. Clone or download this repository.
2. Open `Credit Risk Modeling.py` in VS Code or your preferred Python IDE.
3. Run the script. All steps from data generation to recommendations and visualizations will execute automatically.

## Dataset Setup

A synthetic microfinance dataset is generated with the following columns:

- `age`, `gender`, `education`, `household_size`, `location_type` – Demographics
- `business_type`, `seasonal_business` – Business characteristics
- `monthly_income`, `loan_amount`, `loan_term`, `assets_value`, `savings_balance` – Financial metrics
- `debt_to_income`, `payment_history`, `previous_loans`, `employment_duration` – Credit history
- `group_member`, `has_savings` – Social/financial inclusion
- `default` – Target variable (1 = default, 0 = repaid)

## Tasks Covered

### 1. Data Generation & SQL Analytics

- Simulate a realistic microfinance client dataset
- Store data in a SQLite database
- Run SQL queries for default rates by business type, education, and location

### 2. Feature Engineering

- Create new features: loan-to-income, payment-to-income, assets-to-loan, savings-to-income, experience score, stability score, high-risk business, vulnerable group

### 3. Data Preprocessing

- Encode categorical variables using label encoding
- Scale numerical features for model compatibility

### 4. Model Training & Evaluation

- Train a Logistic Regression model with hyperparameter tuning (GridSearchCV)
- Evaluate using ROC-AUC, precision, recall, F1-score, and classification report
- Display top 10 most important features

### 5. Visualization

- Plot ROC curve, precision-recall curve, confusion matrix, and feature importance

### 6. Shariah-Compliant Recommendations

- Analyze risk by business type and segment
- Generate actionable, Shariah-compliant lending and risk mitigation strategies
- Portfolio optimization metrics and recommendations

## Output Highlights

- Default rate and portfolio metrics for each business type
- SQL analytics for business, education, and location segments
- Model performance: ROC-AUC, precision, recall, F1-score
- Top features influencing default risk
- Visualizations for model evaluation
- Shariah-compliant recommendations for microfinance operations

## Notes

- All code is modular, well-commented, and organized by analysis stage
- Random seed ensures reproducibility of results
- SQL queries use double quotes for reserved keywords (e.g., `"default"`)
- Recommendations and risk analysis are based on actual segment statistics from the synthetic data

---

**Author:** Jaffar Hasan  
**Date:** July 3, 2025