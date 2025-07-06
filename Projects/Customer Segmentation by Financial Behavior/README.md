# Customer Segmentation by Financial Behavior - Data Analytics Project

## Overview

This project performs an end-to-end customer segmentation analysis on microfinance users (Bede-like) by leveraging SQL for data aggregation, Python (K-Means) for clustering, and Tableau for interactive visualization. Using a synthetic dataset of transactions, the workflow demonstrates data loading, SQL-based feature extraction, cluster modeling, and business-driven insights.

## Objectives

* Segment microfinance customers based on transaction frequency, loan size, and repayment behavior
* Identify distinct customer profiles (e.g., high-frequency small loans vs. low-frequency long-term loans)
* Provide targeted microfinance plan recommendations and notification strategies per segment
* Showcase integration of SQL, Python, and Tableau in a seamless analytics pipeline

## Prerequisites

* **Python** 3.8 or higher
* **DBeaver** or any SQL client
* **Tableau Desktop** or **Tableau Public**
* Required Python libraries:

  * pandas
  * numpy
  * matplotlib
  * scikit-learn
* Install dependencies with:

  ```bash
  pip install pandas numpy matplotlib scikit-learn
  ```

## Repository Structure

```
├── transactions.csv             # Synthetic raw transaction data  
├── customer_features_raw.csv    # Aggregated features per customer  
├── customer_segments.csv        # K-Means cluster assignments  
├── segmentation_database.db     # SQLite database (for SQL practice)  
├── K-Means Clustering.py        # Python script performing scaling & clustering  
├── elbow_plot.png               # Elbow method chart for k selection  
└── README.md                    # Project documentation  
```

## How to Run

1. **Load Raw Data into SQL** (optional):

   * Use DBeaver to import `transactions.csv` into a database/table named `transactions`.
   * Run the aggregation SQL (see Section 2) to create `customer_features_raw`.
2. **Aggregate Features (Python alternative)**:

   ```python
   python aggregate_features.py  # reads transactions.csv, outputs customer_features_raw.csv
   ```
3. **Cluster Customers**:

   ```bash
   python "K-Means Clustering.py"  # reads customer_features_raw.csv, outputs customer_segments.csv & elbow_plot.png
   ```
4. **Visualize in Tableau**:

   * Open Tableau and connect to `customer_segments.csv`.
   * Build visualizations as per Section 4 of the project doc.

## Dataset Setup

* **`transactions.csv`**: Contains simulated transaction records with columns:

  * `customer_id` (int)
  * `txn_date` (YYYY-MM-DD)
  * `amount` (float)
  * `repayment_date` (YYYY-MM-DD)
* **`customer_features_raw.csv`**: Aggregated metrics per customer:

  * `txn_count`, `avg_txn_amount`, `total_amount`, `avg_repayment_days`
* **`customer_segments.csv`**: Includes `cluster` label (0–3) alongside feature columns.

## Tasks Covered

### 1. Data Aggregation (SQL & pandas)

* SQL: `WITH cust_agg AS (...) CREATE TABLE customer_features_raw AS SELECT * FROM cust_agg;`
* pandas alternative for CSV aggregation

### 2. K-Means Clustering in Python

* Data scaling with `StandardScaler`
* Elbow Method to determine optimal `k`
* Assign cluster labels and export results

### 3. Dashboarding in Tableau

* Scatter plot of `avg_txn_amount` vs. `txn_count`, colored by `cluster`
* Box plot of `avg_repayment_days` by `cluster`
* KPI cards, segment profiles, and recommended actions layout
* Optional live clustering via TabPy integration

## Output Highlights

* **Elbow Plot:** Visual guide to optimal cluster count
* **Cluster Profiles:** Four distinct user segments with summary metrics
* **Tableau Dashboard:** Interactive exploration of segmentation results

## Notes

* Synthetic data ensures privacy and reproducibility
* SQL syntax varies by engine; see comments for SQLite vs. MySQL/PostgreSQL
* Python scripts are modular and commented for clarity
* Tableau workbook not included but steps outlined for rapid dashboard creation

---

**Author:** Jaffar Hasan
**Date:** July 4, 2025
