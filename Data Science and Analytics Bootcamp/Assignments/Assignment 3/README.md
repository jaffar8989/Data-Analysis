# Data Analysis with Pandas Assignment

## Overview

This assignment explores core pandas functionalities through data exploration, manipulation, and analysis. The tasks are performed on a mock e-commerce dataset representing customer orders over 10 days.

## Objectives

- Practice basic DataFrame operations and statistics
- Perform column creation and arithmetic
- Apply sorting, ranking, and filtering techniques
- Use custom functions with apply()
- Work with indices and time-based data

## Prerequisites

- Python 3.6 or higher
- Pandas and NumPy installed (`pip install pandas numpy`)

## How to Run

1. Make sure you have `pandas` and `numpy` installed.
2. Open the `Pandas-Assignment.ipynb` Jupyter Notebook.
3. Run each cell in order to execute all tasks.

## Dataset Setup

A synthetic e-commerce dataset is created using NumPy and pandas with the following columns:

- `order_id` – Unique ID for each order  
- `customer_id` – Random customer identifier  
- `product_id` – Random product identifier  
- `quantity` – Units purchased (random 1–4)  
- `price` – Unit price (random float between $10 and $100)  
- `order_date` – Sequential dates from January 1, 2021  

## Tasks Covered

### === Task 1: Basic Data Exploration ===

**1.1** Display data types of each column  
**1.2** Calculate mean, min, and max for `quantity`  
**1.3** Check for missing values in the DataFrame

### === Task 2: Data Manipulation & Arithmetic ===

**2.1** Create new column `total_amount = quantity * price`  
**2.2** Calculate daily revenue (group by date)  
**2.3** Add 5% tax to prices in new column `price_with_tax`  
**2.4** Find orders with quantity above the mean

### === Task 3: Sorting & Ranking ===

**3.1** Sort DataFrame by `total_amount` (descending)  
**3.2** Rank orders by `price` (highest = 1)  
**3.3** Get top 3 orders by `total_amount`  
**3.4** Sort by `order_date` and then by `quantity`

### === Task 4: Function Application ===

**4.1** Define a function to classify `total_amount` as:  
- "High" > $200  
- "Medium" $100–$200  
- "Low" < $100  

**4.2** Apply the function to create `order_category`  
**4.3** Format `price` and `total_amount` as currency (2 decimals)  
**4.4** Calculate cumulative `total_amount` ordered by date

### === Task 5: Index Operations ===

**5.1** Set `order_date` as DataFrame index  
**5.2** Select orders from the first 5 days  
**5.3** Reset index to default  
**5.4** Create new DataFrame with `order_id` as the index

## Output Highlights

- Descriptive statistics and data types displayed  
- Revenue and totals computed and categorized  
- Orders ranked and sorted by value  
- Data indexed and sliced using time-based logic  
- Currency formatting applied for clarity

## Notes

- All code is commented and organized into sections  
- Uses pandas functions such as `describe()`, `groupby()`, `apply()`, `sort_values()`, `rank()`, `cumsum()`, and `reset_index()`  
- Dataset is randomly generated but reproducible due to `np.random.seed(2024)`

---

**Author:** Jaffar Hasan  
**Date:** March 7, 2025
