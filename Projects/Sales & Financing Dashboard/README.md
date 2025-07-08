# Sales & Financing Dashboard

A streamlined, end-to-end solution for visualizing loan performance and user demographics—leveraging SQL (or CSV), a Python conversion script, and Tableau for rich, interactive insights.

---

## 🚀 Overview

Aggregate and analyze retail loan data to monitor key metrics—approved amounts, repayment behavior, and applicant profiles—within an elegant Tableau dashboard. This repository provides both SQL and CSV workflows, a Python helper script, and a ready-to-publish Tableau workbook.

---

## 🎯 Objectives

* **Data Preparation**: Define schema and sample data in `loans.sql`, with CSV fallback via `sql_to_csv.py`.
* **Metric Aggregation**: Compute total approvals, average repayment periods, on‑time rates, and delay counts.
* **Visual Storytelling**: Craft KPI cards and demographic charts for quick executive overviews.
* **Interactive Exploration**: Enable filters by region, loan type, repayment status, and delays.
* **Publication**: Package and share the dashboard via Tableau Public or as a `.twbx` file.

---

## ⚙️ Prerequisites

* **Tableau Desktop** (or Tableau Public) for visualization
* **Python 3.8+** with **pandas** (for SQL→CSV conversion)
* (Optional) **MySQL** or any ANSI‑compliant SQL engine for direct `.sql` import

```bash
pip install pandas
```

---

## 📁 Project Structure

```
financing-dashboard/
├── loans.sql                        # DDL + sample INSERTs
├── loans.csv                        # Generated flat file
├── sql_to_csv.py                    # SQL-to-CSV conversion script
├── Sales and Financing Dashboard.twb # Tableau workbook
└── README.md                        # Project documentation
```

---

## 🗃️ Dataset Schema

| Column            | Type        | Description                    |
| ----------------- | ----------- | ------------------------------ |
| loan\_id          | INT         | Unique loan identifier         |
| region            | VARCHAR(50) | e.g., North, South, East, West |
| loan\_type        | VARCHAR(50) | Home, Auto, Personal           |
| approved\_amount  | DECIMAL     | Loan amount                    |
| repayment\_period | INT         | Months to full repayment       |
| delayed\_payments | INT         | Count of missed installments   |
| applicant\_age    | INT         | Borrower age                   |
| applicant\_gender | VARCHAR(10) | Male, Female                   |

---

## 🏁 Getting Started

### 1. (Optional) SQL Workflow

```bash
mysql -u <user> -p dashboard_demo < loans.sql
```

**OR** skip to CSV.

### 2. Generate CSV

```bash
python sql_to_csv.py
```

This reads `loans.sql` and outputs `loans.csv` with proper headers.

### 3. Connect & Visualize in Tableau

1. Open **Tableau Desktop**.
2. **Connect** → **Text File** → select `loans.csv`.
3. Validate data types and field names.
4. Build worksheets (see Key Features).
5. Assemble a dashboard and expose filters.
6. Save as `.twbx` or publish to Tableau Public.

---

## 🌟 Key Features

1. **KPI Cards**

   * **Total Approved Amount**
   * **Average Repayment Period**
   * **On‑Time Repayment Rate (%)**
2. **Regional & Loan‑Type Analysis**

   * Sum of delayed payments by region
   * Loan volume & amounts by type
3. **Demographic Insights**

   * Gender distribution
   * Custom age‑group breakdown (Under 30, 30–45, Above 45)
4. **Interactive Filters**

   * Region, Loan Type, Delayed Payments, Repayment Period
5. **Live Refresh** (SQL mode)

   * Real‑time updates from `dashboard_demo` database
6. **Elegant Design**

   * Consistent color palette, clean typography, responsive layout

---

## 🛠️ Advanced Tips

* **Custom SQL**: In Tableau’s Data tab, use your own aggregate queries for optimized performance.
* **Calculated Fields**: Leverage Tableau to compute `OnTimeRate%` or dynamic age groups without altering source data.
* **Dashboard Actions**: Add highlight or URL actions to guide users to deeper content or external reports.

---

## 📅 Author

**Jaffar Hasan** — July 8, 2025
