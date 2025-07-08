import re
import csv

# Path to your loans.sql
sql_file = 'C:\\Users\\Crypto\\Documents\\GitHub\\Data-Analysis\\Projects\\Sales & Financing Dashboard\\loans.sql'
csv_file = 'C:\\Users\\Crypto\\Documents\\GitHub\\Data-Analysis\\Projects\\Sales & Financing Dashboard\\loans.csv'

# Read the .sql
with open(sql_file, 'r') as f:
    text = f.read()

# Extract only the values inside INSERT ... VALUES(...)
# This handles multiple rows in one statement.
values_block = re.search(r"INSERT INTO\s+loans\s+VALUES\s*(\(.+\));", text, re.S).group(1)

# Split into individual tuples
rows = re.findall(r"\((.*?)\)", values_block, re.S)

# Prepare CSV header
header = ['loan_id','region','loan_type','approved_amount',
          'repayment_period','delayed_payments','applicant_age','applicant_gender']

with open(csv_file, 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(header)
    for row in rows:
        # Convert SQL row to Python list, stripping quotes and spaces
        # Split on commas that are not inside quotes
        parts = re.findall(r"(?:'[^']*'|[^,])+?", row)
        clean = [p.strip().strip("'") for p in parts]
        writer.writerow(clean)

print(f"Created {csv_file}")
