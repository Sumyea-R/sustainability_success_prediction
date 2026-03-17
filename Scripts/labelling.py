import os
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_pre_labels.csv")
output_file_labels = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")
output_file_post_labels = os.path.join(data_clean_folder, "dealroom_post_labels.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# Median imputation for Launch year column 
df["Launch year"] = df["Launch year"].fillna(df["Launch year"].median()).round()

def analyze_exit(last_round_string):
    if pd.isna(last_round_string):
        return 0
    
    last_rounds = str(last_round_string).split(";")
    last_rounds = [x.strip().upper() for x in last_rounds]

    if not last_rounds:
        return 0
    
    # Check for IPO
    full_text = ";".join(last_rounds)
    if "IPO" in full_text:
        return 1
    
    # Check for M&A
    ma_keywords = ["ACQUISITION", "MERGER", "BUYOUT"]
    failure_keywords = ["BANKRUPTCY"]

    # Is latest event an M&A?
    is_ma = any(k in last_rounds[0] for k in ma_keywords)
    if is_ma:
        # Check context properly for substrings in the last 3 events
        recent_events = last_rounds[:3]  
        is_failure = any(fail_word in event for event in recent_events for fail_word in failure_keywords)
        
        if not is_failure:
            return 1
        
    return 0

def calculate_employee_lifetime_cagr(row):
    employees_col = "Employees latest number"
    founders_col = "Founders"

    if employees_col not in row.index or founders_col not in row.index:
        return 0

    # Counting the number of founders based on the number of semicolons. Default to 1.
    raw_founders_string = str(row[founders_col])
    founders_num = raw_founders_string.count(";") + 1 if pd.notna(row[founders_col]) and raw_founders_string.strip() != "" else 1

    launch_year = row['Launch year']
    if pd.isna(launch_year) or launch_year == "":
        return 0

    # Check for minimum threshold of 10 employees
    if pd.notna(row[employees_col]) and row[employees_col] > 10:
        employees_num = row[employees_col]
    else:
        return 0

    # FIX: Use dynamic current year instead of hardcoding
    current_year = datetime.now().year
    num_years = current_year - launch_year
    
    if num_years < 5:
        return 0
    
    try:
        cagr = (employees_num / founders_num) ** (1 / num_years) - 1
    except ZeroDivisionError:
        return 0
    
    # 20% lifetime CAGR threshold
    if cagr >= 0.20: 
        return 1
        
    return 0

print("Calculating high growth status for each company...")
df['IS_HIGH_GROWTH'] = df.apply(calculate_employee_lifetime_cagr, axis=1)

print("Analyzing exit status for each company...")
df["IS_EXIT"] = df["Last round"].apply(analyze_exit) == 1

print("Determining active status for each company...")
df["IS_ACTIVE"] = ~df["Company status"].astype(str).str.contains("closed|low-activity", case=False, na=False)

# Final target variable
df['IS_SUCCESS'] = (
    (df['IS_EXIT'] == 1) | 
    ((df['IS_ACTIVE'] == True) & (df['IS_HIGH_GROWTH'] == 1))
).astype(int)

# Check 
print("Breakdown:")
print(f"High Growth (CAGR):    {df['IS_HIGH_GROWTH'].sum()}")
print(f"Successful Exits:      {df['IS_EXIT'].sum()}")
print(f"TOTAL SUCCESSES (1):   {df['IS_SUCCESS'].sum()}")
print("-" * 30)
print("Class Distribution:")
print(df["IS_SUCCESS"].value_counts(normalize=True))

# Export ID + Name + y
columns_to_export = ["ID", "Name", "IS_SUCCESS"]
try:
    df.to_csv(output_file_labels, index=False, columns=columns_to_export, sep=';')
    print(f"Successfully exported cleaned data to: {output_file_labels}")
except Exception as e:
    print(f"Error exporting cleaned data: {e}")


# Export full dataset with all columns for reference
df.to_csv(output_file_post_labels, index=False, sep=';')
