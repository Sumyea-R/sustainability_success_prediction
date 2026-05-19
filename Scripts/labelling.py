import os
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_all_pre_clean.csv")
output_file_labels = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")
output_file_post_labels = os.path.join(data_clean_folder, "dealroom_post_labels.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

df = df[df['Launch year'] <= 2021]
# Median imputation for Launch year column 
df["Launch year"] = df["Launch year"].fillna(df["Launch year"].median()).round()

# Drop exact ID duplicates
if 'ID' in df.columns:
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)

# Drop logical duplicates based on Name, Launch Year, and Country
if {'Name', 'Launch year', 'HQ country'}.issubset(df.columns):
    df.drop_duplicates(subset=['Name', 'Launch year', 'HQ country'], keep='first', inplace=True)

def is_pre_2022_exit(last_round_string, date_str, company_status):
    # Check if already dead/closed
    if pd.notna(company_status) and any(word in str(company_status).lower() for word in ['closed', 'low activity']):
        # If it's closed and the last activity was before 2022, drop it.
        pass 

    if pd.isna(last_round_string) or pd.isna(date_str):
        return False
        
    last_rounds = str(last_round_string).upper()
    dates = str(date_str).split(";")
    
    # Check if it was an exit
    is_exit_event = any(k in last_rounds for k in ["IPO", "ACQUISITION", "MERGER", "BUYOUT"])
    
    if is_exit_event:
        try:
            # Extract the year of the most recent event
            first_year = int(dates[0].split('/')[-1])
            # If the exit happened BEFORE 2022, flag for removal
            if first_year < 2022:
                return True
        except (ValueError, IndexError):
            return False
            
    return False



def analyze_exit(last_round_string, each_round_date):
    if pd.isna(last_round_string):
        return 0
    
    last_rounds = str(last_round_string).split(";")
    last_rounds = [x.strip().upper() for x in last_rounds]

    if not last_rounds:
        return 0
        
    def is_valid_date(date_str):
        if pd.isna(date_str):
            return False
        dates = str(date_str).split(";")
        if not dates:
            return False
        try:
            # Take the first date (e.g. 'nov/2025'), split by '/', and grab the year
            first_year = int(dates[0].split('/')[-1])
            return 2022 <= first_year <= 2026
        except (ValueError, IndexError):
            return False

    # Check for IPO
    full_text = ";".join(last_rounds)
    if "IPO" in full_text:
        # Check date before returning 1
        return 1 if is_valid_date(each_round_date) else 0
    
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
            # Check date before returning 1
            return 1 if is_valid_date(each_round_date) else 0
        
    return 0


def calculate_employee_lifetime_cagr(row):
    employees_col = "Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026)"

    raw_employee_string = str(row[employees_col]) if pd.notna(row[employees_col]) else ""
    parts = raw_employee_string.split(";")
    
    emp_2022_str = parts[6] if len(parts) > 6 else "1"
    emp_2025_str = parts[9] if len(parts) > 9 else "1"

    # Safely convert strings to numbers. If it hits "n/a", it falls back to 0.
    try:
        emp_2022 = float(emp_2022_str)
        emp_2025 = float(emp_2025_str)
    except ValueError:
        return 0 

    # Check for minimum threshold of 10 employees
    if emp_2022 <= 10:
        return 0

    num_years = 3    
    try:
        cagr = (emp_2025 / emp_2022) ** (1 / num_years) - 1
    except ZeroDivisionError:
        return 0
    
    # 20% lifetime CAGR threshold
    if cagr >= 0.2: 
        return 1
        
    return 0


print("Filtering out startups that exited before 2022...")
df['Pre_2022_Exit'] = df.apply(
    lambda row: is_pre_2022_exit(row['Last round'], row['Each round date'], row['Company status']), 
    axis=1
)
print(f"Number of companies flagged for pre-2022 exit: {df['Pre_2022_Exit'].sum()}")

# Keep only the ones that did NOT exit before 2022
df = df[df['Pre_2022_Exit'] == False].copy()
df.drop(columns=['Pre_2022_Exit'], inplace=True)


print("Calculating high growth status for each company...")
df['IS_HIGH_GROWTH'] = df.apply(calculate_employee_lifetime_cagr, axis=1)

df['IS_EXIT'] = df.apply(
    lambda row: analyze_exit(row['Last round'], row['Each round date']), 
    axis=1
)


print("Determining active status for each company...")
df["IS_ACTIVE"] = ~df["Company status"].astype(str).str.contains("closed|low-activity", case=False, na=False)

# Final target variables
df['IS_EXIT'] = (df['IS_EXIT'] == 1).astype(int)
df['IS_HIGH_GROWTH'] = ((df['IS_HIGH_GROWTH'] == 1) & (df['IS_ACTIVE'] == 1)).astype(int)


# Check 
print("Breakdown:")
print(f"High Growth (CAGR):    {df['IS_HIGH_GROWTH'].sum()}")
print(f"Successful Exits:      {df['IS_EXIT'].sum()}")
print("-" * 30)
print("Class Distribution:")
print(df["IS_EXIT"].value_counts(normalize=True))
print(df["IS_HIGH_GROWTH"].value_counts(normalize=True))

# Export ID + Name + y
columns_to_export = ["ID", "Name", "IS_EXIT", "IS_HIGH_GROWTH"]
try:
    df.to_csv(output_file_labels, index=False, columns=columns_to_export, sep=';')
    print(f"Successfully exported cleaned data to: {output_file_labels}")
except Exception as e:
    print(f"Error exporting cleaned data: {e}")


# Export full dataset with all columns for reference
df.to_csv(output_file_post_labels, index=False, sep=';')
