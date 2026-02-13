import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_2.csv")
output_file_labels = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")
output_file_features = os.path.join(data_clean_folder, "dealroom_clean_features_3_labels.csv")



print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# Filter for launch years between 2016 and 2021 inclusive
# 2016: concrete data start
# 2021: requirement of survival for at least 5 years
df_clean = df[ (df['Launch year'] >= 2016) & (df['Launch year'] <= 2021) ].copy()

# Remaining entries after filtering
print(f" Remaining entries in data set after filtering: {len(df_clean)}")


def safe_parse_employees(employee_string):
    if pd.isna(employee_string) or str(employee_string).strip() == "":
        return []
    try:
        # Convert the string to a list of integers, ignoring non-numeric values
        return [int(x) for x in str(employee_string).split(";") if x.strip().isdigit()]
    except ValueError:
        return []

"""
def calculate_employee_cagr(row):
    launch_year = row['Launch year']

    # no launch year, cannot calculate growth
    if pd.isna(launch_year):
        return 0
    
    start_year = int(launch_year + 2)  # Start growth calculation 2 years after launch
    end_year = int(launch_year + 5)  # End growth calculation 5 years after launch

    col_name = "Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026)"
    employee_data = safe_parse_employees(row[col_name])

    if not employee_data:
        return 0

    start_idx = start_year - 2016 # Index for the start year, e.g. 2018 corresponds to index 2 (2018 - 2016)
    end_idx = end_year - 2016 

    if start_idx < 0 or end_idx >= len(employee_data):
        return 0

    
    # Calculate CAGR (Compound Annual Growth Rate: (end / start)^(1/number of years) - 1 )
    start_employees = employee_data[start_idx]
    end_employees = employee_data[end_idx]

    if start_employees < 10:
        # Ignore companies with less than 10 employees as suggested by OECD's definition of high growth companies
        # Prevent division by zero
        return 0

    if end_employees < start_employees:
        # Negative growth, not high growth
        return 0
    
    years = end_year - start_year 
    
    try:
        # Calculate CAGR
        cagr = (end_employees / start_employees) ** (1 / years) - 1
    except ZeroDivisionError:
        return 0

    if cagr >= 0.1: 
        # Company is considered growth if CAGR is 10% or higher
        return 1
    else:
        return 0
    """


def analyze_exit(last_round_string):

    if pd.isna(last_round_string):
        return 0
    last_rounds = last_round_string.split(";")
    last_rounds = [x.strip().upper() for x in last_rounds]

    if not last_rounds:
        return 0
    
    # Check for IPO
    # Data contains "POST IPO EQUITY", "SPAC IPO", "IPO" as indicators of an IPO exit, so we check for the presence of "IPO" in any of the last rounds.
    full_text = ";".join(last_rounds)
    if "IPO" in full_text:
        return 1
    
    # Check for M&A
    ma_keywords = [
        "ACQUISITION", 
        "MERGER",
        "BUYOUT",
    ]
    # Check for failure
    failure_keywords = [
        "BANKRUPTCY",
    ]

    # Is latest event a M&A?
    is_ma = any( k in last_rounds[0] for k in ma_keywords)
    if is_ma:
        # Check context (e.g. exclude acquisitons due to bankruptcy)
        recent_events = last_rounds[:3]  # Check the last 3 events for failure keywords
        is_failure = any( k in recent_events for k in failure_keywords)
        if not is_failure:
            return 1
        
    return 0

def check_scale_up(row):
    latest_emp_val = 0

    # Looking for number in 'Employees latest number' column first
    if 'Employees latest number' in row.index:
        val = pd.to_numeric(row['Employees latest number'], errors='coerce')
        if not pd.isna(val):
            latest_emp_val = val

    # If no valid number found, look into the historical employee data for the maximum value (as a backup)
    if latest_emp_val == 0:
        col_hist = "Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026)"
        if col_hist in row.index:
            try:
                hist_data = [int(x) for x in str(row[col_hist]).split(";") if x.strip().isdigit()]
                if hist_data:
                    latest_emp_val = max(hist_data)
            except:
                pass

    # Considering a company a scale up if it has >50 employees today
    if latest_emp_val >= 50:
        return 1
    
    return 0


def calculate_employee_cagr_dynamic(row):
    col = "Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026)"

    if col not in row.index:
        return 0
    
    raw_string = str(row[col])
    if pd.isna(raw_string) or raw_string.strip() == "": return 0

    try:
        # Converting n/a values to 0 (not necessary anymore after correct cleaning)
        data_list = []
        for x in raw_string.split(";"):
            x = x.strip()
            if x.isdigit():
                data_list.append(int(x))
            else:
                data_list.append(0)
    except:
        return 0
    
    first_val = 0
    first_year_idx = -1

    launch_year = row['Launch year']
    start_search_idx = 0
    if not pd.isna(launch_year): 
        start_search_idx = int(launch_year) - 2016

    # Finding the first non-zero value in the employee data list starting from the launch year
    for i in range(start_search_idx, len(data_list)):
        if data_list[i] > 0:
            first_val = data_list[i]
            first_year_idx = i
            break
    
    # If no value found or first value is 0, we cannot calculate growth
    if first_val == 0:
        return 0
    
    last_val = 0
    last_year_idx = -1

    # Looking for last value in Employees latest number column first (if it exists and is valid)
    latest_col = 'Employees latest number' 
    if latest_col in row.index and pd.notna(row[latest_col]) and row[latest_col] > 0:
        last_val = row[latest_col]
        last_year_idx = len(data_list) - 1

    else:
        # Fallback: Searching in list for the last non-zero value (starting from the end of the list and moving backwards)
        for i in range(len(data_list) - 1, first_year_idx, -1):
            if data_list[i] > 0:
                last_val = data_list[i]
                last_year_idx = i
                break

    # If no valid last value found, or if the last year index is not greater than the first year index, we cannot calculate growth
    if last_val == 0 or last_year_idx <= first_year_idx:
        return 0
    

    # Apply a minimum threshold of 10 employees to consider the company for growth calculation, as per OECD's definition of high growth companies. This also helps to avoid noise from very small companies where small absolute changes can lead to very high percentage growth rates.
    if first_val < 10:
        return 0
    
    # At least 3 years of growth data required to calculate CAGR (e.g. from 2018 to 2021), otherwise we cannot reliably calculate growth
    num_years = last_year_idx - first_year_idx
    if num_years < 3:
        return 0
    
    # Calculate dynamic CAGR based on first and last available employee numbers, regardless of the specific years they correspond to. This allows us to capture growth even for companies with missing data in some years, as long as we have a valid starting point and an endpoint.
    try:
        cagr = (last_val / first_val) ** (1 / num_years) - 1
    except:
        return 0
    
    # 20% CAGR threshold is a common benchmark for high growth companies 
    if cagr >= 0.20: 
        return 1
        
    return 0

print("Calculating high growth status for each company...")
# df_clean["IS_HIGH_GROWTH"] = df_clean.apply(calculate_employee_cagr, axis=1)
df_clean['IS_HIGH_GROWTH'] = df_clean.apply(calculate_employee_cagr_dynamic, axis=1)

print("Analyzing exit status for each company...")
df_clean["IS_EXIT"] = df_clean["Last round"].apply(analyze_exit) == 1

print("Determining active status for each company...")
df_clean["IS_ACTIVE"] = ~df_clean["Company status"].astype(str).str.contains(("closed|low-activity"), case=False, na=False)

print("Identifying scale-ups (backup for missing CAGR growth data)...")
# We consider a company to be a scale-up if it has at least 50 employees today, even if we cannot calculate historical growth due to missing data. This allows us to capture successful companies that may have incomplete employee history but are clearly established based on their current size.
df_clean["IS_SCALE_UP"] = df_clean.apply(check_scale_up, axis=1)


# Final target variable
# 1. Valid Exit: IPO or M&A (without bankruptcy context)
# 2. Active High Growth: Still active and classified as high growth based on employee growth
# 3. Scale-up: Still active and has at least 50 employees in the 5th year after launch (backup for missing growth data)
df_clean['IS_SUCCESS'] = (
    (df_clean['IS_EXIT'] == 1) | 
    ((df_clean['IS_ACTIVE'] == True) & (df_clean['IS_HIGH_GROWTH'] == 1)) |
    ((df_clean['IS_ACTIVE'] == True) & (df_clean['IS_SCALE_UP'] == 1))
).astype(int)

# Check 
print("Breakdown:")
print(f"High Growth (CAGR):    {df_clean['IS_HIGH_GROWTH'].sum()}")
print(f"Successful Exits:      {df_clean['IS_EXIT'].sum()}")
print(f"Scale-ups (>50 Emp):   {df_clean['IS_SCALE_UP'].sum()}")
print(f"TOTAL SUCCESSES (1):   {df_clean['IS_SUCCESS'].sum()}")
print("-" * 30)
print("Class Distribution:")
print(df_clean["IS_SUCCESS"].value_counts(normalize=True))



# Export ID + Name + y
columns_to_export = ["ID", "Name", "IS_SUCCESS"]
try:
    df_clean.to_csv(output_file_labels, index=False, columns=columns_to_export, sep=';')
    print(f"Successfully exported cleaned data to: {output_file_labels}")
except Exception as e:
    print(f"Error exporting cleaned data: {e}")


# Export full dataset with features and labels 
df_clean.to_csv(output_file_features, index=False, sep=';')
