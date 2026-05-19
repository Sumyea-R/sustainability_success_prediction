import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "Dealroom_All_Companies_SDG.csv")
output_file_labels = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")
output_file_post_labels = os.path.join(data_clean_folder, "dealroom_post_labels.csv")
output_file_features = os.path.join(data_clean_folder, "dealroom_clean_features_improved_improved.csv")

print(f"Loading data from: {input_file}")
# Using low_memory=False because Dealroom data has mixed types in many columns
df = pd.read_csv(input_file, sep=';', low_memory=False)

# Safety Deduplication
if 'ID' in df.columns:
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    df.set_index('ID', inplace=True)

# Basic Cleaning of Launch Year
df["Launch year"] = pd.to_numeric(df["Launch year"], errors='coerce')
df["Launch year"] = df["Launch year"].fillna(df["Launch year"].median()).round().astype(int)

# Define Observation Year: Launch Year + 5
df['Observation_Year'] = df['Launch year'] + 5


# filter for companies where Observation_Year <= 2021.
df = df[df['Observation_Year'] <= 2021].copy()

print(f"Dataset size after horizon filtering: {len(df)}")

# ==============================================================================
# HELPER FUNCTIONS FOR PARSING
# ==============================================================================

def parse_dr_date(date_str):
    """Parses Dealroom dates like 'nov/2025', '2021-10-01', or just '2025'"""
    if pd.isna(date_str) or str(date_str).strip().lower() in ['n/a', '']:
        return None
    date_str = str(date_str).strip().lower()
    
    # Format: nov/2025
    if '/' in date_str:
        try:
            parts = date_str.split('/')
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = month_map.get(parts[0][:3], 1)
            year = int(parts[1])
            return datetime(year, month, 1)
        except:
            pass
            
    # Format: 2021-10-01
    if '-' in date_str:
        try:
            return pd.to_datetime(date_str)
        except:
            pass
            
    # Format: 2025
    if len(date_str) == 4 and date_str.isdigit():
        return datetime(int(date_str), 1, 1)
        
    return None

def get_year(dt):
    return dt.year if dt else None

def split_list(val, sep=';'):
    if pd.isna(val) or str(val).strip() == '':
        return []
    return [x.strip() for x in str(val).split(sep) if x.strip()]

def count_list_items(val):
    if pd.isna(val) or str(val).strip() == '': return 0
    return len(str(val).split(';'))

# ==============================================================================
# 2. TARGET LABELLING (POST-OBSERVATION)
# ==============================================================================
print("Labelling targets...")

def label_exit(row):
    """IS_EXIT_WITHIN_5Y_AFTER_OBS: 1 if IPO, acquisition, merger, buyout between obs_year and obs_year + 5"""
    obs_year = row['Observation_Year']
    exit_horizon = obs_year + 5
    
    round_types = split_list(row.get('Each round type', ''))
    round_dates = split_list(row.get('Each round date', ''))
    
    exit_keywords = ["IPO", "ACQUISITION", "MERGER", "BUYOUT"]
    
    for r_type, r_date_str in zip(round_types, round_dates):
        r_type = r_type.upper()
        if any(kw in r_type for kw in exit_keywords):
            r_dt = parse_dr_date(r_date_str)
            if r_dt and obs_year < r_dt.year <= exit_horizon:
                return 1
    return 0

def label_high_growth(row):
    """IS_HIGH_GROWTH_WITHIN_3Y_AFTER_OBS: 1 if CAGR >= 20% between obs_year and obs_year + 3"""
    obs_year = row['Observation_Year']
    growth_horizon = obs_year + 3
    
    # We use the yearly employee columns if available
    emp_col_base = "Employees ("
    # Find the column that starts with "Employees (2016" or similar
    emp_col = next((c for c in df.columns if c.startswith(emp_col_base)), None)
    
    if not emp_col or pd.isna(row[emp_col]):
        return 0
    
    parts = str(row[emp_col]).split(";")
    # The columns are 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026
    years = list(range(2016, 2027))
    year_to_idx = {y: i for i, y in enumerate(years)}
    
    if obs_year not in year_to_idx or growth_horizon not in year_to_idx:
        return 0
    
    try:
        emp_start_str = parts[year_to_idx[obs_year]].strip()
        emp_end_str = parts[year_to_idx[growth_horizon]].strip()
        
        if emp_start_str.lower() == 'n/a' or emp_end_str.lower() == 'n/a':
            return 0
            
        emp_start = float(emp_start_str)
        emp_end = float(emp_end_str)
        
        if emp_start <= 10: 
            return 0
            
        cagr = (emp_end / emp_start) ** (1/3) - 1
        if cagr >= 0.2:
            return 1
    except:
        pass
        
    return 0

df['IS_EXIT_WITHIN_5Y_AFTER_OBS'] = df.apply(label_exit, axis=1)
df['IS_HIGH_GROWTH_WITHIN_3Y_AFTER_OBS'] = df.apply(label_high_growth, axis=1)

# 3-class target
# 1: EXIT
# 2: HIGH_GROWTH (if not exit)
# 0: NONE (NOT_EXIT_OR_HIGH_GROWTH)
def combine_targets(row):
    if row['IS_EXIT_WITHIN_5Y_AFTER_OBS'] == 1:
        return 1
    if row['IS_HIGH_GROWTH_WITHIN_3Y_AFTER_OBS'] == 1:
        return 2
    return 0

df['TARGET_3CLASS'] = df.apply(combine_targets, axis=1)

# Export Labels
df[['Name', 'IS_EXIT_WITHIN_5Y_AFTER_OBS', 'IS_HIGH_GROWTH_WITHIN_3Y_AFTER_OBS', 'TARGET_3CLASS']].to_csv(output_file_labels, sep=';')
df.to_csv(output_file_post_labels, sep=';')

print("Labels created and saved.")

# ==============================================================================
# 3. FEATURE ENGINEERING (BEFORE OR AT OBSERVATION)
# ==============================================================================
print("Engineering features relative to observation year...")

def engineer_funding_features(row):
    obs_year = row['Observation_Year']
    launch_year = row['Launch year']
    
    round_types = split_list(row.get('Each round type', ''))
    round_amounts = split_list(row.get('Each round amount', ''))
    round_dates = split_list(row.get('Each round date', ''))
    
    # Filter for rounds before or at observation year
    valid_rounds = []
    for r_type, r_amt, r_date_str in zip(round_types, round_amounts, round_dates):
        r_dt = parse_dr_date(r_date_str)
        if r_dt and r_dt.year <= obs_year:
            # Clean amount
            try:
                amt = float(str(r_amt).replace('n/a', '0'))
            except:
                amt = 0.0
            valid_rounds.append({'type': r_type, 'amount': amt, 'date': r_dt})
    
    # Sort by date
    valid_rounds.sort(key=lambda x: x['date'])
    
    res = {}
    res['Has_Funding_By_Obs'] = 1 if len(valid_rounds) > 0 else 0
    res['Num_Rounds_By_Obs'] = len(valid_rounds)
    
    if valid_rounds:
        first_funding_dt = valid_rounds[0]['date']
        last_funding_dt = valid_rounds[-1]['date']
        
        # Months to first funding
        launch_dt = datetime(launch_year, 1, 1)
        months_to_first = (first_funding_dt.year - launch_dt.year) * 12 + (first_funding_dt.month - launch_dt.month)
        res['Months_To_First_Funding_By_Obs'] = max(0, months_to_first)
        
        # Months since last funding
        obs_dt = datetime(obs_year, 12, 31)
        months_since_last = (obs_dt.year - last_funding_dt.year) * 12 + (obs_dt.month - last_funding_dt.month)
        res['Months_Since_Last_Funding_At_Obs'] = max(0, months_since_last)
        
        res['First_Round_Type_By_Obs'] = valid_rounds[0]['type']
        res['Last_Round_Type_By_Obs'] = valid_rounds[-1]['type']
        
        # Total and Avg
        amts = [r['amount'] for r in valid_rounds if r['amount'] > 0]
        res['Total_Funding_By_Obs'] = sum(amts)
        res['Avg_Round_Size_By_Obs'] = np.mean(amts) if amts else 0
        
        # Growth (Last / First)
        if len(amts) >= 2 and amts[0] > 0:
            res['Round_Size_Growth_By_Obs'] = amts[-1] / amts[0]
        else:
            res['Round_Size_Growth_By_Obs'] = 1.0
            
        # Time between rounds
        if len(valid_rounds) >= 2:
            intervals = []
            for i in range(1, len(valid_rounds)):
                diff = (valid_rounds[i]['date'].year - valid_rounds[i-1]['date'].year) * 12 + \
                       (valid_rounds[i]['date'].month - valid_rounds[i-1]['date'].month)
                intervals.append(max(0, diff))
            res['Avg_Months_Between_Rounds_By_Obs'] = np.mean(intervals)
            res['Min_Months_Between_Rounds_By_Obs'] = np.min(intervals)
            res['Max_Months_Between_Rounds_By_Obs'] = np.max(intervals)
        else:
            res['Avg_Months_Between_Rounds_By_Obs'] = 0
            res['Min_Months_Between_Rounds_By_Obs'] = 0
            res['Max_Months_Between_Rounds_By_Obs'] = 0
            
        # Shares and Types
        types_str = " ".join([r['type'].lower() for r in valid_rounds])
        res['Has_Grant_By_Obs'] = 1 if 'grant' in types_str else 0
        res['Has_Debt_By_Obs'] = 1 if 'debt' in types_str else 0
        res['Has_VC_or_Equity_By_Obs'] = 1 if any(x in types_str for x in ['series', 'venture', 'seed', 'angel', 'equity']) else 0
        
        num_grant = sum(1 for r in valid_rounds if 'grant' in r['type'].lower())
        num_debt = sum(1 for r in valid_rounds if 'debt' in r['type'].lower())
        num_equity = sum(1 for r in valid_rounds if any(x in r['type'].lower() for x in ['series', 'venture', 'seed', 'angel', 'equity']))
        
        res['Share_Grant_Rounds_By_Obs'] = num_grant / len(valid_rounds)
        res['Share_Debt_Rounds_By_Obs'] = num_debt / len(valid_rounds)
        res['Share_Equity_or_VC_Rounds_By_Obs'] = num_equity / len(valid_rounds)
        
    else:
        res['Months_To_First_Funding_By_Obs'] = np.nan
        res['Months_Since_Last_Funding_At_Obs'] = np.nan
        res['First_Round_Type_By_Obs'] = 'n/a'
        res['Last_Round_Type_By_Obs'] = 'n/a'
        res['Total_Funding_By_Obs'] = 0
        res['Avg_Round_Size_By_Obs'] = 0
        res['Round_Size_Growth_By_Obs'] = 0
        res['Avg_Months_Between_Rounds_By_Obs'] = 0
        res['Min_Months_Between_Rounds_By_Obs'] = 0
        res['Max_Months_Between_Rounds_By_Obs'] = 0
        res['Has_Grant_By_Obs'] = 0
        res['Has_Debt_By_Obs'] = 0
        res['Has_VC_or_Equity_By_Obs'] = 0
        res['Share_Grant_Rounds_By_Obs'] = 0
        res['Share_Debt_Rounds_By_Obs'] = 0
        res['Share_Equity_or_VC_Rounds_By_Obs'] = 0
        
    return pd.Series(res)


def engineer_investor_features(row):
    obs_year = row['Observation_Year']

    # 1. Build name-to-type map for THIS company
    inv_names_total = split_list(row.get('Investors names', ''), sep=';')
    inv_types_total = split_list(row.get('Each investor type', ''), sep=';')

    name_to_type = {}
    for name, t_str in zip(inv_names_total, inv_types_total):
        name_to_type[name] = t_str.lower()

    # 2. Identify investors in rounds before/at Observation_Year
    round_investors_str = split_list(row.get('Each round investors', ''), sep=';')
    round_dates = split_list(row.get('Each round date', ''), sep=';')

    investors_at_obs = [] # Flat list of all participations
    unique_investors_at_obs = set()

    for r_inv_str, r_date_str in zip(round_investors_str, round_dates):
        r_dt = parse_dr_date(r_date_str)
        if r_dt and r_dt.year <= obs_year:
            # Round investors are separated by ++
            invs = [x.strip() for x in r_inv_str.split('++') if x.strip()]
            investors_at_obs.extend(invs)
            unique_investors_at_obs.update(invs)

    # 3. Calculate features
    mapping = {
        'corporate_venture_fund': 'Has_CVC_Investor_By_Obs',
        'corporate': 'Has_Corporate_Investor_By_Obs',
        'government': 'Has_Government_Investor_By_Obs',
        'university': 'Has_University_Investor_By_Obs',
        'accelerator': 'Has_Accelerator_or_Incubator_By_Obs',
        'incubator': 'Has_Accelerator_or_Incubator_By_Obs',
        'venture_capital': 'Has_VC_Investor_By_Obs',
        'angel': 'Has_Angel_Investor_By_Obs'
    }

    res = {v: 0 for v in set(mapping.values())}
    res['Has_Investor_Data_By_Obs'] = 1 if unique_investors_at_obs else 0
    res['Num_Unique_Investors_By_Obs'] = len(unique_investors_at_obs)

    # Repeat investor: if an investor name appears in more than one round before/at Obs
    inv_counts = Counter(investors_at_obs)
    res['Has_Repeat_Investor_By_Obs'] = 1 if any(c > 1 for c in inv_counts.values()) else 0

    found_types = set()
    for inv_name in unique_investors_at_obs:
        t_low = name_to_type.get(inv_name, '')
        for key, feat in mapping.items():
            if key in t_low:
                res[feat] = 1
                found_types.add(feat)

    res['Num_Investor_Types_By_Obs'] = len(found_types)
    return pd.Series(res)
# Applying Funding Features
df_funding = df.apply(engineer_funding_features, axis=1)
df = pd.concat([df, df_funding], axis=1)

# Applying Investor Features
df_investors = df.apply(engineer_investor_features, axis=1)
df = pd.concat([df, df_investors], axis=1)

def engineer_employee_features(row):
    obs_year = row['Observation_Year']
    
    emp_col = next((c for c in df.columns if c.startswith("Employees (2016")), None)
    if not emp_col or pd.isna(row[emp_col]):
        return pd.Series({'Emp_At_Obs': np.nan}) 
        
    parts = str(row[emp_col]).split(";")
    years = list(range(2016, 2027))
    year_to_idx = {y: i for i, y in enumerate(years)}
    
    def get_emp(y):
        if y in year_to_idx and year_to_idx[y] < len(parts):
            val = parts[year_to_idx[y]].strip()
            if val.lower() != 'n/a' and val != '':
                try: return float(val)
                except: return np.nan
        return np.nan

    res = {}
    res['Emp_At_Obs'] = get_emp(obs_year)
    res['Emp_1Y_Before_Obs'] = get_emp(obs_year - 1)
    res['Emp_2Y_Before_Obs'] = get_emp(obs_year - 2)
    res['Emp_3Y_Before_Obs'] = get_emp(obs_year - 3)
    
    # Growth
    if pd.notna(res['Emp_At_Obs']) and pd.notna(res['Emp_1Y_Before_Obs']) and res['Emp_1Y_Before_Obs'] > 0:
        res['Employee_Growth_1Y_PreObs'] = (res['Emp_At_Obs'] - res['Emp_1Y_Before_Obs']) / res['Emp_1Y_Before_Obs']
    else:
        res['Employee_Growth_1Y_PreObs'] = np.nan
        
    if pd.notna(res['Emp_At_Obs']) and pd.notna(res['Emp_2Y_Before_Obs']) and res['Emp_2Y_Before_Obs'] > 0:
        res['Employee_Growth_2Y_PreObs'] = (res['Emp_At_Obs'] - res['Emp_2Y_Before_Obs']) / res['Emp_2Y_Before_Obs']
    else:
        res['Employee_Growth_2Y_PreObs'] = np.nan

    # CAGR (3 year)
    if pd.notna(res['Emp_At_Obs']) and pd.notna(res['Emp_3Y_Before_Obs']) and res['Emp_3Y_Before_Obs'] > 0:
        res['Employee_CAGR_PreObs'] = (res['Emp_At_Obs'] / res['Emp_3Y_Before_Obs']) ** (1/3) - 1
    else:
        res['Employee_CAGR_PreObs'] = np.nan
        
    # Stats over all available years before/at Obs
    hist_vals = []
    for y in range(2016, obs_year + 1):
        v = get_emp(y)
        if pd.notna(v): hist_vals.append(v)
        
    res['Num_Employee_Observations_PreObs'] = len(hist_vals)
    res['Employee_Has_History_PreObs'] = 1 if len(hist_vals) > 1 else 0
    res['Employee_Max_PreObs'] = np.max(hist_vals) if hist_vals else np.nan
    res['Employee_Min_PreObs'] = np.min(hist_vals) if hist_vals else np.nan
    res['Employee_Volatility_PreObs'] = np.std(hist_vals) if len(hist_vals) > 1 else 0
    
    # Slope (simple linear regression)
    if len(hist_vals) > 1:
        x = np.arange(len(hist_vals))
        slope, _ = np.polyfit(x, hist_vals, 1)
        res['Employee_Slope_PreObs'] = slope
        
        # Acceleration (change in growth)
        if len(hist_vals) > 2:
            growths = [hist_vals[i] - hist_vals[i-1] for i in range(1, len(hist_vals))]
            accel = growths[-1] - growths[-2]
            res['Employee_Acceleration_PreObs'] = accel
        else:
            res['Employee_Acceleration_PreObs'] = 0
    else:
        res['Employee_Slope_PreObs'] = 0
        res['Employee_Acceleration_PreObs'] = 0
        
    return pd.Series(res)

def engineer_valuation_features(row):
    obs_year = row['Observation_Year']
    
    val_dates = split_list(row.get('Historical valuations - dates', ''), sep=';')
    val_values = split_list(row.get('Historical valuations - values (USD M)', ''), sep=';')
    
    valid_vals = []
    for d_str, v_str in zip(val_dates, val_values):
        dt = parse_dr_date(d_str)
        if dt and dt.year <= obs_year:
            try:
                # Handle range like "232-348" by taking mean
                if '-' in v_str:
                    low, high = v_str.split('-')
                    val = (float(low) + float(high)) / 2
                else:
                    val = float(v_str)
                valid_vals.append({'date': dt, 'value': val})
            except:
                pass
    
    valid_vals.sort(key=lambda x: x['date'])
    
    res = {}
    res['Has_Valuation_By_Obs'] = 1 if valid_vals else 0
    if valid_vals:
        res['Latest_Valuation_By_Obs'] = valid_vals[-1]['value']
        res['Years_Since_Last_Valuation_At_Obs'] = obs_year - valid_vals[-1]['date'].year
        
        if len(valid_vals) >= 2 and valid_vals[0]['value'] > 0:
            res['Valuation_Growth_PreObs'] = valid_vals[-1]['value'] / valid_vals[0]['value']
        else:
            res['Valuation_Growth_PreObs'] = 1.0
    else:
        res['Latest_Valuation_By_Obs'] = np.nan
        res['Years_Since_Last_Valuation_At_Obs'] = np.nan
        res['Valuation_Growth_PreObs'] = np.nan
        
    return pd.Series(res)

df_emp = df.apply(engineer_employee_features, axis=1)
df = pd.concat([df, df_emp], axis=1)

df_val = df.apply(engineer_valuation_features, axis=1)
df = pd.concat([df, df_val], axis=1)

# ==============================================================================
# 4. FOUNDER & MISSINGNESS FEATURES
# ==============================================================================
print("Engineering founder and missingness features...")

def count_in_list(val, keyword):
    if pd.isna(val) or str(val).strip() == '': return 0
    items = str(val).split(';')
    return sum(1 for x in items if keyword.lower() in x.lower())

def get_share(count, total):
    if total > 0: return count / total
    return 0

# Founder Features
df['Num_Founders'] = df['Founders'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)
df['Num_Serial_Founders'] = df['Is serial founder (yes/no)'].apply(lambda x: str(x).lower().count('yes'))
df['Share_Serial_Founders'] = df.apply(lambda r: get_share(r['Num_Serial_Founders'], r['Num_Founders']), axis=1)
df['Num_Top_Uni_Founders'] = df['Founder is from top university (yes/no)'].apply(lambda x: str(x).lower().count('yes'))
df['Share_Top_Uni_Founders'] = df.apply(lambda r: get_share(r['Num_Top_Uni_Founders'], r['Num_Founders']), axis=1)

# Missingness Features
df['Missing_Employee_Data'] = df.apply(lambda r: 1 if pd.isna(r['Emp_At_Obs']) else 0, axis=1)
df['Missing_Founder_Data'] = df['Founders'].isna().astype(int)
df['Missing_Investor_Data'] = (df['Has_Investor_Data_By_Obs'] == 0).astype(int)
df['Missing_SubIndustry'] = df['Sub industries'].isna().astype(int)
df['Missing_Founder_Education'] = df['Founders universities'].isna().astype(int)
df['Missing_Founder_Background'] = df['Founders backgrounds'].isna().astype(int)
df['Missing_Valuation'] = (df['Has_Valuation_By_Obs'] == 0).astype(int)
df['Missing_Funding_Round_Data'] = (df['Has_Funding_By_Obs'] == 0).astype(int)
df['Missing_Round_Amount_Data'] = df['Each round amount'].apply(lambda x: 1 if 'n/a' in str(x).lower() or pd.isna(x) else 0)
df['Missing_Round_Type_Data'] = df['Each round type'].isna().astype(int)

# ==============================================================================
# 5. CATEGORICAL ENCODING & FREQUENCY
# ==============================================================================
print("Encoding categoricals and frequency...")

# Frequency Encoding
sub_ind_counts = df['Sub industries'].value_counts()
df['SubIndustry_Frequency'] = df['Sub industries'].map(sub_ind_counts).fillna(0)

country_counts = df['HQ country'].value_counts()
df['Country_Frequency'] = df['HQ country'].map(country_counts).fillna(0)

# Region (One-Hot)
df_region = pd.get_dummies(df['HQ region'], prefix='Region', dtype=int)
df = pd.concat([df, df_region], axis=1)

# ==============================================================================
# 6. LOG TRANSFORMATIONS & PATENTS
# ==============================================================================
print("Applying log transformations...")

df['Number of patents'] = pd.to_numeric(df['Number of patents'], errors='coerce').fillna(0)
df['Log_Num_Patents'] = np.log1p(df['Number of patents'])
df['Log_Num_Investors'] = np.log1p(df['Num_Unique_Investors_By_Obs'])
df['Log_Founder_Prior_Funding'] = np.log1p(pd.to_numeric(df['Founders founded companies total funding'], errors='coerce').fillna(0))

# ==============================================================================
# 7. FINAL EXPORT
# ==============================================================================
print("Exporting final features...")

# Define the set of features to keep
feature_cols = [
    'Observation_Year', 'Launch year',
    'Has_Funding_By_Obs', 'Months_To_First_Funding_By_Obs', 'Num_Rounds_By_Obs',
    'Months_Since_Last_Funding_At_Obs', 'Total_Funding_By_Obs', 'Avg_Round_Size_By_Obs',
    'Round_Size_Growth_By_Obs', 'Avg_Months_Between_Rounds_By_Obs', 'Min_Months_Between_Rounds_By_Obs',
    'Max_Months_Between_Rounds_By_Obs', 'Has_Grant_By_Obs', 'Has_Debt_By_Obs',
    'Has_VC_or_Equity_By_Obs', 'Share_Grant_Rounds_By_Obs', 'Share_Debt_Rounds_By_Obs',
    'Share_Equity_or_VC_Rounds_By_Obs', 'First_Round_Type_By_Obs', 'Last_Round_Type_By_Obs',
    'Has_Investor_Data_By_Obs', 'Num_Investor_Types_By_Obs', 'Has_Corporate_Investor_By_Obs',
    'Has_Government_Investor_By_Obs', 'Has_University_Investor_By_Obs', 'Has_CVC_Investor_By_Obs',
    'Has_Accelerator_or_Incubator_By_Obs', 'Has_Angel_Investor_By_Obs', 'Has_VC_Investor_By_Obs',
    'Has_Repeat_Investor_By_Obs', 'Num_Unique_Investors_By_Obs',
    'Emp_At_Obs', 'Emp_1Y_Before_Obs', 'Emp_2Y_Before_Obs', 'Emp_3Y_Before_Obs',
    'Employee_Growth_1Y_PreObs', 'Employee_Growth_2Y_PreObs', 'Employee_CAGR_PreObs',
    'Employee_Slope_PreObs', 'Employee_Acceleration_PreObs', 'Employee_Volatility_PreObs',
    'Employee_Max_PreObs', 'Employee_Min_PreObs', 'Employee_Has_History_PreObs',
    'Num_Employee_Observations_PreObs', 'Has_Valuation_By_Obs', 'Latest_Valuation_By_Obs',
    'Years_Since_Last_Valuation_At_Obs', 'Valuation_Growth_PreObs',
    'Num_Founders', 'Num_Serial_Founders', 'Share_Serial_Founders', 'Num_Top_Uni_Founders',
    'Share_Top_Uni_Founders', 'Missing_Employee_Data', 'Missing_Founder_Data',
    'Missing_Investor_Data', 'Missing_SubIndustry', 'Missing_Founder_Education',
    'Missing_Founder_Background', 'Missing_Valuation', 'Missing_Funding_Round_Data',
    'Missing_Round_Amount_Data', 'Missing_Round_Type_Data', 'SubIndustry_Frequency',
    'Country_Frequency', 'Number of patents', 'Log_Num_Patents', 'Log_Num_Investors',
    'Log_Founder_Prior_Funding'
] + list(df_region.columns)


X_final = df[feature_cols].copy()

# Fill NaNs with 0 for most numeric features where appropriate, or leave for model (CatBoost handles NaNs)


X_final.to_csv(output_file_features, sep=';')

print(f"Done! Final feature set saved as {output_file_features}")
print(f"Final shape: {X_final.shape}")



