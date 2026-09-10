import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_post_labels.csv")
output_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# Safety Deduplication
if 'ID' in df.columns:
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    df.set_index('ID', inplace=True)

# ==============================================================================
# 2. HORIZON FILTER (Strictly 2021)
# ==============================================================================
# Ensure we only look at startups launched by 2021
df["Launch year"] = df["Launch year"].fillna(df["Launch year"].median()).round()
df = df[df['Launch year'] <= 2021].copy()

# Replace Company_Age with Age_in_2021
df['Age_in_2021'] = 2021 - df['Launch year']

# ==============================================================================
# 3. TIME SERIES SPLITTING (Extracting 2016-2021)
# ==============================================================================
print("Parsing historical time series features up to 2021...")

def extract_years(row_val, start_year, end_year):
    if pd.isna(row_val):
        return [np.nan] * (end_year - start_year + 1)
    parts = str(row_val).split(";")
    res = []
    for i in range(end_year - start_year + 1):
        if i < len(parts) and parts[i].strip() not in ['n/a', '']:
            try:
                res.append(float(parts[i]))
            except ValueError:
                res.append(np.nan)
        else:
            res.append(np.nan)
    return res

# A. Employees
emp_col = "Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026)"
if emp_col in df.columns:
    emp_data = df[emp_col].apply(lambda x: extract_years(x, 2016, 2021))
    for i, year in enumerate(range(2016, 2022)):
        df[f'Emp_{year}'] = emp_data.apply(lambda x: x[i])

# B. Profit
profit_col = "Profit (USD) (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027)"
if profit_col in df.columns:
    profit_data = df[profit_col].apply(lambda x: extract_years(x, 2016, 2021))
    for i, year in enumerate(range(2016, 2022)):
        df[f'Profit_{year}'] = profit_data.apply(lambda x: x[i])

# C. Valuations
val_col = "Historical valuations - yearly since 2010 (USD M)"
if val_col in df.columns:
    val_data = df[val_col].apply(lambda x: extract_years(x, 2010, 2021))
    for i, year in enumerate(range(2010, 2022)):
        df[f'Valuation_{year}'] = val_data.apply(lambda x: x[i])

# ==============================================================================
# 4. POINT-IN-TIME FUNDING FEATURES
# ==============================================================================
def parse_first_funding(val):
    if pd.isna(val): return np.nan, np.nan
    try:
        parts = str(val).split('/')
        if len(parts) == 2: return int(parts[1]), int(parts[0]) # year, month
        if len(parts) == 1: return int(parts[0]), 1
    except:
        pass
    return np.nan, np.nan

df['FF_Year'], df['FF_Month'] = zip(*df['First funding date'].apply(parse_first_funding))

# Has funding by 2021
df['Has_Funding_by_2021'] = (df['FF_Year'] <= 2021).astype(int)

# Months to first funding (only if funded by 2021)
def calc_months_to_funding(row):
    if row['Has_Funding_by_2021'] == 1 and pd.notna(row['FF_Year']) and pd.notna(row['Launch year']):
        years_diff = row['FF_Year'] - row['Launch year']
        months = (years_diff * 12) + row['FF_Month']
        return max(0, months) # Avoid negatives if founded late in the year
    return np.nan

df['Months_to_First_Funding'] = df.apply(calc_months_to_funding, axis=1)

# ==============================================================================
# 5. BASE FEATURES (Safe non-leaking columns)
# ==============================================================================
def count_list_items(val):
    if pd.isna(val) or str(val).strip() == '': return 0
    return len(str(val).split(';'))

def binary_map(val):
    return 1 if "yes" in str(val).strip().lower() else 0

df['Num_Founders'] = df['Founders'].apply(count_list_items).clip(lower=1)
df['Num_SDGs'] = df['SDGs'].apply(count_list_items)
df['Has_Serial_Founder'] = df['Is serial founder (yes/no)'].apply(binary_map)
df['Has_Top_Uni_Founder'] = df['Founder is from top university (yes/no)'].apply(binary_map)

# ==============================================================================
# 6. CATEGORICAL ENCODING
# ==============================================================================
print("Encoding categoricals...")

def parse_multilabel_string(val):
    if pd.isna(val): return []
    return [x.strip().lower() for x in str(val).split(';') if x.strip()]

def encode_top_n_multilabel_with_other(dataframe, column_name, prefix, top_n=20):
    lists = dataframe[column_name].apply(parse_multilabel_string)
    all_items = [item for sublist in lists for item in sublist]
    top_items = set(item for item, count in Counter(all_items).most_common(top_n))
    
    def filter_top_with_other(l):
        filtered = [i for i in l if i in top_items]
        if len(filtered) < len(l): filtered.append("Other")
        return filtered
    
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(lists.apply(filter_top_with_other))
    cols = [f"{prefix}_{str(c).replace(' ', '_').replace('&', '').replace('__', '_')}" for c in mlb.classes_]
    return pd.DataFrame(encoded, columns=cols, index=dataframe.index)

# Top 20 Encodings
df_sub_ind = encode_top_n_multilabel_with_other(df, 'Sub industries', 'SubInd', top_n=20)
df_ind = encode_top_n_multilabel_with_other(df, 'Industries', 'Ind', top_n=20)

top_countries = df['HQ country'].value_counts().nlargest(20).index
df['Country_Top'] = df['HQ country'].apply(lambda x: x if x in top_countries else 'Other')
df_country = pd.get_dummies(df['Country_Top'], prefix='Country', dtype=int)

# Standard MultiLabel/Dummies
def parse_client(val):
    raw = parse_multilabel_string(val)
    mapped = []
    for x in raw:
        if 'business' in x: mapped.append('b2b')
        elif 'consumer' in x: mapped.append('b2c')
    return list(set(mapped))

mlb_cli = MultiLabelBinarizer()
df_client = pd.DataFrame(mlb_cli.fit_transform(df['Client focus'].apply(parse_client)), columns=[f"Client_{c}" for c in mlb_cli.classes_], index=df.index)

mlb_rev = MultiLabelBinarizer()
df_rev = pd.DataFrame(mlb_rev.fit_transform(df['Revenue model'].apply(parse_multilabel_string)), columns=[f"Rev_{c.replace(' ', '_')}" for c in mlb_rev.classes_], index=df.index)

def extract_sdg_nums(val):
    if pd.isna(val): return []
    return re.findall(r'\b\d+\b', str(val))

mlb_sdg = MultiLabelBinarizer()
df_sdg = pd.DataFrame(mlb_sdg.fit_transform(df['SDGs'].apply(extract_sdg_nums)), columns=[f"SDG_{c}" for c in mlb_sdg.classes_], index=df.index)

df_sdg_type = pd.get_dummies(df['SDG core or side value'], prefix='SDG_Type', dtype=int)
df_region = pd.get_dummies(df['HQ region'], prefix='Region_', dtype=int)

# ==============================================================================
# 7. DROPPING LEAKAGE & FINAL EXPORT
# ==============================================================================
# Explicitly removing features you flagged that leak future data/aggregates
cols_to_drop = [
    'Total funding (USD M)', 'Last round', 'Last funding amount', 
    'Last funding date', 'Employees latest', 'Investors names', 
    'Number of patents', 'Company_Age', 'Company status', 
    emp_col, profit_col, val_col, 'First funding date', 'FF_Year', 'FF_Month',
    'Sub industries', 'Industries', 'Client focus', 'Revenue model', 
    'SDGs', 'HQ country', 'HQ region', 'SDG core or side value', 'Country_Top', 'Founders'
]

# Base numeric features generated in this script
generated_numerics = [
    'Age_in_2021', 'Has_Funding_by_2021', 'Months_to_First_Funding', 
    'Num_Founders', 'Num_SDGs', 'Has_Serial_Founder', 'Has_Top_Uni_Founder'
]

# Gather the time series columns we just made
ts_cols = [c for c in df.columns if c.startswith(('Emp_', 'Profit_', 'Valuation_'))]

# Combine everything
X_final = pd.concat([
    df[generated_numerics + ts_cols], 
    df_ind, df_sub_ind, df_client, df_rev, df_sdg, df_sdg_type, df_region, df_country
], axis=1)

# Drop any remaining unneeded text/leakage columns that somehow snuck through
X_final.drop(columns=[c for c in cols_to_drop if c in X_final.columns], inplace=True, errors='ignore')

X_final.to_csv(output_file, sep=';')
print(f"Done! Clean, leakage-free features saved as {output_file}")
print(f"Final shape: {X_final.shape}")