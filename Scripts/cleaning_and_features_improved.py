import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold

# SETUP & DATA LOADING
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_post_labels.csv")
output_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# Make sure ID is a column for deduplication logic  
if 'ID' not in df.columns and df.index.name == 'ID':
    df.reset_index(inplace=True)
elif 'ID' not in df.columns and 'ID' in df.index.names:
    df.reset_index(inplace=True)

print(f"Entries before removing duplicates: {len(df)}")


# DEDUPLICATION
df['Tmp_Date_Str'] = df['Last funding date'].astype(str).str.replace('/', '-', regex=False)
df['Tmp_Date'] = pd.to_datetime(df['Tmp_Date_Str'], errors='coerce')
df['Tmp_Funding'] = pd.to_numeric(df['Total funding (USD M)'], errors='coerce').fillna(0)

df.sort_values(by=['Tmp_Date', 'Tmp_Funding'], ascending=[False, False], inplace=True)

if 'ID' in df.columns:
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)

if {'Name', 'Launch year', 'HQ country'}.issubset(df.columns):
    df.drop_duplicates(subset=['Name', 'Launch year', 'HQ country'], keep='first', inplace=True)

df.drop(columns=['Tmp_Date', 'Tmp_Date_Str', 'Tmp_Funding'], inplace=True)

cols_to_index = [c for c in ['ID', 'Name'] if c in df.columns]
if cols_to_index:
    df.set_index(cols_to_index, inplace=True)

print(f"Entries after removing duplicates: {len(df)}")

# HELPER FUNCTIONS
def count_list_items(val):
    if pd.isna(val) or str(val).strip() == '': return 0
    return len(str(val).split(';'))

def get_year_from_date(val):
    if pd.isna(val): return np.nan
    match = re.search(r'\d{4}', str(val))
    if match: return int(match.group(0))
    return np.nan

def binary_map(val):
    s = str(val).strip().lower()
    if "yes" in s: return 1
    return 0

def parse_multilabel_string(val):
    if pd.isna(val): return []
    return [x.strip().lower() for x in str(val).split(';') if x.strip()]

def calc_gender_category(row):
    genders_str = str(row['Founders genders']).lower()
    if pd.isna(genders_str) or genders_str.strip() in ['n/a', '']: return -1 
    
    parts = [x.strip() for x in genders_str.split(';') if x.strip()]
    if not parts: return -1
        
    female_count = sum(1 for p in parts if 'female' in p)
    valid_entries = sum(1 for p in parts if 'female' in p or 'male' in p)
    
    if valid_entries == 0: return -1
    return 1 if female_count > 0 else 0

def calc_capital_intensity(row):
    funding = row['Funding_Total_USD']
    if funding == 0: return 0
    start = row['Year_First_Funding']
    end = row['Year_Last_Funding']
    
    if pd.isna(start) or pd.isna(end): return funding
    duration = max(1, end - start)
    return funding / duration

def encode_top_n_multilabel_with_other(dataframe, column_name, prefix, top_n=20):
    lists = dataframe[column_name].apply(parse_multilabel_string)
    all_items = [item for sublist in lists for item in sublist]
    top_items = set(item for item, count in Counter(all_items).most_common(top_n))
    
    def filter_top_with_other(l):
        filtered = [i for i in l if i in top_items]
        if len(filtered) < len(l):
            filtered.append("Other")
        return filtered
    
    filtered_lists = lists.apply(filter_top_with_other)
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(filtered_lists)
    cols = [f"{prefix}_{str(c).replace(' ', '_').replace('&', '').replace('__', '_')}" for c in mlb.classes_]
    return pd.DataFrame(encoded, columns=cols, index=dataframe.index)

def parse_client(val):
    raw = parse_multilabel_string(val)
    mapped = []
    for x in raw:
        if 'business' in x: mapped.append('b2b')
        elif 'consumer' in x: mapped.append('b2c')
    return list(set(mapped))

def extract_sdg_nums(val):
    if pd.isna(val): return []
    return re.findall(r'\b\d+\b', str(val))

def calc_months_since_funding(row, current_date):
    date_val = row['Last funding date']
    launch_val = row['Launch year']
    
    if pd.isna(date_val) or str(date_val).lower() in ['', 'n/a']:
        if pd.isna(launch_val): return 999 
        try:
            launch_year = int(float(str(launch_val)))
            return max(0, (current_date.year - launch_year) * 12)
        except: return 999

    try:
        date_str = str(date_val)
        if '/' in date_str and any(c.isalpha() for c in date_str):
            date_str = date_str.replace('/', '-')

        last_funding = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(last_funding):
             match = re.search(r'\d{4}', date_str)
             if match: last_funding = datetime(int(match.group(0)), 1, 1)
             else:
                 launch_year = int(float(str(launch_val)))
                 return max(0, (current_date.year - launch_year) * 12)
        
        delta = current_date - last_funding
        return max(0, int(delta.days / 30.44)) 
    except: return 999

# OOF Target Encoding Helpers
def create_multilabel_target_encoder(df_train, col_name, target_col, m=10):
    global_mean = df_train[target_col].mean()
    exploded = df_train[[col_name, target_col]].explode(col_name)
    stats = exploded.groupby(col_name)[target_col].agg(['count', 'mean'])
    stats['smoothed_mean'] = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    return stats['smoothed_mean'].to_dict(), global_mean

def apply_multilabel_target_encoding(dataframe, col_name, encoding_map, global_mean):
    def get_average_encoding(sub_ind_list):
        if not isinstance(sub_ind_list, list) or len(sub_ind_list) == 0:
            return global_mean
        scores = [encoding_map.get(sub_ind, global_mean) for sub_ind in sub_ind_list]
        return np.mean(scores)
    return dataframe[col_name].apply(get_average_encoding)


# FEATURE ENGINEERING
df['Num_Investors'] = df['Investors names'].apply(count_list_items)
df['Num_Founders'] = df['Founders'].apply(count_list_items).clip(lower=1)
df['Num_SDGs'] = df['SDGs'].apply(count_list_items)

df['Has_Serial_Founder'] = df['Is serial founder (yes/no)'].apply(binary_map)
df['Has_Top_Uni_Founder'] = df['Founder is from top university (yes/no)'].apply(binary_map)

df['Gender_Category'] = df.apply(calc_gender_category, axis=1)
df_gender_ohe = pd.get_dummies(df['Gender_Category'], prefix='Gen', dtype=int)
df_gender_ohe.rename(columns={'Gen_-1': 'Gen_Unknown', 'Gen_0': 'Gen_MaleOnly', 'Gen_1': 'Gen_Diverse'}, inplace=True)

current_year = 2026
df['Launch_Year_Num'] = pd.to_numeric(df['Launch year'], errors='coerce')
df['Company_Age'] = current_year - df['Launch_Year_Num'].fillna(current_year)

df['Year_First_Funding'] = df['First funding date'].apply(get_year_from_date)
df['Year_Last_Funding'] = df['Last funding date'].apply(get_year_from_date)
df['Funding_Total_USD'] = pd.to_numeric(df['Total funding (USD M)'], errors='coerce').fillna(0)
df['Capital_Intensity'] = df.apply(calc_capital_intensity, axis=1)

df['Total_Rounds'] = pd.to_numeric(df['Total rounds number'], errors='coerce').fillna(1)
df['Avg_Round_Size'] = df['Funding_Total_USD'] / df['Total_Rounds']

df['Funding_Is_Undisclosed'] = ((df['Funding_Total_USD'] == 0) & (df['Num_Investors'] > 0)).astype(int)

curr_date = datetime(2026, 1, 1)
df['Months_Since_Last_Funding'] = df.apply(lambda row: calc_months_since_funding(row, curr_date), axis=1)


df['Has_patents'] = (df['Number of patents'] > 0).astype(int)
df['Num_patents'] = df['Number of patents'].fillna(0).clip(upper=(df['Number of patents'].quantile(0.99)))  


# CATEGORICAL ENCODING

# Industry (Top 20 + Other)
df_ind = encode_top_n_multilabel_with_other(df, 'Industries', 'Ind', top_n=20)


# Find the top 20 most frequent countries
top_countries = df['HQ country'].value_counts().nlargest(20).index
# Map the rest to 'Other'
df['Country_Top'] = df['HQ country'].apply(lambda x: x if x in top_countries else 'Other')
# One-Hot Encode
df_country = pd.get_dummies(df['Country_Top'], prefix='Country', dtype=int)

# Ensure target variable is numeric for encoding
df['IS_SUCCESS'] = pd.to_numeric(df['IS_SUCCESS'], errors='coerce').fillna(0)

# Sub-Industry (Out-of-Fold Target Encoding)
print("Applying Out-of-Fold Target Encoding for Sub-Industries...")
df['Sub_Industries_Parsed'] = df['Sub industries'].apply(parse_multilabel_string)
df['SubInd_Target_Encoded'] = np.nan

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(df):
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    encoding_map, fold_global_mean = create_multilabel_target_encoder(
        df_train, 'Sub_Industries_Parsed', 'IS_SUCCESS', m=10
    )
    val_encoded = apply_multilabel_target_encoding(
        df_val, 'Sub_Industries_Parsed', encoding_map, fold_global_mean
    )
    df.iloc[val_idx, df.columns.get_loc('SubInd_Target_Encoded')] = val_encoded.values

df.drop(columns=['Sub_Industries_Parsed'], inplace=True)

# Other categorical columns
mlb_cli = MultiLabelBinarizer()
df_client = pd.DataFrame(
    mlb_cli.fit_transform(df['Client focus'].apply(parse_client)),
    columns=[f"Client_{c}" for c in mlb_cli.classes_], index=df.index
)

mlb_rev = MultiLabelBinarizer()
df_rev = pd.DataFrame(
    mlb_rev.fit_transform(df['Revenue model'].apply(parse_multilabel_string)),
    columns=[f"Rev_{c.replace(' ', '_')}" for c in mlb_rev.classes_], index=df.index
)

mlb_sdg = MultiLabelBinarizer()
df_sdg = pd.DataFrame(
    mlb_sdg.fit_transform(df['SDGs'].apply(extract_sdg_nums)),
    columns=[f"SDG_{c}" for c in mlb_sdg.classes_], index=df.index
)

df_sdg_type = pd.get_dummies(df['SDG core or side value'], prefix='SDG_Type', dtype=int)
df_region = pd.get_dummies(df['HQ region'], prefix='Region_', dtype=int)


# LOG TRANSFORMATIONS & FINAL EXPORT

cols_to_log = ['Funding_Total_USD', 'Capital_Intensity', 'Avg_Round_Size', 'Num_patents']
for col in cols_to_log:
    df[f'Log_{col}'] = np.log1p(df[col])

keep_numeric = [
    'Log_Funding_Total_USD', 'Funding_Is_Undisclosed', 'Total_Rounds', 
    'Log_Avg_Round_Size', 'Log_Capital_Intensity', 'Company_Age',
    'Num_Investors', 'Num_Founders', 'Num_SDGs', 'Has_Serial_Founder',
    'Has_Top_Uni_Founder', 'IS_SUCCESS', 'Has_patents', 'Num_patents', 'SubInd_Target_Encoded'
]

X_final = pd.concat([
    df[keep_numeric], df_gender_ohe, df_ind, df_client, df_rev, 
    df_sdg, df_sdg_type, df_region, df_country
], axis=1)

X_final = X_final.fillna(0)
X_final.to_csv(output_file, sep=';')
print(f"Done, saved as {output_file}")

# EDA PLOTS
print("\n" + "="*40)
print("EDA & SANITY CHECKS")
print("="*40)

print("Distribution of gender category:")
print(df['Gender_Category'].value_counts(normalize=True))

print("\nDistribution of IS_SUCCESS:")
print(df["IS_SUCCESS"].value_counts(normalize=True))

print("-" * 30)
print("ANALYSIS: Recency & Overdue Features")
print("-" * 30)
real_months = df[df['Months_Since_Last_Funding'] < 900]['Months_Since_Last_Funding']
print(f"Stats for months without fallback strategy:\n{real_months.describe()}")
unknown_count = (df['Months_Since_Last_Funding'] >= 900).sum()
print(f"\nNumber of 'unknown/fallback' (999): {unknown_count} ({unknown_count/len(df):.1%})")


print("\n" + "-"*40)
inf_count = np.isinf(df[[f'Log_{c}' for c in cols_to_log]]).sum()
if inf_count.sum() > 0:
    print(f"WARNING: Infinite values detected!\n{inf_count}")
else:
    print("No infinite values detected. Log transformation seems safe.")

num_neg = (df['IS_SUCCESS'] == 0).sum()
num_pos = (df['IS_SUCCESS'] == 1).sum()
if num_pos > 0:
    print(f"\nXGBoost scale_pos_weight recommendation: {num_neg / num_pos:.2f}")

print("\nGenerating Visual EDA Dashboard...")
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Target Balance
sns.barplot(x=df['IS_SUCCESS'].value_counts(normalize=True).index, 
            y=df['IS_SUCCESS'].value_counts(normalize=True).values, ax=axes[0], palette='coolwarm')
axes[0].set_title('Target Class Imbalance (IS_SUCCESS)')
axes[0].set_ylabel('Proportion')

# Sub-Industry Target Encoded Distribution
sns.histplot(df['SubInd_Target_Encoded'], bins=30, kde=True, ax=axes[1], color='purple')
axes[1].set_title('Distribution of Sub-Industry\nTarget Encoded Values')
axes[1].set_xlabel('Encoded Success Probability')

# Target Encoded Boxplot vs Success
sns.boxplot(x='IS_SUCCESS', y='SubInd_Target_Encoded', data=df, ax=axes[2], palette='coolwarm')
axes[2].set_title('Predictive Power of\nSub-Industry Target Encoding')

plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 10))
cols_for_corr = ['Has_patents', 'Num_patents', 'SubInd_Target_Encoded', 'IS_SUCCESS']
correlations = df[cols_for_corr].corr()[['IS_SUCCESS']].sort_values(by='IS_SUCCESS', ascending=False).drop(index='IS_SUCCESS')
sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', cbar_kws={'label': 'Pearson Correlation'})
plt.title('Correlation of Key Engineered Features with Target')
plt.show()