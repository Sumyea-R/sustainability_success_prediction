import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import os
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_3_labels.csv")
output_file = os.path.join(data_clean_folder, "dealroom_clean_features_final.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# Make sure ID is a column and not just an index, as we need it for deduplication logic  
if 'ID' not in df.columns and df.index.name == 'ID':
    df.reset_index(inplace=True)
elif 'ID' not in df.columns and 'ID' in df.index.names:
    df.reset_index(inplace=True)

print(f"Entries before removing duplicates: {len(df)}")

# Replace slashes with dashes in date columns to help with sorting, and create temporary columns for deduplication logic
df['Tmp_Date_Str'] = df['Last funding date'].astype(str).str.replace('/', '-', regex=False)
df['Tmp_Date'] = pd.to_datetime(df['Tmp_Date_Str'], errors='coerce')

# Convert funding to numeric
df['Tmp_Funding'] = pd.to_numeric(df['Total funding (USD M)'], errors='coerce').fillna(0)

# Sort by date and funding to keep the most recent and highest funded entry when dropping duplicates
df.sort_values(by=['Tmp_Date', 'Tmp_Funding'], ascending=[False, False], inplace=True)

# ID must be unique 
if 'ID' in df.columns:
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)

# Name, Launch year and country combination must be unique to identify a startup as unique
if {'Name', 'Launch year'}.issubset(df.columns):
    df.drop_duplicates(subset=['Name', 'Launch year', 'HQ country'], keep='first', inplace=True)


# Drop temporary columns used for deduplication
df.drop(columns=['Tmp_Date', 'Tmp_Date_Str', 'Tmp_Funding'], inplace=True)

# Set index back to ID and Name
cols_to_index = [c for c in ['ID', 'Name'] if c in df.columns]
if cols_to_index:
    df.set_index(cols_to_index, inplace=True)

print(f"Entries after removing duplicates: {len(df)}")




def count_list_items(val):
    if pd.isna(val) or str(val).strip() == '': return 0
    return len(str(val).split(';'))

def get_year_from_date(val):
    if pd.isna(val): 
        return np.nan
    
    match = re.search(r'\d{4}', str(val))
    if match: 
        return int(match.group(0))
    
    return np.nan

def binary_map(val):
    s = str(val).strip().lower()
    if "yes" in s: 
        return 1
    
    return 0

def parse_multilabel_string(val):
    if pd.isna(val): 
        return []

    return [x.strip().lower() for x in str(val).split(';') if x.strip()]



def calc_gender_category(row):
    genders_str = str(row['Founders genders']).lower()
    
    # Handle missing/unknown values
    if pd.isna(genders_str) or genders_str.strip() in ['n/a', '']:
        return -1 
    
    # Parsing 
    parts = [x.strip() for x in genders_str.split(';') if x.strip()]
    
    if not parts:
        return -1
        
    female_count = 0
    male_count = 0
    valid_entries = 0
    
    for p in parts:
        if 'female' in p:
            female_count += 1
            valid_entries += 1
        elif 'male' in p:
            male_count += 1
            valid_entries += 1
            
    # Null or n/a entries after parsing 
    if valid_entries == 0:
        return -1

    if female_count > 0:
        return 1  # Diverse founding team (at least 1 female founder)
    else:
        return 0  # Male only
    



def calc_capital_intensity(row):
    funding = row['Funding_Total_USD']

    if funding == 0: 
        return 0

    start = row['Year_First_Funding']
    end = row['Year_Last_Funding']

    if pd.isna(start) or pd.isna(end): 
        return funding

    duration = end - start

    # If only one date, total = intensity
    if duration < 1: 
        duration = 1

    return funding / duration


def encode_top_n_multilabel(dataframe, column_name, prefix, top_n=20):
    lists = dataframe[column_name].apply(parse_multilabel_string)
    
    # Find top 20 most common items
    all_items = [item for sublist in lists for item in sublist]
    top_items = [item for item, count in Counter(all_items).most_common(top_n)]
    
    # Filter lists to only include top items
    def filter_top(l):
        return [i for i in l if i in top_items]
    
    filtered_lists = lists.apply(filter_top)
    
    # Binarize the filtered lists
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(filtered_lists)
    
    # Clean column names: replace spaces and special chars
    cols = [f"{prefix}_{c.replace(' ', '_').replace('&', '').replace('__', '_')}" for c in mlb.classes_]
    return pd.DataFrame(encoded, columns=cols, index=dataframe.index)

# Parse client focus to extract b2b/b2c focus, allowing for multiple labels
def parse_client(val):
    raw = parse_multilabel_string(val)
    mapped = []
    for x in raw:
        if 'business' in x: mapped.append('b2b')
        elif 'consumer' in x: mapped.append('b2c')
    return list(set(mapped))

# Extract SDG numbers from SDG column, which may contain text like "Climate Action (#13)" - we want to extract the "13"
def extract_sdg_nums(val):
    if pd.isna(val): return []
    return re.findall(r'\b\d+\b', str(val))



def calc_months_since_funding(row, current_date):
    date_val = row['Last funding date']
    launch_val = row['Launch year']
    
    # If no last funding date, take founding year as fallback
    if pd.isna(date_val) or str(date_val).lower() in ['', 'n/a']:
        if pd.isna(launch_val): 
            return 999 # If still no launch year, return a large number to indicate unknown age
        try:
            # Parse launch year and calculate age in months
            launch_year = int(float(str(launch_val)))
            age_months = (current_date.year - launch_year) * 12
            return max(0, age_months)
        except:
            return 999

    # Parse last funding date with multiple strategies to handle different formats
    try:
        date_str = str(date_val)
        if '/' in date_str and any(c.isalpha() for c in date_str):
            date_str = date_str.replace('/', '-')

        last_funding = pd.to_datetime(date_str, errors='coerce')
        
        if pd.isna(last_funding):
             match = re.search(r'\d{4}', date_str)
             if match:
                 last_funding = datetime(int(match.group(0)), 1, 1)
             else:
                 # Fallback to launch year if no valid date found
                 launch_year = int(float(str(launch_val)))
                 return max(0, (current_date.year - launch_year) * 12)
        
        # Calculate months since last funding
        delta = current_date - last_funding
        months = int(delta.days / 30.44)
        return max(0, months) 
        
    except:
        return 999


# Counting number of items in multi-label columns 
df['Num_Investors'] = df['Investors names'].apply(count_list_items)
df['Num_Founders'] = df['Founders'].apply(count_list_items)
df['Num_SDGs'] = df['SDGs'].apply(count_list_items)

# Binary mapping for yes/no columns
df['Has_Serial_Founder'] = df['Is serial founder (yes/no)'].apply(binary_map)
df['Has_Top_Uni_Founder'] = df['Founder is from top university (yes/no)'].apply(binary_map)

# Calculate gender diversity
df['Gender_Category'] = df.apply(calc_gender_category, axis=1)
df_gender_ohe = pd.get_dummies(df['Gender_Category'], prefix='Gen', dtype=int)
df_gender_ohe.rename(columns={
    'Gen_-1': 'Gen_Unknown',
    'Gen_0':  'Gen_MaleOnly',
    'Gen_1':  'Gen_Diverse'
}, inplace=True)

# Calculate company age
current_year = 2026
df['Launch_Year_Num'] = pd.to_numeric(df['Launch year'], errors='coerce')
df['Company_Age'] = current_year - df['Launch_Year_Num'].fillna(current_year)

# Capital intensity calculation
df['Year_First_Funding'] = df['First funding date'].apply(get_year_from_date)
df['Year_Last_Funding'] = df['Last funding date'].apply(get_year_from_date)
df['Funding_Total_USD'] = pd.to_numeric(df['Total funding (USD M)'], errors='coerce').fillna(0)
df['Capital_Intensity'] = df.apply(calc_capital_intensity, axis=1)

# Avg Round Size
df['Total_Rounds'] = pd.to_numeric(df['Total rounds number'], errors='coerce').fillna(1) # avoid div/0
df['Avg_Round_Size'] = df['Funding_Total_USD'] / df['Total_Rounds']


# Encoding multi-label columns using MultiLabelBinarizer, limiting to top 20 and top 60 most common labels
df_ind = encode_top_n_multilabel(df, 'Industries', 'Ind', top_n=20)
df_sub = encode_top_n_multilabel(df, 'Sub industries', 'SubInd', top_n=60)

# Parsing and encoding client focus
mlb_cli = MultiLabelBinarizer()
df_client = pd.DataFrame(
    mlb_cli.fit_transform(df['Client focus'].apply(parse_client)),
    columns=[f"Client_{c}" for c in mlb_cli.classes_],
    index=df.index
)

# Revenue Model
mlb_rev = MultiLabelBinarizer()
df_rev = pd.DataFrame(
    mlb_rev.fit_transform(df['Revenue model'].apply(parse_multilabel_string)),
    columns=[f"Rev_{c.replace(' ', '_')}" for c in mlb_rev.classes_],
    index=df.index
)

# SDGs - extract numbers and encode as multi-label
mlb_sdg = MultiLabelBinarizer()
df_sdg = pd.DataFrame(
    mlb_sdg.fit_transform(df['SDGs'].apply(extract_sdg_nums)),
    columns=[f"SDG_{c}" for c in mlb_sdg.classes_],
    index=df.index
)


# SDG Core/Side Value
df_sdg_type = pd.get_dummies(df['SDG core or side value'], prefix='SDG_Type', dtype=int)

# HQ region one-hot encoding
df_region = pd.get_dummies(df['HQ region'], prefix='Region', dtype=int)

# OHE top 40 countries, rest as "Other" 
top_countries = df['HQ country'].value_counts().nlargest(40).index
df['Country_Top'] = df['HQ country'].apply(lambda x: x if x in top_countries else 'Other')
df_country = pd.get_dummies(df['Country_Top'], prefix='Country', dtype=int)


# Create a feature indicating if funding is undisclosed (0 USD but has investors) as this might be a different signal than truly having 0 funding with no investors
df['Funding_Is_Undisclosed'] = ((df['Funding_Total_USD'] == 0) & (df['Num_Investors'] > 0)).astype(int)


# Calculate recency features based on last funding date and launch year, with fallbacks for missing data
curr_date = datetime.datetime(2026, 1, 1)
df['Months_Since_Last_Funding'] = df.apply(
    lambda row: calc_months_since_funding(row, curr_date), axis=1
)
# Calculate "overdue" flag - if it's been more than 24 months since last funding, we consider it overdue (potentially a "zombie" startup)
df['Is_Funding_Overdue'] = (df['Months_Since_Last_Funding'] > 24).astype(int)
new_recency_features = ['Months_Since_Last_Funding', 'Is_Funding_Overdue']



# Final feature set - we keep all numeric features and add the encoded categorical features

keep_numeric = [
    'Funding_Total_USD',
    'Funding_Is_Undisclosed',
    'Total_Rounds', 
    'Avg_Round_Size',
    'Capital_Intensity',
    'Company_Age',
    'Num_Investors',
    'Num_Founders',
    'Num_SDGs',
    'Has_Serial_Founder',
    'Has_Top_Uni_Founder',
    'IS_SUCCESS',
    'Months_Since_Last_Funding', 
    'Is_Funding_Overdue'
]



if 'IS_SUCCESS' not in df.columns:
    print("Warning: 'IS_SUCCESS' column not found in dataframe. Make sure it exists before running this script.")
    keep_numeric.remove('IS_SUCCESS')


X_final = pd.concat([
    df[keep_numeric],
    df_gender_ohe,
    df_ind,
    df_sub,
    df_client,
    df_rev,
    df_sdg,
    df_sdg_type,
    df_region,
    df_country
], axis=1)



X_final = X_final.fillna(0)

# Save final dataset
X_final.to_csv(output_file, sep=';')
print(f"Done, saved as {output_file}")

# Check lost information from industry and sub-industry encoding
lost_ind = (df_ind.sum(axis=1) == 0).mean()
lost_sub = (df_sub.sum(axis=1) == 0).mean()

print(f"Startups without Top-20 Industry: {lost_ind:.1%}")
print(f"Startups without Top-60 Sub-Industry: {lost_sub:.1%}")

print("Distribution of gender category:")
print(df['Gender_Category'].value_counts(normalize=True))

print(df["IS_SUCCESS"].value_counts(normalize=True))


# Coverage Check: How many startups are covered by the top countries vs. how many fall into the "long tail" of less common countries?
startups_in_top_countries = df['HQ country'].isin(top_countries).sum()
coverage_pct = startups_in_top_countries / len(df)
lost_pct = 1 - coverage_pct

print(f"--- Country Coverage Check ---")
print(f"Top countries cover: {coverage_pct:.1%}")
print(f"Startups in 'long tail': {lost_pct:.1%}")


print("-" * 30)
print("ANALYSIS: Recency & Overdue Features")
print("-" * 30)

# Analyze the distribution of the "Months_Since_Last_Funding" feature, especially looking at the impact of the fallback value of 999
real_months = df[df['Months_Since_Last_Funding'] < 900]['Months_Since_Last_Funding']

print(f"Stats for months without fallback strategy:")
print(real_months.describe())

# How many have the "unknown/fallback" value of 999, and what percentage is that?
unknown_count = (df['Months_Since_Last_Funding'] >= 900).sum()
print(f"\nNumber of 'unknown/fallback' (999): {unknown_count} ({unknown_count/len(df):.1%})")

# Overdue distribution
print("\Distribution 'Is_Funding_Overdue' (> 24 months):")
print(df['Is_Funding_Overdue'].value_counts(normalize=True))

# Cross-Check: Overdue vs. Success
print("\nCrosstab: Overdue vs. Success:")
print(pd.crosstab(df['Is_Funding_Overdue'], df['IS_SUCCESS'], normalize='index'))
print("-" * 30)