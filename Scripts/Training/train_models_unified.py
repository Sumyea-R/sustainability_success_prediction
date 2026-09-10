import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# Append current folder to path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import models from separate files
from model_lr import train_lr
from model_rf import train_rf
from model_xgb import train_xgb
from model_cat import train_cat
from model_ffnn import train_ffnn
from model_nn import train_nn

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_clean_folder = os.path.join(project_root, "Data_Clean")
reports_folder = os.path.join(project_root, "Reports", "model_results")
os.makedirs(reports_folder, exist_ok=True)

features_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved_improved.csv")
labels_file = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")

print(f"Loading features from: {features_file}")
df_features = pd.read_csv(features_file, sep=';')
df_labels = pd.read_csv(labels_file, sep=';')

# Merge features and targets (TARGET_3CLASS)
df = pd.merge(df_features, df_labels[['ID', 'TARGET_3CLASS']], on='ID', how='inner')

if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'], errors='ignore')

# Keep only SDG_i_Combined and drop raw SDG_i and SDG_i_Keywords
cols_to_drop = []
for col in df.columns:
    if 'SDG_' in col and ('Keywords' in col or col.split('_')[-1].isdigit()):
        cols_to_drop.append(col)
print(f"Dropping redundant SDG columns: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Drop columns that shouldn't be features
base_drops = ['TARGET_3CLASS', 'Observation_Year', 'Launch year']
X = df.drop(columns=[c for c in base_drops if c in df.columns])
y = df['TARGET_3CLASS']

# Handle categorical feature indicators for CatBoost and XGBoost
cols_to_freq_encode = ['Sub industries', 'HQ country']
cat_feature_names = [col for col in X.columns if (X[col].dtype == 'object' or 'Has_' in col or 'Missing_' in col or 'Region' in col or 'SDG' in col) and col not in cols_to_freq_encode]
print(f"Detected categorical features for encoding: {cat_feature_names}")

# Explicitly encode categories
for col in cat_feature_names:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        X[col] = X[col].fillna(0).astype(int)
    X[col] = X[col].astype('category')

# ==============================================================================
# 2. DATA SPLITTING
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# Frequency encoding based on training split only to prevent fit leakage
print("Applying frequency encoding on training split only...")
for col in cols_to_freq_encode:
    freq = X_train[col].value_counts()
    
    # Map to splits and handle unseen categories by filling with 0
    X_train[f'{col.replace(" ", "")}_Frequency'] = X_train[col].map(freq).fillna(0)
    X_val[f'{col.replace(" ", "")}_Frequency'] = X_val[col].map(freq).fillna(0)
    X_test[f'{col.replace(" ", "")}_Frequency'] = X_test[col].map(freq).fillna(0)
    
    # Drop raw text columns
    X_train.drop(columns=[col], inplace=True)
    X_val.drop(columns=[col], inplace=True)
    X_test.drop(columns=[col], inplace=True)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print("Class Distribution in Training:")
print(y_train.value_counts(normalize=True))

# Update cat_features indices based on the new columns in X_train
cat_features = [i for i, col in enumerate(X_train.columns) if col in cat_feature_names]

# ==============================================================================
# 3. PREPROCESSING (For LR and NN)
# ==============================================================================
# Continuous columns for scaling
continuous_cols = [c for c in X_train.columns if not (c in cat_feature_names)]
cont_indices = [X_train.columns.get_loc(c) for c in continuous_cols]
passthrough_indices = [i for i in range(len(X_train.columns)) if i not in cont_indices]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])

# ==============================================================================
# 4. RUN PIPELINE FOR ALL MODELS
# ==============================================================================
results = {}

# 1. Logistic Regression
results['Logistic Regression'] = train_lr(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder)

# 2. Random Forest
results['Random Forest'] = train_rf(X_train, y_train, X_val, y_val, X_test, y_test, reports_folder)

# 3. XGBoost
results['XGBoost'] = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, reports_folder)

# 4. CatBoost
results['CatBoost'] = train_cat(X_train, y_train, X_val, y_val, X_test, y_test, cat_features, reports_folder)

# 5. NN (Classic MLP)
results['NN'] = train_nn(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder)

# 6. FFNN (Feed-Forward Neural Network)
results['FFNN'] = train_ffnn(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder)

# ==============================================================================
# 5. FINAL COMPARISON
# ==============================================================================
res_df = pd.DataFrame(results).T
print("\n" + "="*50)
print("FINAL MODEL COMPARISON (SDG_i_Combined only)")
print("="*50)
print(res_df)

comparison_file = os.path.join(reports_folder, "model_comparison.csv")
res_df.to_csv(comparison_file)
print(f"\nTraining completed! Results saved in {comparison_file}")
