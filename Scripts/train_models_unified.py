import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
reports_folder = os.path.abspath(os.path.join(current_dir, "..", "Reports", "model_results"))
os.makedirs(reports_folder, exist_ok=True)

features_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved_improved.csv")
labels_file = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")

print(f"Loading features from: {features_file}")
df_features = pd.read_csv(features_file, sep=';')
df_labels = pd.read_csv(labels_file, sep=';')

# Merge features and targets (TARGET_3CLASS)
# 0: None, 1: EXIT, 2: HIGH_GROWTH
df = pd.merge(df_features, df_labels[['ID', 'TARGET_3CLASS']], on='ID', how='inner')

if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'], errors='ignore')

# Drop columns that shouldn't be features
cols_to_drop = ['TARGET_3CLASS', 'Observation_Year', 'Launch year']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['TARGET_3CLASS']

# Handle categorical feature indicators for CatBoost and XGBoost
cat_feature_names = [col for col in X.columns if X[col].dtype == 'object' or 'Has_' in col or 'Missing_' in col or 'Region_' in col]
print(f"Detected categorical features: {cat_feature_names}")

# Explicitly encode categories
for col in cat_feature_names:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        X[col] = X[col].fillna(0).astype(int)
    X[col] = X[col].astype('category')

cat_features = [i for i, col in enumerate(X.columns) if col in cat_feature_names]

# ==============================================================================
# 2. DATA SPLITTING & RESAMPLING
# ==============================================================================
# Since all companies are at the same stage (age 5), we use random split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print("Class Distribution in Training Before Resampling:")
print(y_train.value_counts(normalize=True))

# Resampling using SMOTENC + TomekLinks
print("\nApplying SMOTENC + TomekLinks resampling...")
# Fill NaNs first because imblearn algorithms cannot handle them
X_train_filled_pre_resample = X_train.fillna(0)

# Apply SMOTENC
smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)
X_train_res, y_train_res = smote_nc.fit_resample(X_train_filled_pre_resample, y_train)

# Apply Tomek Links
tomek = TomekLinks()
X_train_res, y_train_res = tomek.fit_resample(X_train_res, y_train_res)

print("Class Distribution in Training After Resampling:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# ==============================================================================
# 3. PREPROCESSING (For LR and NN)
# ==============================================================================
# Continuous columns for scaling
continuous_cols = [c for c in X.columns if not ('Has_' in c or 'Missing_' in c or 'Region_' in c or 'Is_' in c)]
cont_indices = [X.columns.get_loc(c) for c in continuous_cols]
passthrough_indices = [i for i in range(len(X.columns)) if i not in cont_indices]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])

# We'll fill NaNs with 0 for LR/NN/XGB for val/test (CatBoost handles them natively but we will use filled to be safe)
X_val_filled = X_val.fillna(0)
X_test_filled = X_test.fillna(0)

# Scale on the resampled, filled training set
X_train_scaled = preprocessor.fit_transform(X_train_res).astype('float32')
X_val_scaled = preprocessor.transform(X_val_filled).astype('float32')
X_test_scaled = preprocessor.transform(X_test_filled).astype('float32')

# Since data is physically balanced, weights are technically uniform now, but we compute just in case it's slightly off
classes = np.unique(y_train_res)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
class_weights_dict = {i: weights[i] for i in range(len(classes))}

# ==============================================================================
# 4. MODEL DEFINITIONS & TUNING
# ==============================================================================

def evaluate_model(model, X_t, y_t, name, is_nn=False, is_scaled=False):
    if is_nn:
        probs = model.predict(X_t, verbose=0)
        preds = np.argmax(probs, axis=1)
    elif is_scaled:
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)
    else:
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)
    
    print(f"\n--- {name} Evaluation ---")
    print(classification_report(y_t, preds))
    auc = roc_auc_score(y_t, probs, multi_class='ovr')
    print(f"ROC-AUC (OVR): {auc:.4f}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_t, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(reports_folder, f"cm_{name.replace(' ', '_')}.png"))
    plt.close()
    
    return {'f1_macro': f1_score(y_t, preds, average='macro'), 'auc': auc}

results = {}

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
study_lr = optuna.create_study(direction='maximize')
def objective_lr(trial):
    C = trial.suggest_float('C', 1e-3, 1e2, log=True)
    lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
    lr.fit(X_train_scaled, y_train_res)
    return f1_score(y_val, lr.predict(X_val_scaled), average='macro')
study_lr.optimize(objective_lr, n_trials=20)
lr_best = LogisticRegression(**study_lr.best_params, max_iter=1000)
lr_best.fit(X_train_scaled, y_train_res)
results['Logistic Regression'] = evaluate_model(lr_best, X_test_scaled, y_test, 'Logistic Regression', is_scaled=True)

# --- Random Forest ---
print("\nOptimizing Random Forest...")
def objective_rf(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'n_jobs': -1,
        'random_state': 42
    }
    rf = RandomForestClassifier(**param)
    rf.fit(X_train_res, y_train_res)
    return f1_score(y_val, rf.predict(X_val_filled), average='macro')

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=20)
rf_best = RandomForestClassifier(**study_rf.best_params, n_jobs=-1, random_state=42)
rf_best.fit(X_train_res, y_train_res)
results['Random Forest'] = evaluate_model(rf_best, X_test_filled, y_test, 'Random Forest')

# --- XGBoost ---
print("\nOptimizing XGBoost...")
def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective': 'multi:softprob',
        'num_class': 3,
        'enable_categorical': True,
        'random_state': 42
    }
    xgb_model = XGBClassifier(**param)
    xgb_model.fit(X_train_res, y_train_res)
    return f1_score(y_val, xgb_model.predict(X_val_filled), average='macro')

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=20)
xgb_best = XGBClassifier(**study_xgb.best_params, objective='multi:softprob', num_class=3, enable_categorical=True, random_state=42)
xgb_best.fit(X_train_res, y_train_res)
results['XGBoost'] = evaluate_model(xgb_best, X_test_filled, y_test, 'XGBoost')

# --- CatBoost ---
print("\nOptimizing CatBoost...")
def objective_cat(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'loss_function': 'MultiClass',
        'verbose': False,
        'random_state': 42
    }
    cat_model = CatBoostClassifier(**param)
    cat_model.fit(X_train_res, y_train_res, cat_features=cat_features, eval_set=(X_val_filled, y_val))
    return f1_score(y_val, cat_model.predict(X_val_filled), average='macro')

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=20)
cat_best = CatBoostClassifier(**study_cat.best_params, loss_function='MultiClass', verbose=False, random_state=42)
cat_best.fit(X_train_res, y_train_res, cat_features=cat_features)
results['CatBoost'] = evaluate_model(cat_best, X_test_filled, y_test, 'CatBoost')

# --- Neural Network ---
print("\nOptimizing Neural Network...")
def objective_nn(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = Sequential()
    model.add(Dense(trial.suggest_int('units_0', 64, 256), activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    for i in range(1, n_layers):
        model.add(Dense(trial.suggest_int(f'units_{i}', 32, 128), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_scaled, y_train_res, validation_data=(X_val_scaled, y_val), 
              epochs=30, batch_size=64, class_weight=class_weights_dict, callbacks=[early_stop], verbose=0)
    
    preds = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
    return f1_score(y_val, preds, average='macro')

study_nn = optuna.create_study(direction='maximize')
study_nn.optimize(objective_nn, n_trials=10) # NNs are slow, keeping trials low

# Train Final NN with best params
best_params = study_nn.best_params
final_nn = Sequential()
final_nn.add(Dense(best_params['units_0'], activation='relu', input_dim=X_train_scaled.shape[1]))
final_nn.add(BatchNormalization())
final_nn.add(Dropout(best_params['dropout_rate']))
for i in range(1, best_params['n_layers']):
    final_nn.add(Dense(best_params[f'units_{i}'], activation='relu'))
    final_nn.add(BatchNormalization())
    final_nn.add(Dropout(best_params['dropout_rate']))
final_nn.add(Dense(3, activation='softmax'))
final_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), 
                 loss='sparse_categorical_crossentropy')

final_nn.fit(X_train_scaled, y_train_res, validation_data=(X_val_scaled, y_val), 
             epochs=50, batch_size=64, class_weight=class_weights_dict, verbose=0)
results['Neural Network'] = evaluate_model(final_nn, X_test_scaled, y_test, 'Neural Network', is_nn=True)

# ==============================================================================
# 5. FINAL COMPARISON
# ==============================================================================
res_df = pd.DataFrame(results).T
print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
print(res_df)
res_df.to_csv(os.path.join(reports_folder, "model_comparison.csv"))

print(f"\nTraining completed! Results saved in {reports_folder}")
