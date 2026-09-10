import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve


# LOAD DATA
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

print(f"Loading dataset from: {input_file}")
df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])


# TIME-BASED SPLIT & LEAKAGE PREVENTION

tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

# Company_Age must be dropped from all X matrices to prevent target leakage
cols_to_drop = ['IS_SUCCESS', 'Company_Age']

#  Splits for Optuna (Age >= 9 for Training, Age == 8 for Validation)
X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=cols_to_drop)
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=cols_to_drop)
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

# Full Tuning Set for Final Model Training (Age >= 8)
X_tune_full = df_tune.drop(columns=cols_to_drop)
y_tune_full = df_tune['IS_SUCCESS']

# The Untouched Holdout Set (Age 5-7)
X_test = df_test.drop(columns=cols_to_drop)
y_test = df_test['IS_SUCCESS']

print("\nHyperparameter Tuning Setup (Chronological Split):")
print(f"  Training Fold (Age >= 9) : {len(X_train_opt)} startups")
print(f"  Validation Fold (Age == 8): {len(X_val_opt)} startups")
print(f"  Holdout Test Set (Age 5-7): {len(X_test)} startups\n")


# BAYESIAN OPTIMIZATION: RANDOM FOREST
print("="*50)
print("PHASE 1: OPTIMIZING RANDOM FOREST (50 Trials)")
print("="*50)

def objective_rf(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 40),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 15),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }
    
    rf = RandomForestClassifier(**param)
    rf.fit(X_train_opt, y_train_opt)
    preds = rf.predict(X_val_opt)
    return f1_score(y_val_opt, preds)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)

print("\n--- Best Random Forest Parameters ---")
print(study_rf.best_params)

# Train Final RF Model
best_rf_params = study_rf.best_params
best_rf_params.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42})
rf_best = RandomForestClassifier(**best_rf_params)
rf_best.fit(X_tune_full, y_tune_full)

# BAYESIAN OPTIMIZATION: XGBOOST
print("\n" + "="*50)
print("PHASE 2: OPTIMIZING XGBOOST (50 Trials)")
print("="*50)

scale_pos_weight_train = (y_train_opt == 0).sum() / (y_train_opt == 1).sum()

def objective_xgb(trial):
    param = {
        'n_estimators': 1000, 
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'scale_pos_weight': scale_pos_weight_train, 
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50 
    }
    
    xgb_model = XGBClassifier(**param)
    xgb_model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        verbose=False
    )
    
    trial.set_user_attr("best_n_estimators", xgb_model.best_iteration + 1)
    preds = xgb_model.predict(X_val_opt)
    return f1_score(y_val_opt, preds)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)

print("\n--- Best XGBoost Parameters ---")
print(study_xgb.best_params)
optimal_n_trees = study_xgb.best_trial.user_attrs["best_n_estimators"]
print(f"Optimal number of trees (via early stopping): {optimal_n_trees}")

# Train Final XGB Model
scale_pos_weight_full = (y_tune_full == 0).sum() / (y_tune_full == 1).sum()
best_xgb_params = study_xgb.best_params
best_xgb_params.update({
    'n_estimators': optimal_n_trees,
    'scale_pos_weight': scale_pos_weight_full,
    'n_jobs': -1, 
    'random_state': 42
})
xgb_best = XGBClassifier(**best_xgb_params)
xgb_best.fit(X_tune_full, y_tune_full)


# PERFORMANCE SCORECARD (HOLDOUT SET)

print("\n" + "="*50)
print("PHASE 3: EVALUATION ON UNTOUCHED HOLDOUT SET (Age 5-7)")
print("="*50)

rf_preds = rf_best.predict(X_test)
rf_probs = rf_best.predict_proba(X_test)[:, 1]

xgb_preds = xgb_best.predict(X_test)
xgb_probs = xgb_best.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")
print(classification_report(y_test, rf_preds))

print("\n--- XGBoost ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
print(classification_report(y_test, xgb_preds))

# Plot ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_probs):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC-AUC Curve: Predicting Startup Success')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_best, X_tune_full.columns, "Top 15 Features - Random Forest (Built-in)")
plot_feature_importance(xgb_best, X_tune_full.columns, "Top 15 Features - XGBoost (Built-in)")


# SHAP VALUES 

print("\n" + "="*50)
print("PHASE 4: SHAP VALUE INTERPRETATION")
print("="*50)

# XGBoost SHAP 
print("Running XGBoost SHAP (Sample = 1000)...")
X_test_xgb_sample = X_test.sample(n=1000, random_state=42)
explainer_xgb = shap.TreeExplainer(xgb_best)
shap_values_xgb = explainer_xgb(X_test_xgb_sample)

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - XGBoost (Predicting Success)")
shap.summary_plot(shap_values_xgb, X_test_xgb_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()

# Random Forest SHAP 
print("Running Random Forest SHAP (Sample = 300, Fast Approx)...")
X_test_rf_sample = X_test.sample(n=300, random_state=42)
explainer_rf = shap.TreeExplainer(rf_best)
shap_values_rf = explainer_rf.shap_values(
    X_test_rf_sample, 
    check_additivity=False, 
    approximate=True
)

if isinstance(shap_values_rf, list):
    shap_vals_class1 = shap_values_rf[1]
elif len(shap_values_rf.shape) == 3:
    shap_vals_class1 = shap_values_rf[:, :, 1]
else:
    shap_vals_class1 = shap_values_rf

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Random Forest (Predicting Success)")
shap.summary_plot(shap_vals_class1, X_test_rf_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()