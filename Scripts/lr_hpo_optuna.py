import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# LOAD DATA
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

print(f"Loading dataset from: {input_file}")
df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])

# 2. TIME-BASED SPLIT & LEAKAGE PREVENTION

tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

cols_to_drop = ['IS_SUCCESS', 'Company_Age']

# Splits for Optuna
X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=cols_to_drop)
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=cols_to_drop)
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

# Full Tuning Set for Final Model
X_tune_full = df_tune.drop(columns=cols_to_drop)
y_tune_full = df_tune['IS_SUCCESS']

# Holdout Set
X_test = df_test.drop(columns=cols_to_drop)
y_test = df_test['IS_SUCCESS']

print("\nSplits:")
print(f"  Train Fold (Optuna): {len(X_train_opt)}")
print(f"  Val Fold (Optuna)  : {len(X_val_opt)}")
print(f"  Full Tune Set      : {len(X_tune_full)}")
print(f"  Holdout Test Set   : {len(X_test)}\n")


# SCALER SETUP
continuous_cols = [
    'Log_Funding_Total_USD', 'Total_Rounds', 'Log_Avg_Round_Size', 
    'Log_Capital_Intensity', 'Num_Investors', 'Num_Founders', 
    'Num_SDGs', 'Num_patents', 'SubInd_Target_Encoded'
]

cont_indices = [X_train_opt.columns.get_loc(c) for c in continuous_cols if c in X_train_opt.columns]
passthrough_indices = [i for i in range(len(X_train_opt.columns)) if i not in cont_indices]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])


# BAYESIAN OPTIMIZATION: LOGISTIC REGRESSION

print("OPTIMIZING LOGISTIC REGRESSION (50 Trials)")

def objective_lr(trial):
    # Search Space
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    
    l1_ratio = None
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
    lr = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        class_weight='balanced',
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    
    # Fit scaler strictly on training fold to prevent leakage into validation fold
    X_train_opt_scaled = preprocessor.fit_transform(X_train_opt)
    X_val_opt_scaled = preprocessor.transform(X_val_opt)
    
    lr.fit(X_train_opt_scaled, y_train_opt)
    preds = lr.predict(X_val_opt_scaled)
    
    return f1_score(y_val_opt, preds)

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=50)

print("\n--- Best Logistic Regression Parameters ---")
print(study_lr.best_params)


# TRAIN FINAL MODEL

# Re-fit scaler on the full tuning dataset
X_tune_full_scaled = preprocessor.fit_transform(X_tune_full)
X_test_scaled = preprocessor.transform(X_test)

best_lr_params = study_lr.best_params
if best_lr_params['penalty'] != 'elasticnet' and 'l1_ratio' in best_lr_params:
    del best_lr_params['l1_ratio']

best_lr_params.update({
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 2000,
    'random_state': 42,
    'n_jobs': -1
})

print("\nTraining final baseline model with best parameters...")
lr_best = LogisticRegression(**best_lr_params)
lr_best.fit(X_tune_full_scaled, y_tune_full)


# EVALUATION ON UNTOUCHED HOLDOUT SET

print("EVALUATION ON UNTOUCHED HOLDOUT SET (Age 5-7)")


preds = lr_best.predict(X_test_scaled)
probs = lr_best.predict_proba(X_test_scaled)[:, 1]

print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}\n")
print(classification_report(y_test, preds))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression Optuna (AUC = {roc_auc_score(y_test, probs):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC-AUC Curve: Baseline Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# FEATURE IMPORTANCE (COEFFICIENTS)
coefficients = lr_best.coef_[0]
feature_names = X_test.columns.tolist()

# Get top 15 absolute coefficients
abs_coefs = np.abs(coefficients)
top_indices = np.argsort(abs_coefs)[::-1][:15]

top_features = [feature_names[i] for i in top_indices]
top_coefs = coefficients[top_indices]

plt.figure(figsize=(10, 6))
# Color red if negative impact, green if positive impact
colors = ['red' if c < 0 else 'green' for c in top_coefs]
plt.barh(top_features[::-1], top_coefs[::-1], color=colors[::-1])
plt.title('Top 15 Feature Coefficients - Logistic Regression Baseline')
plt.xlabel('Coefficient Value (Impact on Log-Odds)')
plt.tight_layout()
plt.show()

# SHAP VALUES FOR LOGISTIC REGRESSION
# Initialize the LinearExplainer
explainer_lr = shap.LinearExplainer(lr_best, X_tune_full_scaled)

# Sample the holdout set for plotting
np.random.seed(42)
test_indices = np.random.choice(X_test_scaled.shape[0], min(1000, X_test_scaled.shape[0]), replace=False)

# Get the scaled data for the model to calculate SHAP values
X_test_lr_sample_scaled = X_test_scaled[test_indices]

# Get the unscaled data strictly for plotting the colors
X_test_lr_sample_unscaled = X_test.iloc[test_indices]

# Calculate SHAP values
shap_values_lr = explainer_lr.shap_values(X_test_lr_sample_scaled)

# Plot the results
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Logistic Regression (Baseline)")

# We pass the calculated SHAP values, but use the unscaled features for the color map
shap.summary_plot(
    shap_values_lr, 
    features=X_test_lr_sample_unscaled, 
    feature_names=X_test.columns.tolist(), 
    max_display=15, 
    show=False
)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("PHASE 4: SHAP VALUE INTERPRETATION")
print("="*50)