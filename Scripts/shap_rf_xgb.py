import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])

leaky_columns = ['Months_Since_Last_Funding', 'Is_Funding_Overdue']
df = df.drop(columns=[col for col in leaky_columns if col in df.columns])


# Tuning Set (Age >= 8) -> For training the final models
tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()
X_tune_full = df_tune.drop(columns=['IS_SUCCESS'])
y_tune_full = df_tune['IS_SUCCESS']

# Holdout Test Set (Age 5-7) -> For SHAP explanations
test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()
X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']

# Train best performing models from previous tuning scripts on the full tuning set (Age >= 8)
print("Training final models...")

# 3A. Random Forest (Optuna Best)
rf_best = RandomForestClassifier(
    n_estimators=490, 
    max_depth=50, 
    min_samples_split=32, 
    min_samples_leaf=11, 
    max_features=0.409765423746428,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_best.fit(X_tune_full, y_tune_full)

# 3B. XGBoost (Optuna Best)
# Calculating precise scale_pos_weight for specific tuning set
scale_pos_weight_full = (y_tune_full == 0).sum() / (y_tune_full == 1).sum()

xgb_best = XGBClassifier(
    max_depth=4, 
    learning_rate=0.11309019783420737, 
    subsample=0.9637046596596964, 
    colsample_bytree=0.9804237799651304, 
    min_child_weight=6,
    n_estimators=211,
    scale_pos_weight=scale_pos_weight_full, 
    random_state=42,
    n_jobs=-1
)
xgb_best.fit(X_tune_full, y_tune_full)


# Built-in Feature Importance Plots (Top 15)
def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15 features
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

print("\nGenerating Built-in Feature Importance Plots...")
plot_feature_importance(rf_best, X_tune_full.columns, "Top 15 Features - Random Forest (Built-in)")
plot_feature_importance(xgb_best, X_tune_full.columns, "Top 15 Features - XGBoost (Built-in)")



print("\nCalculating SHAP values... (Using a 1,000-row sample to prevent freezing)")

# 1. Take a random sample of 1,000 startups from the test set. 
X_test_sampled = X_test.sample(n=1000, random_state=42)

# 2. XGBoost SHAP 
print("Running XGBoost SHAP...")
explainer_xgb = shap.TreeExplainer(xgb_best)
shap_values_xgb = explainer_xgb(X_test_sampled)

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - XGBoost (Predicting Success)")
shap.summary_plot(shap_values_xgb, X_test_sampled, max_display=15, show=False)
plt.tight_layout()
plt.show()

# 3. Random Forest SHAP 
print("Running Random Forest SHAP... (Using fast approximation)")

X_test_rf_sample = X_test.sample(n=300, random_state=42)

explainer_rf = shap.TreeExplainer(rf_best)
shap_values_rf = explainer_rf.shap_values(
    X_test_rf_sample, 
    check_additivity=False, 
    approximate=True
)


if isinstance(shap_values_rf, list):
    # Older SHAP versions: it's a list, so grab index 1
    shap_vals_class1 = shap_values_rf[1]
elif len(shap_values_rf.shape) == 3:
    # Newer SHAP versions: it's a 3D array (samples, features, classes)
    shap_vals_class1 = shap_values_rf[:, :, 1]
else:
    # Fallback just in case it's a binary array
    shap_vals_class1 = shap_values_rf


plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Random Forest (Predicting Success)")
shap.summary_plot(shap_vals_class1, X_test_rf_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()