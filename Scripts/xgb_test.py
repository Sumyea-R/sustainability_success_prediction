import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

X = df.drop(columns=['IS_SUCCESS'])
y = df['IS_SUCCESS']

if 'ID' in X.columns:
    X = X.drop(columns=['ID'])

# XGBoost doesn't like special characters (like <, >, [, ]) in feature names.
# This cleans them up automatically just in case.
X.columns = [re.sub(r'[\[\]<>]', '_', col) for col in X.columns]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class imbalance ratio for XGBoost
# scale_pos_weight = total_negative_examples / total_positive_examples
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# Initialize and train the XGBoost Classifier
print("Training the XGBoost model...")
xgb_model = xgb.XGBClassifier(n_estimators=100, 
                              learning_rate=0.1,
                              scale_pos_weight=scale_weight, 
                              random_state=42,
                              eval_metric='logloss',
                              n_jobs=-1)

xgb_model.fit(X_train, y_train)

# Evaluate the model (Default Threshold)
print("\n--- XGBoost Classification Report ---")
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- SHAP ANALYSIS ---
print("\nCalculating SHAP values (this might take a few seconds)...")

# Initialize the TreeExplainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Generate the SHAP Summary Plot
plt.figure(figsize=(10, 8))
plt.title('SHAP Summary Plot - Startup Success Drivers', fontsize=16)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()


# --- INDIVIDUAL STARTUP EXPLANATIONS (WATERFALL PLOTS) ---
print("\nGenerating individual SHAP Waterfall Plots...")

# 1. We need the new 'Explanation' object format for waterfall plots
explainer = shap.TreeExplainer(xgb_model)
shap_values_explanation = explainer(X_test)

# 2. Find the index of one predicted Success (1) and one predicted Failure (0)
# We use the y_pred array we generated earlier
predicted_success_idx = np.where(y_pred == 1)[0][0]
predicted_failure_idx = np.where(y_pred == 0)[0][0]

# 3. Plot for a Predicted SUCCESS
print(f"\nVisualizing a startup predicted to SUCCEED (Test Set Index: {predicted_success_idx})")
# Create a new figure to prevent overlapping plots
plt.figure(figsize=(10, 6)) 
shap.plots.waterfall(shap_values_explanation[predicted_success_idx], max_display=10, show=False)
plt.title(f"Why did the model predict SUCCESS for this startup?", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 4. Plot for a Predicted FAILURE
print(f"\nVisualizing a startup predicted to FAIL (Test Set Index: {predicted_failure_idx})")
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_explanation[predicted_failure_idx], max_display=10, show=False)
plt.title(f"Why did the model predict FAILURE for this startup?", fontsize=14, pad=20)
plt.tight_layout()
plt.show()