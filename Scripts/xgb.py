import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# If ID/Name are columns, set them as index so they aren't used as features
if 'ID' in df.columns and 'Name' in df.columns:
    df = df.set_index(['ID', 'Name'])
elif 'ID' in df.columns:
    df = df.set_index('ID')


# We must drop these because they are calculated relative to 2026, which breaks the time machine rule
leaky_columns = ['Months_Since_Last_Funding', 'Is_Funding_Overdue']
df = df.drop(columns=[col for col in leaky_columns if col in df.columns])


# Company_Age is (2026 - Launch Year)
# Train on mature startups (Launch <= 2018 -> Age >= 8)
train_mask = df['Company_Age'] >= 8

# Test on younger startups (Launch 2019-2021 -> Age 5 to 7)
test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)

# Startups with Age < 5 are completely dropped because they haven't had time to hit the 5-year CAGR rule

df_train = df[train_mask]
df_test = df[test_mask]

print(f"\n--- Cohort Split Results ---")
print(f"Training Set (Age >= 8):  {len(df_train)} startups")
print(f"Testing Set  (Age 5-7):   {len(df_test)} startups")
print(f"Excluded     (Age < 5):   {len(df) - len(df_train) - len(df_test)} startups")

# Separate Features and Target
X_train = df_train.drop(columns=['IS_SUCCESS'])
y_train = df_train['IS_SUCCESS']

X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']


# Calculate scale_pos_weight dynamically to handle the imbalance in the training set
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
print(scale_weight)

print("\nTraining the XGBoost model on Mature Cohort...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- Out-Of-Time Classification Report (Younger Cohort) ---")
print(classification_report(y_test, y_pred))





"""
# Let's look at standard feature importance first
importances = xgb_model.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 15 Most Important Features (OOT Model) ---")
print(feature_importance_df.head(15))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), hue='Feature', palette='magma', legend=False)
plt.title('Top 15 Predictors of Future Success (Trained on Mature Startups)', fontsize=14)
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.show()



print("\nCalculating SHAP values... (this may take a minute)")

# Initialize the TreeExplainer with your trained XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the Test Set (Younger Cohort)
# We use X_test so we are interpreting how the model makes decisions on the "future" data
shap_values = explainer.shap_values(X_test)


# This looks at the mean absolute SHAP value for each feature.
# It is more reliable than XGBoost's built-in feature importance.
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
plt.title('SHAP Global Feature Importance (Top 15)', fontsize=16)
plt.tight_layout()
plt.show()


# It shows not just what is important, but how it impacts the prediction.
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title('SHAP Summary: Directional Impact on Success', fontsize=16)
plt.tight_layout()
plt.show()


# How does Funding interact with the prediction
top_feature = "Log_Funding_Total_USD"
if top_feature in X_test.columns:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature}', fontsize=14)
    plt.tight_layout()
    plt.show()
    """