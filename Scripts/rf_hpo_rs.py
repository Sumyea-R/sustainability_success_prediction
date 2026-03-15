import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import classification_report


current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])

leaky_columns = ['Months_Since_Last_Funding', 'Is_Funding_Overdue']
df = df.drop(columns=[col for col in leaky_columns if col in df.columns])

tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

X_tune = df_tune.drop(columns=['IS_SUCCESS'])
y_tune = df_tune['IS_SUCCESS']

X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']


# Train on Age >= 9, Validate on Age == 8
test_fold = np.where(df_tune['Company_Age'] == 8, 0, -1)
ps = PredefinedSplit(test_fold)

print(f"Hyperparameter Tuning Setup (Random Forest):")
print(f"Training Fold (Age >= 9): {(test_fold == -1).sum()} startups")
print(f"Validation Fold (Age = 8): {(test_fold == 0).sum()} startups")
print(f"Holdout Test Set (Age 5-7): {len(df_test)} startups\n")


# Define a rigorous search space for Random Forest
param_distributions_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Random search
print("Running rigorous hyperparameter optimization for Random Forest... (This may take a few minutes)")
random_search_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions_rf,
    n_iter=50, 
    scoring='f1',       
    cv=ps,              
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_tune, y_tune)


print("\n--- BEST HYPERPARAMETERS (RANDOM FOREST) ---")
print(random_search_rf.best_params_)

print("\n--- EVALUATING OPTIMIZED RF MODEL ON UNTOUCHED TEST SET ---")
best_rf_model = random_search_rf.best_estimator_
y_pred_test_rf = best_rf_model.predict(X_test)

print(classification_report(y_test, y_pred_test_rf))