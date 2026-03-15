import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
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


test_fold = np.where(df_tune['Company_Age'] == 8, 0, -1)
ps = PredefinedSplit(test_fold)

print(f"Hyperparameter Tuning Setup (Grid Search):")
print(f"Training Fold (Age >= 9): {(test_fold == -1).sum()} startups")
print(f"Validation Fold (Age = 8): {(test_fold == 0).sum()} startups")
print(f"Holdout Test Set (Age 5-7): {len(df_test)} startups\n")


# Centered around the best parameters from Randomized Search
param_grid_rf = {
    'n_estimators': [400, 500, 600],
    'max_depth': [None, 30, 50],
    'min_samples_split': [15, 20, 25],
    'min_samples_leaf': [6, 8, 10],
    'max_features': [None, 0.5], # Testing 'None' (all features) vs 50% of features
    'class_weight': ['balanced']
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# Total combinations: 3 * 3 * 3 * 3 * 2 * 1 = 162 fits.
print("Running exhaustive Grid Search... (This will take a bit longer, ~162 fits)")
grid_search_rf = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf,
    scoring='f1',       
    cv=ps,              
    verbose=1,
    n_jobs=-1
)

grid_search_rf.fit(X_tune, y_tune)


print("\n--- BEST HYPERPARAMETERS (GRID SEARCH) ---")
print(grid_search_rf.best_params_)

print("\n--- EVALUATING GRID SEARCH MODEL ON UNTOUCHED TEST SET ---")
best_gs_rf_model = grid_search_rf.best_estimator_
y_pred_test_gs = best_gs_rf_model.predict(X_test)

print(classification_report(y_test, y_pred_test_gs))