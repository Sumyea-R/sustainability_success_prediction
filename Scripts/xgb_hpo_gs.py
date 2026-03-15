import os
import pandas as pd
import numpy as np
import xgboost as xgb
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

print(f"Hyperparameter Tuning Setup (XGBoost Grid Search):")
print(f"Training Fold (Age >= 9): {(test_fold == -1).sum()} startups")
print(f"Validation Fold (Age = 8): {(test_fold == 0).sum()} startups")
print(f"Holdout Test Set (Age 5-7): {len(df_test)} startups\n")

# Calculate dynamic scale_pos_weight
train_only_y = y_tune[test_fold == -1]
scale_weight = (train_only_y == 0).sum() / (train_only_y == 1).sum() if (train_only_y == 1).sum() > 0 else 1.0


# Centered directly around the Randomized Search winners
param_grid_xgb = {
    'n_estimators': [400, 500, 600],
    'max_depth': [2, 3, 4],               # Testing slightly shallower/deeper
    'learning_rate': [0.1, 0.2],          # 0.2 was best, but 0.1 is standard
    'subsample': [0.7, 0.8, 0.9],         # Bracket the winning 0.8
    'colsample_bytree': [0.8],            # Keeping fixed to reduce total fits
    'min_child_weight': [1, 2],           # Checking if slight regularization helps
    'gamma': [0]                          # Keeping fixed
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)


# Total combinations: 3 * 3 * 2 * 3 * 1 * 2 * 1 = 108 fits.
print("Running exhaustive Grid Search for XGBoost... (~108 fits)")
grid_search_xgb = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid_xgb,
    scoring='f1',       
    cv=ps,              
    verbose=1,
    n_jobs=-1
)

grid_search_xgb.fit(X_tune, y_tune)


print("\n--- BEST HYPERPARAMETERS (XGBOOST GRID SEARCH) ---")
print(grid_search_xgb.best_params_)

print("\n--- EVALUATING XGBOOST GRID SEARCH ON UNTOUCHED TEST SET ---")
best_gs_xgb_model = grid_search_xgb.best_estimator_
y_pred_test_gs_xgb = best_gs_xgb_model.predict(X_test)

print(classification_report(y_test, y_pred_test_gs_xgb))