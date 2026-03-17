import os
import pandas as pd
import numpy as np
import xgboost as xgb
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


# Tuning Data (Train + Validation)
tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

# 2. Final Test Data (Untouched Holdout)
test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

X_tune = df_tune.drop(columns=['IS_SUCCESS'])
y_tune = df_tune['IS_SUCCESS']

X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']


# Assign -1 to training rows, and 0 to validation rows.
# Train on Age >= 9, Validate on Age == 8
test_fold = np.where(df_tune['Company_Age'] == 8, 0, -1)
ps = PredefinedSplit(test_fold)

print(f"Hyperparameter Tuning Setup:")
print(f"Training Fold (Age >= 9): {(test_fold == -1).sum()} startups")
print(f"Validation Fold (Age = 8): {(test_fold == 0).sum()} startups")
print(f"Holdout Test Set (Age 5-7): {len(df_test)} startups\n")


# Calculate dynamic scale_pos_weight based ONLY on the training fold
train_only_y = y_tune[test_fold == -1]
scale_weight = (train_only_y == 0).sum() / (train_only_y == 1).sum() if (train_only_y == 1).sum() > 0 else 1.0

# Define a rigorous search space for XGBoost
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],               
    'colsample_bytree': [0.6, 0.8, 1.0],        
    'min_child_weight': [1, 3, 5, 7],           
    'gamma': [0, 0.1, 0.5, 1, 5]                
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)


print("Running rigorous hyperparameter optimization...")

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=50, 
    scoring='f1',       
    cv=ps,              
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_tune, y_tune)


print(random_search.best_params_)

best_xgb_model = random_search.best_estimator_
y_pred_test = best_xgb_model.predict(X_test)

print(classification_report(y_test, y_pred_test))