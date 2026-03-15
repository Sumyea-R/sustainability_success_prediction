import os
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

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

# For Optuna, it is much faster to explicitly slice the Train and Val sets
X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=['IS_SUCCESS'])
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=['IS_SUCCESS'])
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

# The untouched holdout set
X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']

print("Hyperparameter Tuning Setup (Bayesian Optimization):")
print(f"Training Fold (Age >= 9): {len(X_train_opt)} startups")
print(f"Validation Fold (Age = 8): {len(X_val_opt)} startups")
print(f"Holdout Test Set (Age 5-7): {len(X_test)} startups\n")

# Optuna's objective function for Random Forest
def objective(trial):
    # 1. Define the continuous and discrete search space
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
    
    # 2. Initialize and train the model on the strict Age >= 9 training set
    rf = RandomForestClassifier(**param)
    rf.fit(X_train_opt, y_train_opt)
    
    # 3. Predict strictly on the Age == 8 validation set
    preds = rf.predict(X_val_opt)
    
    # 4. Return the F1-score of the Success class for Optuna to maximize
    return f1_score(y_val_opt, preds)


print("Running Bayesian Optimization... (50 trials)")
# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


print("\n--- BEST HYPERPARAMETERS (OPTUNA) ---")
print(study.best_params)

# Train the absolute final model using the best params on the FULL tuning set
print("\n--- EVALUATING OPTUNA MODEL ON UNTOUCHED TEST SET ---")
best_params = study.best_params
best_params['class_weight'] = 'balanced'
best_params['random_state'] = 42
best_params['n_jobs'] = -1

final_rf_model = RandomForestClassifier(**best_params)

# We train on the combined Train + Val (Age >= 8) to give it maximum data before testing
X_tune_full = df_tune.drop(columns=['IS_SUCCESS'])
y_tune_full = df_tune['IS_SUCCESS']
final_rf_model.fit(X_tune_full, y_tune_full)

# Final prediction on the Holdout Test Set (Age 5-7)
y_pred_test_opt = final_rf_model.predict(X_test)

print(classification_report(y_test, y_pred_test_opt))