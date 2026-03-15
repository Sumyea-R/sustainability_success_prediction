import os
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
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

# For Optuna, explicitly slice the Train and Val sets
X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=['IS_SUCCESS'])
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=['IS_SUCCESS'])
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

# The untouched holdout set
X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']

print("Hyperparameter Tuning Setup (Bayesian Optimization with Early Stopping):")
print(f"Training Fold (Age >= 9): {len(X_train_opt)} startups")
print(f"Validation Fold (Age = 8): {len(X_val_opt)} startups")
print(f"Holdout Test Set (Age 5-7): {len(X_test)} startups\n")

# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight_train = (y_train_opt == 0).sum() / (y_train_opt == 1).sum()


def objective(trial):
    # 1. Define the search space
    param = {
        'n_estimators': 1000, # Set high; early stopping will cut this off automatically
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'scale_pos_weight': scale_pos_weight_train, 
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50 # Stop if validation score doesn't improve for 50 rounds
    }
    
    # 2. Initialize the model
    xgb_model = XGBClassifier(**param)
    
    # 3. Train with the validation set passed to eval_set for early stopping
    xgb_model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        verbose=False
    )
    
    # Save the actual number of trees used before early stopping triggered
    # This allows us to use the exact right number of trees in the final model
    trial.set_user_attr("best_n_estimators", xgb_model.best_iteration + 1)
    
    # 4. Predict strictly on the Age == 8 validation set
    preds = xgb_model.predict(X_val_opt)
    
    # 5. Return the F1-score of the Success class for Optuna to maximize
    return f1_score(y_val_opt, preds)


print("Running Bayesian Optimization... (50 trials)")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


print("\n--- BEST HYPERPARAMETERS (OPTUNA) ---")
print(study.best_params)

# Retrieve the optimal number of trees discovered by early stopping
optimal_n_estimators = study.best_trial.user_attrs["best_n_estimators"]
print(f"Optimal number of trees (found via early stopping): {optimal_n_estimators}")

print("\n--- EVALUATING OPTUNA MODEL ON UNTOUCHED TEST SET ---")

# Recalculate scale_pos_weight for the combined tuning set
X_tune_full = df_tune.drop(columns=['IS_SUCCESS'])
y_tune_full = df_tune['IS_SUCCESS']
scale_pos_weight_full = (y_tune_full == 0).sum() / (y_tune_full == 1).sum()

# Compile the final parameters
best_params = study.best_params
best_params['n_estimators'] = optimal_n_estimators # Inject the optimal tree count
best_params['scale_pos_weight'] = scale_pos_weight_full
best_params['random_state'] = 42
best_params['n_jobs'] = -1

# Initialize the final model (without early stopping, since we are using all tuning data)
final_xgb_model = XGBClassifier(**best_params)

# Fit on combined Train + Val (Age >= 8)
final_xgb_model.fit(X_tune_full, y_tune_full, verbose=False)

# Final prediction on the Holdout Test Set (Age 5-7)
y_pred_test_opt = final_xgb_model.predict(X_test)

print(classification_report(y_test, y_pred_test_opt))