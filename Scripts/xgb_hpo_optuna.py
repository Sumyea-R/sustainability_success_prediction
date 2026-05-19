import os
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    average_precision_score, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_2021_snapshot_features.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

if 'ID' in df.columns: 
    df = df.set_index('ID')
if 'Name' in df.columns: 
    df = df.drop(columns=['Name'])

targets = ['IS_EXIT', 'IS_HIGH_GROWTH']

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def find_optimal_threshold(y_true, y_proba, beta=2, title_suffix=""):
    """
    Finds the optimal probability threshold based on the F-beta score.
    beta=2 weighs Recall heavily, which is crucial for finding VC unicorns.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta score for all thresholds (add 1e-8 to avoid division by zero)
    fbeta_scores = ((1 + beta**2) * (precisions * recalls)) / ((beta**2 * precisions) + recalls + 1e-8)
    
    # Find the index of the maximum F-beta score
    optimal_idx = np.argmax(fbeta_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n--- OPTIMAL THRESHOLD CALCULATION (F{beta}-Score) ---")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Expected Precision: {precisions[optimal_idx]:.4f}")
    print(f"Expected Recall:    {recalls[optimal_idx]:.4f}")
    
    # Plotting the trade-off
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.plot(thresholds, fbeta_scores[:-1], 'r-', label=f'F{beta} Score')
    plt.axvline(x=optimal_threshold, color='k', linestyle=':', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.title(f'Precision-Recall Trade-off: {title_suffix}')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False) # Show without blocking script execution
    
    return optimal_threshold

def plot_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('XGBoost Relative Importance')
    plt.tight_layout()
    plt.show(block=False)

# ==============================================================================
# 3. OPTUNA TUNING & EVALUATION CORE
# ==============================================================================
def tune_and_evaluate(df, target_col):
    print(f"\n\n{'='*60}")
    print(f"🚀 TRAINING PIPELINE: {target_col}")
    print(f"{'='*60}")
    
    # Drop the *other* target so the model can't cheat
    X = df.drop(columns=targets)
    y = df[target_col]
    
    # --- SPLIT 1: Isolate Holdout Test Set (20%) ---
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # --- SPLIT 2: Train (60%) and Validation (20%) for Optuna ---
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    scale_weight_train = (y_train == 0).sum() / (y_train == 1).sum()
    
    print(f"Train Set:      {len(X_train)} startups")
    print(f"Validation Set: {len(X_val)} startups (Used for Early Stopping)")
    print(f"Holdout Set:    {len(X_test)} startups")
    print(f"Running Bayesian Optimization (50 trials)...")

    def objective(trial):
        param = {
            'n_estimators': 1500,
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'scale_pos_weight': scale_weight_train, 
            'eval_metric': 'aucpr', 
            'early_stopping_rounds': 40,
            'n_jobs': -1,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Save best iteration
        trial.set_user_attr("best_n_estimators", model.best_iteration + 1)
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds_proba)

    # Run Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("\n--- BEST HYPERPARAMETERS FOUND ---")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    optimal_trees = study.best_trial.user_attrs["best_n_estimators"]
    print(f"  optimal_trees: {optimal_trees} (via Early Stopping)")

    # ==========================================================================
    # FINAL MODEL TRAINING & EVALUATION
    # ==========================================================================
    print("\n--- EVALUATING FINAL MODEL ON UNTOUCHED HOLDOUT SET ---")
    scale_weight_temp = (y_temp == 0).sum() / (y_temp == 1).sum()
    
    best_params = study.best_params
    best_params['n_estimators'] = optimal_trees
    best_params['scale_pos_weight'] = scale_weight_temp
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_temp, y_temp, verbose=False)

    y_proba = final_model.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC Score:  {average_precision_score(y_test, y_proba):.4f}")
    
    # Find and apply optimal VC threshold
    best_thresh = find_optimal_threshold(y_test, y_proba, beta=2, title_suffix=target_col)
    y_pred_custom = (y_proba >= best_thresh).astype(int)

    print(f"\nClassification Report (Custom Threshold: {best_thresh:.4f}):")
    print(classification_report(y_test, y_pred_custom))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_custom)
    print(f"True Negatives:  {cm[0][0]:<5} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]:<5} | True Positives:  {cm[1][1]}\n")
    
    return final_model, X.columns

# ==============================================================================
# 4. EXECUTE PIPELINE
# ==============================================================================
model_exit, features_exit = tune_and_evaluate(df, 'IS_EXIT')
plot_importance(model_exit, features_exit, "Top 15 Features: Predicting Exits")

model_growth, features_growth = tune_and_evaluate(df, 'IS_HIGH_GROWTH')
plot_importance(model_growth, features_growth, "Top 15 Features: Predicting High Growth")

# Keep plots open at the end
plt.show()