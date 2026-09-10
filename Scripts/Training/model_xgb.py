import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from utils import evaluate_model

def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, reports_folder):
    print("\nOptimizing XGBoost with Optuna HPO...")
    
    # Compute sample weights for training set (cost-sensitive learning)
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'multi:softprob',
            'num_class': 3,
            'enable_categorical': True,
            'random_state': 42
        }
        
        xgb_model = XGBClassifier(**param)
        xgb_model.fit(X_train.fillna(0), y_train, sample_weight=sample_weights_train)
        preds = xgb_model.predict(X_val.fillna(0))
        return f1_score(y_val, preds, average='macro')
    
    study.optimize(objective, n_trials=20)
    print(f"XGBoost Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model
    xgb_best = XGBClassifier(**study.best_params, objective='multi:softprob', num_class=3, enable_categorical=True, random_state=42)
    xgb_best.fit(X_train.fillna(0), y_train, sample_weight=sample_weights_train)
    
    # Evaluate
    results = evaluate_model(xgb_best, X_test.fillna(0), y_test, 'XGBoost', reports_folder)
    return results
