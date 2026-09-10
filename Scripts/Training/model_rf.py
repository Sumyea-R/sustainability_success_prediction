import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from utils import evaluate_model

def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, reports_folder):
    print("\nOptimizing Random Forest with Optuna HPO...")
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        # Convert None to a value or choice
        max_depth_choice = trial.suggest_categorical('max_depth', ['None', 10, 20, 30])
        max_depth = None if max_depth_choice == 'None' else max_depth_choice
        
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': max_depth,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        }
        
        rf = RandomForestClassifier(**param)
        rf.fit(X_train.fillna(0), y_train)
        preds = rf.predict(X_val.fillna(0))
        return f1_score(y_val, preds, average='macro')
    
    study.optimize(objective, n_trials=20)
    print(f"Random Forest Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model
    best_params = study.best_params.copy()
    max_depth_choice = best_params.get('max_depth', 'None')
    best_params['max_depth'] = None if max_depth_choice == 'None' else max_depth_choice
    
    rf_best = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_best.fit(X_train.fillna(0), y_train)
    
    # Evaluate
    results = evaluate_model(rf_best, X_test.fillna(0), y_test, 'Random Forest', reports_folder)
    return results
