import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from utils import evaluate_model

def train_cat(X_train, y_train, X_val, y_val, X_test, y_test, cat_features, reports_folder):
    print("\nOptimizing CatBoost with Optuna HPO...")
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
            'loss_function': 'MultiClass',
            'auto_class_weights': 'Balanced',
            'verbose': False,
            'random_state': 42
        }
        
        cat_model = CatBoostClassifier(**param)
        cat_model.fit(X_train.fillna(0), y_train, cat_features=cat_features, eval_set=(X_val.fillna(0), y_val), early_stopping_rounds=20, verbose=False)
        preds = cat_model.predict(X_val.fillna(0))
        return f1_score(y_val, preds, average='macro')
    
    study.optimize(objective, n_trials=20)
    print(f"CatBoost Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model
    cat_best = CatBoostClassifier(**study.best_params, loss_function='MultiClass', auto_class_weights='Balanced', verbose=False, random_state=42)
    cat_best.fit(X_train.fillna(0), y_train, cat_features=cat_features)
    
    # Evaluate
    results = evaluate_model(cat_best, X_test.fillna(0), y_test, 'CatBoost', reports_folder)
    return results
