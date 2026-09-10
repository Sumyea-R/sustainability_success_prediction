import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from utils import evaluate_model

def train_lr(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder):
    print("\nTraining Logistic Regression with Optuna HPO...")
    
    # Preprocess splits for linear models
    X_train_s = preprocessor.fit_transform(X_train.fillna(0)).astype('float32')
    X_val_s = preprocessor.transform(X_val.fillna(0)).astype('float32')
    X_test_s = preprocessor.transform(X_test.fillna(0)).astype('float32')
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        # Use class_weight='balanced' to handle severe imbalance
        lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_train_s, y_train)
        preds = lr.predict(X_val_s)
        return f1_score(y_val, preds, average='macro')
    
    study.optimize(objective, n_trials=20)
    print(f"Logistic Regression Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model on train+validation combined (or just train for simplicity)
    best_c = study.best_params['C']
    lr_best = LogisticRegression(C=best_c, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
    lr_best.fit(X_train_s, y_train)
    
    # Evaluate
    results = evaluate_model(lr_best, X_test_s, y_test, 'Logistic Regression', reports_folder, is_scaled=True)
    return results
