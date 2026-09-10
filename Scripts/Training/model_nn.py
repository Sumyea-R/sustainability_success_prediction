import optuna
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from utils import evaluate_model

def train_nn(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder):
    print("\nOptimizing Classic Neural Network (MLPClassifier) with Optuna HPO...")
    
    # Preprocess splits
    X_train_s = preprocessor.fit_transform(X_train.fillna(0)).astype('float32')
    X_val_s = preprocessor.transform(X_val.fillna(0)).astype('float32')
    X_test_s = preprocessor.transform(X_test.fillna(0)).astype('float32')
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        hidden_layer_choice = trial.suggest_categorical('hidden_layer_sizes', [
            (64,), (128,), (128, 64), (64, 32)
        ])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_choice,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
        
        mlp.fit(X_train_s, y_train)
        preds = mlp.predict(X_val_s)
        return f1_score(y_val, preds, average='macro')
        
    study.optimize(objective, n_trials=10)
    print(f"Classic Neural Network Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model
    best_params = study.best_params
    mlp_best = MLPClassifier(
        **best_params,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    mlp_best.fit(X_train_s, y_train)
    
    # Evaluate
    results = evaluate_model(mlp_best, X_test_s, y_test, 'NN', reports_folder, is_scaled=True)
    return results
