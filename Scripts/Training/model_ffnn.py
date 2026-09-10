import optuna
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from utils import evaluate_model

def train_ffnn(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, reports_folder):
    print("\nOptimizing Neural Network with Optuna HPO...")
    
    # Preprocess splits for Neural Network
    X_train_s = preprocessor.fit_transform(X_train.fillna(0)).astype('float32')
    X_val_s = preprocessor.transform(X_val.fillna(0)).astype('float32')
    X_test_s = preprocessor.transform(X_test.fillna(0)).astype('float32')
    
    # Compute class weights (cost-sensitive learning)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {i: weights[i] for i in range(len(classes))}
    
    # Hide Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        model = Sequential()
        units_0 = trial.suggest_int('units_0', 64, 256)
        model.add(Dense(units_0, activation='relu', input_dim=X_train_s.shape[1]))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        for i in range(1, n_layers):
            units_i = trial.suggest_int(f'units_{i}', 32, 128)
            model.add(Dense(units_i, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='sparse_categorical_crossentropy')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), 
                  epochs=30, batch_size=64, class_weight=class_weights_dict, callbacks=[early_stop], verbose=0)
        
        probs = model.predict(X_val_s, verbose=0)
        preds = np.argmax(probs, axis=1)
        return f1_score(y_val, preds, average='macro')
    
    study.optimize(objective, n_trials=10)
    print(f"Neural Network Best Trial F1-Macro: {study.best_value:.4f}")
    
    # Train final model with best params
    best_params = study.best_params
    final_nn = Sequential()
    final_nn.add(Dense(best_params['units_0'], activation='relu', input_dim=X_train_s.shape[1]))
    final_nn.add(BatchNormalization())
    final_nn.add(Dropout(best_params['dropout_rate']))
    
    for i in range(1, best_params['n_layers']):
        final_nn.add(Dense(best_params[f'units_{i}'], activation='relu'))
        final_nn.add(BatchNormalization())
        final_nn.add(Dropout(best_params['dropout_rate']))
        
    final_nn.add(Dense(3, activation='softmax'))
    final_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), 
                     loss='sparse_categorical_crossentropy')
    
    # Train on best params with validation data and early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    final_nn.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), 
                 epochs=50, batch_size=64, class_weight=class_weights_dict, 
                 callbacks=[early_stop], verbose=0)
    
    # Evaluate
    results = evaluate_model(final_nn, X_test_s, y_test, 'FFNN', reports_folder, is_nn=True)
    return results
