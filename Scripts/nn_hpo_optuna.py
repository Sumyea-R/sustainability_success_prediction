import os
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers


# LOAD DATA & CLEAN LEAKAGE
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])

# 2. TIME-BASED SPLIT 
tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

# Drop both Target and Age from X!
cols_to_drop = ['IS_SUCCESS', 'Company_Age']

X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=cols_to_drop)
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=cols_to_drop)
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

X_tune_full = df_tune.drop(columns=cols_to_drop)
y_tune_full = df_tune['IS_SUCCESS']

X_test = df_test.drop(columns=cols_to_drop)
y_test = df_test['IS_SUCCESS']

print("Hyperparameter Tuning Setup (Advanced Neural Network):")
print(f"  Training Fold: {len(X_train_opt)} startups")
print(f"  Validation Fold: {len(X_val_opt)} startups")
print(f"  Holdout Test Set: {len(X_test)} startups\n")

# TARGETED FEATURE SCALING
continuous_cols = [
    'Log_Funding_Total_USD', 'Total_Rounds', 'Log_Avg_Round_Size', 
    'Log_Capital_Intensity', 'Num_Investors', 'Num_Founders', 
    'Num_SDGs', 'Num_patents', 'SubInd_Target_Encoded'
]
# Get indices of continuous columns for the ColumnTransformer
cont_indices = [X_train_opt.columns.get_loc(c) for c in continuous_cols if c in X_train_opt.columns]
passthrough_indices = [i for i in range(len(X_train_opt.columns)) if i not in cont_indices]

# Build the targeted scaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])

X_train_opt_scaled = preprocessor.fit_transform(X_train_opt)
X_val_opt_scaled = preprocessor.transform(X_val_opt)

# Class Weights
classes = np.unique(y_train_opt)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_opt)
class_weight_dict = dict(zip(classes, weights))

# ADVANCED MODEL BUILDER

def create_model(params, input_dim):
    model = Sequential()
    
    # Layer 0 - with L2 regularization
    model.add(Dense(
        params['units_layer_0'], 
        activation='swish', 
        kernel_regularizer=regularizers.l2(params['l2_reg']),
        input_shape=(input_dim,)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    
    # Dynamic Hidden Layers
    for i in range(1, params['n_layers']):
        model.add(Dense(
            params[f'units_layer_{i}'], 
            activation='swish',
            kernel_regularizer=regularizers.l2(params['l2_reg'])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# OPTUNA OPTIMIZATION

def objective(trial):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 3), 
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True), 
        'units_layer_0': trial.suggest_int('units_layer_0', 64, 256, step=32)
    }
    
    for i in range(1, params['n_layers']):
        params[f'units_layer_{i}'] = trial.suggest_int(f'units_layer_{i}', 32, 128, step=16)
        
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    threshold = trial.suggest_float('threshold', 0.3, 0.7) 
    
    model = create_model(params, X_train_opt_scaled.shape[1])
    
    # ReduceLROnPlateau drops learning rate if val_loss stagnates
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    
    model.fit(
        X_train_opt_scaled, y_train_opt,
        validation_data=(X_val_opt_scaled, y_val_opt),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, lr_scheduler],
        verbose=0 
    )
    
    preds_proba = model.predict(X_val_opt_scaled, verbose=0)
    preds = (preds_proba > threshold).astype(int).flatten()
    
    return f1_score(y_val_opt, preds)

print("Running Bayesian Optimization for Advanced NN... (50 trials)")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Parameters found by Optuna:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# TRAIN FINAL MODEL 
# Fit scaler on full tuning set
preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])

X_tune_full_scaled = preprocessor_full.fit_transform(X_tune_full)
X_test_scaled = preprocessor_full.transform(X_test)

classes_full = np.unique(y_tune_full)
weights_full = compute_class_weight('balanced', classes=classes_full, y=y_tune_full)
class_weight_dict_full = dict(zip(classes_full, weights_full))

best_params = study.best_params
final_nn_model = create_model(best_params, X_tune_full_scaled.shape[1])
best_batch_size = best_params.get('batch_size', 64) 
best_threshold = best_params.get('threshold', 0.5)

print(f"\nTraining final model with optimized threshold: {best_threshold:.4f}...")

# We use an early stopping split from the tuning data just to prevent overfitting the final training phase
final_early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

final_nn_model.fit(
    X_tune_full_scaled, y_tune_full,
    validation_split=0.15, 
    epochs=40, 
    batch_size=best_batch_size,
    class_weight=class_weight_dict_full,
    callbacks=[final_early_stopping],
    verbose=1
)


# EVALUATION

y_pred_proba_test = final_nn_model.predict(X_test_scaled, verbose=0)
y_pred_test_opt = (y_pred_proba_test > best_threshold).astype(int).flatten()

print("\n" + "="*50)
print("FINAL HOLDOUT EVALUATION (ADVANCED NEURAL NETWORK)")
print("="*50)
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print(classification_report(y_test, y_pred_test_opt))