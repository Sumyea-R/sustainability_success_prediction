
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers


# LOAD DATA
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved.csv")

print(f"Loading dataset from: {input_file}")
df = pd.read_csv(input_file, sep=';')
if 'ID' in df.columns: df = df.set_index('ID')
if 'Name' in df.columns: df = df.drop(columns=['Name'])


# TIME-BASED SPLIT & LEAKAGE PREVENTION

tune_mask = df['Company_Age'] >= 8
df_tune = df[tune_mask].copy()

test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)
df_test = df[test_mask].copy()

# Company_Age must be dropped from all X matrices to prevent target leakage
cols_to_drop = ['IS_SUCCESS', 'Company_Age']

#  Splits for Optuna (Age >= 9 for Training, Age == 8 for Validation)
X_train_opt = df_tune[df_tune['Company_Age'] >= 9].drop(columns=cols_to_drop)
y_train_opt = df_tune[df_tune['Company_Age'] >= 9]['IS_SUCCESS']

X_val_opt = df_tune[df_tune['Company_Age'] == 8].drop(columns=cols_to_drop)
y_val_opt = df_tune[df_tune['Company_Age'] == 8]['IS_SUCCESS']

# Full Tuning Set for Final Model Training (Age >= 8)
X_tune_full = df_tune.drop(columns=cols_to_drop)
y_tune_full = df_tune['IS_SUCCESS']

# The Untouched Holdout Set (Age 5-7)
X_test = df_test.drop(columns=cols_to_drop)
y_test = df_test['IS_SUCCESS']

print("\nHyperparameter Tuning Setup (Chronological Split):")
print(f"  Training Fold (Age >= 9) : {len(X_train_opt)} startups")
print(f"  Validation Fold (Age == 8): {len(X_val_opt)} startups")
print(f"  Holdout Test Set (Age 5-7): {len(X_test)} startups\n")


# BAYESIAN OPTIMIZATION: RANDOM FOREST
print("="*50)
print("PHASE 1: OPTIMIZING RANDOM FOREST (50 Trials)")
print("="*50)

def objective_rf(trial):
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
    
    rf = RandomForestClassifier(**param)
    rf.fit(X_train_opt, y_train_opt)
    preds = rf.predict(X_val_opt)
    return f1_score(y_val_opt, preds)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)

print("\n--- Best Random Forest Parameters ---")
print(study_rf.best_params)

# Train Final RF Model
best_rf_params = study_rf.best_params
best_rf_params.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42})
rf_best = RandomForestClassifier(**best_rf_params)
rf_best.fit(X_tune_full, y_tune_full)

# BAYESIAN OPTIMIZATION: XGBOOST
print("\n" + "="*50)
print("PHASE 2: OPTIMIZING XGBOOST (50 Trials)")
print("="*50)

scale_pos_weight_train = (y_train_opt == 0).sum() / (y_train_opt == 1).sum()

def objective_xgb(trial):
    param = {
        'n_estimators': 1000, 
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'scale_pos_weight': scale_pos_weight_train, 
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50 
    }
    
    xgb_model = XGBClassifier(**param)
    xgb_model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        verbose=False
    )
    
    trial.set_user_attr("best_n_estimators", xgb_model.best_iteration + 1)
    preds = xgb_model.predict(X_val_opt)
    return f1_score(y_val_opt, preds)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)

print("\n--- Best XGBoost Parameters ---")
print(study_xgb.best_params)
optimal_n_trees = study_xgb.best_trial.user_attrs["best_n_estimators"]
print(f"Optimal number of trees (via early stopping): {optimal_n_trees}")

# Train Final XGB Model
scale_pos_weight_full = (y_tune_full == 0).sum() / (y_tune_full == 1).sum()
best_xgb_params = study_xgb.best_params
best_xgb_params.update({
    'n_estimators': optimal_n_trees,
    'scale_pos_weight': scale_pos_weight_full,
    'n_jobs': -1, 
    'random_state': 42
})
xgb_best = XGBClassifier(**best_xgb_params)
xgb_best.fit(X_tune_full, y_tune_full)


# BAYESIAN OPTIMIZATION: NEURAL NETWORK

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


# BAYESIAN OPTIMIZATION: LOGISTIC REGRESSION

# SCALER SETUP
continuous_cols = [
    'Log_Funding_Total_USD', 'Total_Rounds', 'Log_Avg_Round_Size', 
    'Log_Capital_Intensity', 'Num_Investors', 'Num_Founders', 
    'Num_SDGs', 'Num_patents', 'SubInd_Target_Encoded'
]

cont_indices = [X_train_opt.columns.get_loc(c) for c in continuous_cols if c in X_train_opt.columns]
passthrough_indices = [i for i in range(len(X_train_opt.columns)) if i not in cont_indices]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cont_indices),
        ('passthrough', 'passthrough', passthrough_indices)
    ])


print("OPTIMIZING LOGISTIC REGRESSION (50 Trials)")

def objective_lr(trial):
    # Search Space
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    
    l1_ratio = None
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
    lr = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        class_weight='balanced',
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    
    # Fit scaler strictly on training fold to prevent leakage into validation fold
    X_train_opt_scaled = preprocessor.fit_transform(X_train_opt)
    X_val_opt_scaled = preprocessor.transform(X_val_opt)
    
    lr.fit(X_train_opt_scaled, y_train_opt)
    preds = lr.predict(X_val_opt_scaled)
    
    return f1_score(y_val_opt, preds)

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=50)

print("\n--- Best Logistic Regression Parameters ---")
print(study_lr.best_params)


# TRAIN FINAL MODEL

# Re-fit scaler on the full tuning dataset
X_tune_full_scaled = preprocessor.fit_transform(X_tune_full)
X_test_scaled = preprocessor.transform(X_test)

best_lr_params = study_lr.best_params
if best_lr_params['penalty'] != 'elasticnet' and 'l1_ratio' in best_lr_params:
    del best_lr_params['l1_ratio']

best_lr_params.update({
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 2000,
    'random_state': 42,
    'n_jobs': -1
})

print("\nTraining final baseline model with best parameters...")
lr_best = LogisticRegression(**best_lr_params)
lr_best.fit(X_tune_full_scaled, y_tune_full)




# PERFORMANCE SCORECARD (HOLDOUT SET)

print("\n" + "="*50)
print("PHASE 3: EVALUATION ON UNTOUCHED HOLDOUT SET (Age 5-7)")
print("="*50)

lr_preds = lr_best.predict(X_test_scaled)
lr_probs = lr_best.predict_proba(X_test_scaled)[:, 1]

rf_preds = rf_best.predict(X_test)
rf_probs = rf_best.predict_proba(X_test)[:, 1]

xgb_preds = xgb_best.predict(X_test)
xgb_probs = xgb_best.predict_proba(X_test)[:, 1]

y_pred_proba_test = final_nn_model.predict(X_test_scaled, verbose=0)
y_pred_test_opt = (y_pred_proba_test > best_threshold).astype(int).flatten()

print("--- Logistic Regression ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_probs):.4f}")
print(classification_report(y_test, lr_preds))

print("\n--- Random Forest ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")
print(classification_report(y_test, rf_preds))

print("\n--- XGBoost ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
print(classification_report(y_test, xgb_preds))

print("\n--- Neural Network ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print(classification_report(y_test, y_pred_test_opt))

# Plot ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
fpr_nn, tpr_nn, _ = roc_curve(y_test, final_nn_model.predict(X_test_scaled).flatten())

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probs):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_probs):.3f})')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {roc_auc_score(y_test, final_nn_model.predict(X_test_scaled).flatten()):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC-AUC Curve: Predicting Startup Success')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# FEATURE IMPORTANCE LR (COEFFICIENTS)
coefficients = lr_best.coef_[0]
feature_names = X_test.columns.tolist()

# Get top 15 absolute coefficients
abs_coefs = np.abs(coefficients)
top_indices = np.argsort(abs_coefs)[::-1][:15]

top_features = [feature_names[i] for i in top_indices]
top_coefs = coefficients[top_indices]

plt.figure(figsize=(10, 6))
# Color red if negative impact, green if positive impact
colors = ['red' if c < 0 else 'green' for c in top_coefs]
plt.barh(top_features[::-1], top_coefs[::-1], color=colors[::-1])
plt.title('Top 15 Feature Coefficients - Logistic Regression Baseline')
plt.xlabel('Coefficient Value (Impact on Log-Odds)')
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE RF & XGB (BUILT-IN)
def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_best, X_tune_full.columns, "Top 15 Features - Random Forest (Built-in)")
plot_feature_importance(xgb_best, X_tune_full.columns, "Top 15 Features - XGBoost (Built-in)")


# SHAP VALUES 


# SHAP VALUES FOR LOGISTIC REGRESSION
# Initialize the LinearExplainer
explainer_lr = shap.LinearExplainer(lr_best, X_tune_full_scaled)

# Sample the holdout set for plotting
np.random.seed(42)
test_indices = np.random.choice(X_test_scaled.shape[0], min(1000, X_test_scaled.shape[0]), replace=False)

# Get the scaled data for the model to calculate SHAP values
X_test_lr_sample_scaled = X_test_scaled[test_indices]

# Get the unscaled data strictly for plotting the colors
X_test_lr_sample_unscaled = X_test.iloc[test_indices]

# Calculate SHAP values
shap_values_lr = explainer_lr.shap_values(X_test_lr_sample_scaled)

# Plot the results
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Logistic Regression (Baseline)")

# We pass the calculated SHAP values, but use the unscaled features for the color map
shap.summary_plot(
    shap_values_lr, 
    features=X_test_lr_sample_unscaled, 
    feature_names=X_test.columns.tolist(), 
    max_display=15, 
    show=False
)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("SHAP VALUE INTERPRETATION")
print("="*50)

# XGBoost SHAP 
print("Running XGBoost SHAP (Sample = 1000)...")
X_test_xgb_sample = X_test.sample(n=1000, random_state=42)
explainer_xgb = shap.TreeExplainer(xgb_best)
shap_values_xgb = explainer_xgb(X_test_xgb_sample)

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - XGBoost (Predicting Success)")
shap.summary_plot(shap_values_xgb, X_test_xgb_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()

# Random Forest SHAP 
print("Running Random Forest SHAP (Sample = 300, Fast Approx)...")
X_test_rf_sample = X_test.sample(n=300, random_state=42)
explainer_rf = shap.TreeExplainer(rf_best)
shap_values_rf = explainer_rf.shap_values(
    X_test_rf_sample, 
    check_additivity=False, 
    approximate=True
)

if isinstance(shap_values_rf, list):
    shap_vals_class1 = shap_values_rf[1]
elif len(shap_values_rf.shape) == 3:
    shap_vals_class1 = shap_values_rf[:, :, 1]
else:
    shap_vals_class1 = shap_values_rf

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Random Forest (Predicting Success)")
shap.summary_plot(shap_vals_class1, X_test_rf_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()


# NEURAL NETWORK SHAP

# KernelExplainer expects a function that takes a 2D array and returns a 1D array of predictions.
def nn_predict_wrapper(data):
    # Ensure it's a numpy array just in case SHAP passes a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    # Predict and flatten to 1D array
    return final_nn_model.predict(data, verbose=0).flatten()

# Summarize the background dataset using k-means
# K-means reduces computational load by clustering the background data and using the cluster centers as representative points
background_summary = shap.kmeans(X_tune_full_scaled, 50)

# Initialize the KernelExplainer
explainer_nn = shap.KernelExplainer(nn_predict_wrapper, background_summary)

# Get a sample of indices to explain
np.random.seed(42)
test_indices = np.random.choice(X_test_scaled.shape[0], 100, replace=False)

# Get the scaled data for the model to actually calculate the SHAP values
X_test_nn_sample_scaled = X_test_scaled[test_indices]

# Get the unscaled data strictly for plotting
X_test_nn_sample_unscaled = X_test.iloc[test_indices]

# 5. Calculate SHAP values
print("Calculating SHAP values")
shap_values_nn = explainer_nn.shap_values(X_test_nn_sample_scaled)

# Depending on the version of SHAP and the model, shap_values_nn might be a list (for multi-class) or a single array.
if isinstance(shap_values_nn, list):
    shap_vals_to_plot = shap_values_nn[0]
else:
    shap_vals_to_plot = shap_values_nn

# Plot the results
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary - Neural Network (Predicting Success)")

# We pass the SHAP values (calculated on scaled data) but feed it the unscaled features for the color mapping
shap.summary_plot(
    shap_vals_to_plot, 
    features=X_test_nn_sample_unscaled,
    feature_names=X_test.columns.tolist(), 
    max_display=15, 
    show=False
)
plt.tight_layout()
plt.show()