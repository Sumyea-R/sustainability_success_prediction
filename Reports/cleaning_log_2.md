### Second attempt cleaning the data

I now use all startups (~26000) and not only those from 2016
Success variable is IPO/AQCUSITION or HIGH_GROWTH (calculated by lifetime emplyoee CAGR >= 20%)
Missing values in launch year (~1000) handled with median inputation
CAGR requirements: +10 employees and 2026-launch_year >= 5

before median inputation: 

High Growth (CAGR):    8823
Successful Exits:      2324
TOTAL SUCCESSES (1):   9928
------------------------------
Class Distribution:
IS_SUCCESS
0    0.625401
1    0.374599

after:

High Growth (CAGR):    8921
Successful Exits:      2324
TOTAL SUCCESSES (1):   10014
------------------------------
Class Distribution:
IS_SUCCESS
0    0.622156
1    0.377844




##### applied old cleaning_and_features.py:

Entries before removing duplicates: 26503

Entries after removing duplicates: 26454

Startups without Top-20 Industry: 4.0%
Startups without Top-60 Sub-Industry: 25.1%
Distribution of gender category:
Gender_Category
-1    0.533341
 0    0.355485
 1    0.111174
Name: proportion, dtype: float64
IS_SUCCESS
0    0.621948
1    0.378052
Name: proportion, dtype: float64
--- Country Coverage Check ---
Top countries cover: 94.3%
Startups in 'long tail': 5.7%
------------------------------
ANALYSIS: Recency & Overdue Features
------------------------------
Stats for months without fallback strategy:
count    26427.000000
mean        70.791917
std         65.135263
min          0.000000
25%         25.000000
50%         52.000000
75%        101.000000
max        852.000000
Name: Months_Since_Last_Funding, dtype: float64

Number of 'unknown/fallback' (999): 27 (0.1%)
\Distribution 'Is_Funding_Overdue' (> 24 months):
Is_Funding_Overdue
1    0.756596
0    0.243404
Name: proportion, dtype: float64

Crosstab: Overdue vs. Success:
IS_SUCCESS                 0         1
Is_Funding_Overdue                    
0                   0.425843  0.574157
1                   0.685036  0.314964
------------------------------
       Log_Funding_Total_USD  Log_Capital_Intensity  Log_Avg_Round_Size
count               26454.00               26454.00            18242.00
mean                    1.01                   0.64                0.89
std                     1.51                   1.04                1.10
min                     0.00                   0.00                0.00
25%                     0.00                   0.00                0.01
50%                     0.06                   0.03                0.45
75%                     1.69                   0.93                1.39
max                     9.42                   7.84                7.84

----------------------------------------
Funding_Total_USD    0
Capital_Intensity    0
Avg_Round_Size       0
dtype: int64
No infinite values detected. Log transformation seems safe.
----------------------------------------
Distribution of success 0/1: 1.65



### Hyperparameter optimization 

I followed a time based train-test split, training on older startups >= 9 years old, validating on startups that are 8 years old, and testing on startups 5-7 years old. 
Startups younger than 5 years were excluded since they dont meet the 5 years CAGR rule and are too young to be "successful" in our definition.

#### Random Forest

I started with the random forest algorithm.
##### I performed random search and got following results 

--- BEST HYPERPARAMETERS (RANDOM FOREST) ---
{'n_estimators': 500, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_features': None, 'max_depth': None, 'class_weight': 'balanced'}

--- EVALUATING OPTIMIZED RF MODEL ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.80      0.84      0.82      3793
           1       0.75      0.70      0.73      2682

    accuracy                           0.78      6475
   macro avg       0.78      0.77      0.77      6475
weighted avg       0.78      0.78      0.78      6475

##### Grid search came to following results:
--- BEST HYPERPARAMETERS (GRID SEARCH) ---
{'class_weight': 'balanced', 'max_depth': 50, 'max_features': 0.5, 'min_samples_leaf': 6, 'min_samples_split': 20, 'n_estimators': 500}

--- EVALUATING GRID SEARCH MODEL ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.80      0.84      0.82      3793
           1       0.75      0.70      0.72      2682

    accuracy                           0.78      6475
   macro avg       0.77      0.77      0.77      6475
weighted avg       0.78      0.78      0.78      6475


##### Bayesian optimization results:
best due to max_features = 0.41 -> rs has max_features = None
=> more efficient and less prone to overfitting 
best in rf vs xgb comparision 

--- BEST HYPERPARAMETERS (OPTUNA) ---
{'n_estimators': 490, 'max_depth': 50, 'min_samples_split': 32, 'min_samples_leaf': 11, 'max_features': 0.409765423746428}

--- EVALUATING OPTUNA MODEL ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.80      0.83      0.82      3793
           1       0.75      0.70      0.73      2682

    accuracy                           0.78      6475
   macro avg       0.77      0.77      0.77      6475
weighted avg       0.78      0.78      0.78      6475


#### XGBoost

Same procedure as with RF

##### Random Search
Hyperparameter Tuning Setup:
Training Fold (Age >= 9): 14910 startups
Validation Fold (Age = 8): 2073 startups
Holdout Test Set (Age 5-7): 6475 startups

Running rigorous hyperparameter optimization... (This may take a few minutes)
Fitting 1 folds for each of 50 candidates, totalling 50 fits

--- BEST HYPERPARAMETERS ---
{'subsample': 0.8, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.2, 'gamma': 0, 'colsample_bytree': 0.8}

--- EVALUATING OPTIMIZED MODEL ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.80      0.79      0.79      3793
           1       0.70      0.72      0.71      2682

    accuracy                           0.76      6475
   macro avg       0.75      0.75      0.75      6475
weighted avg       0.76      0.76      0.76      6475


##### Grid Search 
better class 1 recall than bayesian (+0.01), but less efficient 
Hyperparameter Tuning Setup (XGBoost Grid Search):
Training Fold (Age >= 9): 14910 startups
Validation Fold (Age = 8): 2073 startups
Holdout Test Set (Age 5-7): 6475 startups

Running exhaustive Grid Search for XGBoost... (~108 fits)
Fitting 1 folds for each of 108 candidates, totalling 108 fits

--- BEST HYPERPARAMETERS (XGBOOST GRID SEARCH) ---
{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 2, 'n_estimators': 600, 'subsample': 0.8}

--- EVALUATING XGBOOST GRID SEARCH ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.81      0.79      0.80      3793
           1       0.72      0.73      0.72      2682

    accuracy                           0.77      6475
   macro avg       0.76      0.76      0.76      6475
weighted avg       0.77      0.77      0.77      6475

##### Bayesian Optimization
better efficiency than gs, 211 trees vs 600 trees with gs

--- BEST HYPERPARAMETERS (OPTUNA) ---
{'max_depth': 4, 'learning_rate': 0.11309019783420737, 'subsample': 0.9637046596596964, 'colsample_bytree': 0.9804237799651304, 'min_child_weight': 6}
Optimal number of trees (found via early stopping): 211

--- EVALUATING OPTUNA MODEL ON UNTOUCHED TEST SET ---
              precision    recall  f1-score   support

           0       0.80      0.80      0.80      3793
           1       0.72      0.72      0.72      2682

    accuracy                           0.77      6475
   macro avg       0.76      0.76      0.76      6475
weighted avg       0.77      0.77      0.77      6475


#### Applying Feature Importance and SHAP to the best models: bayesian optimized RF and XGB

![image](../Reports/Top%2015%20Features%20-%20Random%20Forest%20(Built-in).png)
![image](../Reports/Top%2015%20Features%20-%20XGBoost%20(Built-in).png)
![image](../Reports/SHAP%20Summary%20-%20Random%20Forest%20(Predicting%20Success).png)
![image](../Reports/SHAP%20Summary%20-%20XGBoost%20(Predicting%20Success).png)



Top global predictors: 
1. Log_Funding_Total_USD (log transformed total funding amount, more funding leads to success)
2. Total_Rounds (total funding rounds, the more funding rounds, the higher the chance for success)
3. Num_Investors (high number of investors leads to success)




#### Further improvements

1. Lower dimensionality: 
   - removed sub-industry OHE, since top 60 only covered 75%
   - instead: applied a Out-of-Fold Target Encoding to prevent data leakage and lower dimensionality to only 1 column

2. Patents
   - added col: Has_patents, Num_patents

3. Removed: Months_Since_Last_Funding, Funding_Overdue columns from table (data leakage, but they have never been used for training)


### Performance on improved data set:

--- Best Random Forest Parameters ---
{'n_estimators': 405, 'max_depth': None, 'min_samples_split': 38, 'min_samples_leaf': 12, 'max_features': 0.31460194724579776}

--- Best XGBoost Parameters ---
{'max_depth': 4, 'learning_rate': 0.025456666165296526, 'subsample': 0.6398681324505449, 'colsample_bytree': 0.7392042533970905, 'min_child_weight': 11}
Optimal number of trees (via early stopping): 354



--- Random Forest ---
ROC-AUC Score: 0.8266
              precision    recall  f1-score   support

           0       0.79      0.82      0.80      3793
           1       0.73      0.69      0.71      2682

    accuracy                           0.77      6475
   macro avg       0.76      0.75      0.76      6475
weighted avg       0.76      0.77      0.76      6475


--- XGBoost ---
ROC-AUC Score: 0.8252
              precision    recall  f1-score   support

           0       0.79      0.81      0.80      3793
           1       0.72      0.69      0.71      2682

    accuracy                           0.76      6475
   macro avg       0.76      0.75      0.75      6475
weighted avg       0.76      0.76      0.76      6475


#### Comparision with Neural Network 

Best Parameters found by Optuna:
  n_layers: 3
  dropout_rate: 0.23623982311067843
  learning_rate: 0.004428709587908151
  l2_reg: 0.0007675234496313034
  units_layer_0: 128
  units_layer_1: 112
  units_layer_2: 96
  batch_size: 64
  threshold: 0.4120038215734789

  ==================================================
FINAL HOLDOUT EVALUATION (ADVANCED NEURAL NETWORK)
==================================================
ROC-AUC Score: 0.8228
              precision    recall  f1-score   support

           0       0.81      0.76      0.78      3793
           1       0.68      0.74      0.71      2682

    accuracy                           0.75      6475
   macro avg       0.74      0.75      0.75      6475
weighted avg       0.75      0.75      0.75      6475


=> Achieves same ROC-AUC Score as XGB and RF do ~0.82
