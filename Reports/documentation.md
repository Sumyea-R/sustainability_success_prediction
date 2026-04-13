
# Sustainability Success Prediction
This is my final process documentation predicting sustainable startup‘s success and identifying key-characteristics that led to their success (high-growth and/or IPO/M&A), based on dealroom data that contains roughly 26000 entries.

# 1. EDA 
To get an overview over my data, I performed an exploratory data analysis (EDA) on the data set that is the base for this study and the source of data for the ML models that shall be trained to predict startup success. The dataset acquired from Dealroom contains 26503 startup entries with 135 columns each. Not all columns will contain necessary information for my study, hence the data set requires prior selection of 'useful' columns and further cleaning.

...

# 2. Data Cleaning & Target Variable
The data cleaning process is necessary and the most time-consuming task in machine learning since useful predictions need useful training data. 

## Initial selection of useful columns
First, I started with reducing the data complexity by removing columns that:
1. Contain plain text information, as my classifcation models are bad at handling text
2. Are identifiers such as website URLs,
3. Are too detailed such as postal codes of headquarters, details about the founding team, valuations in currencies other than USD etc.,
4. Are Scores and ranks given by Dealroom since I have no information about how these were calculated,
5. Or Contain excessive missing values.

## Target variable calculation
Financial data such as profit is highly distorted due to Corona pandemic that lead to negative margins. Hence, I chose employee number over revenue: the values are not as volatile and profitable startups could still see growth during the pandemic.
Final success rule: (IPO) or (M&A) or ((operational status) and (>20% lifetime employee CAGR for startups with at least 10 employees)).
This finds startups with successful exits (which is the academia standard) and high-growth operating startups based on OECD's definition of high-growth with >20% CAGR. This study applies this rule to the total lifetime of a startup because the original rule by OECD focusses on the last 3 years, but our dataset contains older, acquired startups and lacks recent data density for the last 3 years.

### IPO and M&A
To examine whether a startup had a succesful exit, I checked the 'last_round' column for 'IPO' and keywords indicating M&A such as 'ACQUISITION', 'MERGER', 'BUYOUT'. To ensure these merges and acquisitions were based on 'success', I checked the last 3 round‘s for 'BANKRUPTCY' and excluded the ones acquired because of bankruptcy.

### Lifetime CAGR Calculation
The lifetime CAGR was calculated as follows:
1. Number of founders as initial number of employees, at least 1 if missing value.
2. Last number of employees
3. Time period: 2026 - founding_year >= 5 (as successful startups need to survive at least 5 years before being taken into account for this study to prevent model bias)
cagr = (number_employees / number_founders) ** (1 / number_years) - 1


All successful startups were labeled with '1' in 'IS_SUCCESS' column, otherwise received the entry '0'.
Distribution after labeling:

High Growth (CAGR):    8921
Successful Exits:      2324
TOTAL SUCCESSES (1):   10014
------------------------------
Class Distribution:
IS_SUCCESS
0    0.622156
1    0.377844


# 3. Further cleaning and feature engineering
After having a first understanding of success distribution, I pursued with feature selection. I evlauated which columns needed to be removed as they are post target variable values and which are not necessary for further feature engineering.
First, I removed duplicate data, focussing on keeping the newest data for a startup (sorted by last funding date and total funding descending under duplicate values) while I identified a startup as unique when it had the same name + founding year + HQ country

I kept ID and Name as index
For every startup that contains 0 founders, I clipped it to 1 as lower bound as every startup needs to have at least 1 founder

I counted number of entries in Num_Investors, Num_Founders, Num_SDGs
I mapped Is_Serial_Founder and Is_Top_Uni_Founder to binary, 1 if at least 1x yes, else 0 => rename to Has_...
I calculated gender category (unknown (-1), male only (0), diverse with at least 1 female (1)), company age with 2026 as current year, capital intensity, average round size
I encoded multi-label columns using MultiLabelBinarizer: Industries, client model (b2b, b2c), SDGs, SDG core or side value, HQ Region, Top 20 countries
I applied a Out-of-Fold Target Encoding to Sub-Industries to prevent data leakage and lower dimensionality to only 1 column as Top 60 sub-industries covered only ~75% of startups.
I created a feature indicating if funding is undisclosed (investor but no funding) as this might be a different signal than truly having 0 funding with no investors
Added Has_patents column if number of patents > 0

Then I log-transformed Total_Rounds, Avg_Round_Size and Capital_Intensity 



Entries before removing duplicates: 26503

Entries after removing duplicates: 26454

Startups without Top-20 Industry: 4.0%

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


# 4. Model Training
For finding the best machine learning model to make predictions on the dataset, I used Logistic Regression, Random Forest, XGBoost and a Neural Network. 
All models are trained on a time-based train-test-split: training on startups older or equal 9 years, validation on startups with the age of 8 years (to prevent overfitting) and tested on startups 5-7 years old. 
For all models, bayesian hyperparameter optimization with optuna proved to produce the best results.

## Baseline: Logistic Regression

--- Best Logistic Regression Parameters ---
{'C': 0.000715956973380679, 'penalty': 'l2'}


ROC-AUC Score: 0.8234

              precision    recall  f1-score   support

           0       0.77      0.88      0.82      3793
           1       0.78      0.62      0.69      2682

    accuracy                           0.77      6475
   macro avg       0.78      0.75      0.76      6475
weighted avg       0.77      0.77      0.77      6475



## Random Forest

--- Best Random Forest Parameters ---
{'n_estimators': 405, 'max_depth': None, 'min_samples_split': 38, 'min_samples_leaf': 12, 'max_features': 0.31460194724579776}


ROC-AUC Score: 0.8266
              precision    recall  f1-score   support

           0       0.79      0.82      0.80      3793
           1       0.73      0.69      0.71      2682

    accuracy                           0.77      6475
   macro avg       0.76      0.75      0.76      6475
weighted avg       0.76      0.77      0.76      6475


## XGBoost

--- Best XGBoost Parameters ---
{'max_depth': 4, 'learning_rate': 0.025456666165296526, 'subsample': 0.6398681324505449, 'colsample_bytree': 0.7392042533970905, 'min_child_weight': 11}
Optimal number of trees (via early stopping): 354


ROC-AUC Score: 0.8252
              precision    recall  f1-score   support

           0       0.79      0.81      0.80      3793
           1       0.72      0.69      0.71      2682

    accuracy                           0.76      6475
   macro avg       0.76      0.75      0.75      6475
weighted avg       0.76      0.76      0.76      6475


## Neural Network

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


ROC-AUC Score: 0.8228
              precision    recall  f1-score   support

           0       0.81      0.76      0.78      3793
           1       0.68      0.74      0.71      2682

    accuracy                           0.75      6475
   macro avg       0.74      0.75      0.75      6475
weighted avg       0.75      0.75      0.75      6475




## Ranking and Performance Evaluation

1. Random Forest 
 It achieves the highest ROC-AUC (0.8266) and ties for the highest Accuracy (0.77). It offers the most balanced trade-off between Precision (0.73) and Recall (0.69) for Class 1, resulting in a joint-highest F1 score of 0.71.

2. XGBoost
 It performs almost identically to the Random Forest but is just a hair behind across almost all metrics (ROC-AUC drops by 0.0014, Accuracy drops by 0.01).

3. Logistic Regression (The Conservative Predictor)
 Surprisingly, the baseline model is highly competitive. It ties the Random Forest for highest Accuracy (0.77). However, it achieves this by being very conservative: it has the highest Precision (0.78) but the lowest Recall (0.62) for Class 1. It misses a lot of Class 1 cases (high false negatives). But, when it does predict Class 1, it is the most likely to be correct. 

4. Neural Network 
 It has the lowest ROC-AUC (0.8228) and Accuracy (0.75). It also required the most complex tuning (Optuna, 3 layers, specific dropouts). It has the highest Recall (0.74) for Class 1. It achieves this by sacrificing Precision (0.68). 