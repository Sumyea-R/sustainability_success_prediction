Looking for ways to improve model performance.
What I have tried already: 
looking at IS_EXIT exclusively. The result: imbalanced dataset with 0.91% failure and 9% success. Class 1 success prediction: precision up to ~0.25 and recall ~0.5. 

1. Split up IS_SUCCESS into IS_EXIT and IS_HIGH_GROWTH
Calculating high growth status for each company...
Analyzing exit status for each company...
Determining active status for each company...
Breakdown:
High Growth (CAGR):    12231
Successful Exits:      3666
------------------------------
Class Distribution:
IS_EXIT
0    0.909003
1    0.090997
Name: proportion, dtype: float64
IS_HIGH_GROWTH
0    0.696403
1    0.303597
Name: proportion, dtype: float64

2. Rebuild dataset at fixed observation horizon

fixed observation point at age 5 -> Launch year 2021. Cut all startups founded after 2021.

 delete all startups younger than 2021. Then I looked up that only few startups younger than 5 years are also high growth or have an exit. Then I will have to look into the most recent entry of each round type column, look for BUYOUT or ACQUISITION and if that is true look into each round date column and see whether this is between 2022 and 2026. OR for High growth calculate 3 year CAGR by OECD and if >20% and first employee number is above 10 its high growth. For all features now applies that the data must be available at 2021. E.g if time to first funding: if first funding was after 2021 so eg 2023, I cannot include it.
 split up the following single rows into separate ones: 
 Employees(2016..2026): Create columns for Emp_2016 through Emp_2021. Reserve Emp_2022 to Emp_2025 strictly for calculating your High Growth target,

 New Features:
 Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026):  one new column encoded for every year to 2021.
 Profit (USD) (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027): same as Employees,
 Historical valuations - yearly since 2010 (USD M): same as Employees
 Split them and keep the columns for 2016 through 2021. Replace "n/a" with NaN.
 
 Also: Has_Funding_by_2021, Months_to_First_Funding (if funded by 2021)
 change company_age to Age_in_2021

 delete these features: total funding, last round, last funding amount, last funding date, employee latest number, investor names, number of patents 

instead of having one feature for sub industry, use top 20 one hot encoding incliding an 'other' industry.


## Features:
ID	Age_in_2021	Has_Funding_by_2021	Num_Founders	Num_SDGs	Has_Serial_Founder	Has_Top_Uni_Founder	Emp_2020	Emp_2021	Valuation_2019	Valuation_2020	Valuation_2021	Ind_Other	Ind_chemicals	Ind_education	Ind_energy	Ind_engineering_and_manufacturing_equipment	Ind_enterprise_software	Ind_fashion	Ind_fintech	Ind_food	Ind_health	Ind_home_living	Ind_jobs_recruitment	Ind_kids	Ind_marketing	Ind_media	Ind_real_estate	Ind_robotics	Ind_security	Ind_semiconductors	Ind_transportation	Ind_wellness_beauty	SubInd_Other	SubInd_agritech	SubInd_apparel	SubInd_biotechnology	SubInd_clean_energy	SubInd_construction	SubInd_energy_efficiency	SubInd_energy_providers	SubInd_energy_storage	SubInd_food_logistics_delivery	SubInd_health_platform	SubInd_innovative_food	SubInd_logistics_delivery	SubInd_maintenance	SubInd_medical_devices	SubInd_mobility	SubInd_oil_gas	SubInd_vehicle_production	SubInd_waste_solution	SubInd_water	SubInd_wealth_management	Client_b2b	Client_b2c	Rev_manufacturing	Rev_marketplace_&_ecommerce	Rev_saas	SDG_1	SDG_10	SDG_11	SDG_12	SDG_13	SDG_14	SDG_15	SDG_16	SDG_17	SDG_2	SDG_3	SDG_4	SDG_5	SDG_6	SDG_7	SDG_8	SDG_9	SDG_Type_core	SDG_Type_side	Region__Africa	Region__Asia	Region__Europe	Region__North America	Region__Oceania	Region__South America	Country_Australia	Country_Belgium	Country_Canada	Country_China	Country_Denmark	Country_Finland	Country_France	Country_Germany	Country_India	Country_Israel	Country_Italy	Country_Netherlands	Country_Norway	Country_Other	Country_Poland	Country_Portugal	Country_Spain	Country_Sweden	Country_Switzerland	Country_United Kingdom	Country_United States


## 0-4, 5, >= 6 (Age_in_2021)
Hyperparameter Tuning Setup (Chronological Split):
  Training Fold (Age >= 6) : 12238 startups
  Validation Fold (Age == 5): 2169 startups
  Holdout Test Set (Age 0-4): 13532 startups

==================================================
PHASE 5: EVALUATION ON HOLDOUT SET (IS_EXIT)
==================================================
--- Logistic Regression ---
ROC-AUC Score: 0.6080
              precision    recall  f1-score   support

           0       0.97      0.71      0.82     12983
           1       0.06      0.46      0.11       549

    accuracy                           0.70     13532
   macro avg       0.52      0.58      0.46     13532
weighted avg       0.93      0.70      0.79     13532


--- Random Forest ---
ROC-AUC Score: 0.6476
              precision    recall  f1-score   support

           0       0.96      0.94      0.95     12983
           1       0.12      0.19      0.15       549

    accuracy                           0.91     13532
   macro avg       0.54      0.57      0.55     13532
weighted avg       0.93      0.91      0.92     13532


--- XGBoost ---
ROC-AUC Score: 0.6337
              precision    recall  f1-score   support

           0       0.96      0.91      0.93     12983
           1       0.09      0.21      0.12       549

    accuracy                           0.88     13532
   macro avg       0.53      0.56      0.53     13532
weighted avg       0.93      0.88      0.90     13532


--- Neural Network ---
ROC-AUC Score: 0.6251
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     12983
           1       0.14      0.01      0.02       549

    accuracy                           0.96     13532
   macro avg       0.55      0.50      0.50     13532
weighted avg       0.93      0.96      0.94     13532





==================================================
PHASE 5: EVALUATION ON HOLDOUT SET (IS_HIGH_GROWTH)
==================================================
--- Logistic Regression ---
ROC-AUC Score: 0.7083
              precision    recall  f1-score   support

           0       0.96      0.62      0.75     12591
           1       0.12      0.69      0.20       941

    accuracy                           0.62     13532
   macro avg       0.54      0.65      0.48     13532
weighted avg       0.91      0.62      0.71     13532


--- Random Forest ---
ROC-AUC Score: 0.7482
              precision    recall  f1-score   support

           0       0.94      0.94      0.94     12591
           1       0.23      0.26      0.24       941

    accuracy                           0.89     13532
   macro avg       0.59      0.60      0.59     13532
weighted avg       0.89      0.89      0.89     13532


--- XGBoost ---
ROC-AUC Score: 0.7594
              precision    recall  f1-score   support

           0       0.96      0.78      0.86     12591
           1       0.16      0.57      0.25       941

    accuracy                           0.77     13532
   macro avg       0.56      0.68      0.56     13532
weighted avg       0.90      0.77      0.82     13532


--- Neural Network ---
ROC-AUC Score: 0.7065
              precision    recall  f1-score   support

           0       0.97      0.43      0.60     12591
           1       0.10      0.84      0.18       941

    accuracy                           0.46     13532
   macro avg       0.54      0.64      0.39     13532
weighted avg       0.91      0.46      0.57     13532



## 0-2, 3, >=4 (Age_in_2021)
Hyperparameter Tuning Setup (Chronological Split):
  Training Fold (Age >= 6) : 16855 startups
  Validation Fold (Age == 5): 2534 startups
  Holdout Test Set (Age 0-4): 8550 startups

==================================================
PHASE 5: EVALUATION ON HOLDOUT SET (IS_EXIT)
==================================================
--- Logistic Regression ---
ROC-AUC Score: 0.6031
              precision    recall  f1-score   support

           0       0.97      0.73      0.84      8266
           1       0.05      0.43      0.09       284

    accuracy                           0.72      8550
   macro avg       0.51      0.58      0.46      8550
weighted avg       0.94      0.72      0.81      8550


--- Random Forest ---
ROC-AUC Score: 0.6486
              precision    recall  f1-score   support

           0       0.97      0.94      0.95      8266
           1       0.10      0.20      0.13       284

    accuracy                           0.91      8550
   macro avg       0.54      0.57      0.54      8550
weighted avg       0.94      0.91      0.93      8550


--- XGBoost ---
ROC-AUC Score: 0.6615
              precision    recall  f1-score   support

           0       0.97      0.87      0.92      8266
           1       0.07      0.28      0.11       284

    accuracy                           0.85      8550
   macro avg       0.52      0.57      0.52      8550
weighted avg       0.94      0.85      0.89      8550


--- Neural Network ---
ROC-AUC Score: 0.5920
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      8266
           1       0.04      0.01      0.02       284

    accuracy                           0.96      8550
   macro avg       0.51      0.50      0.50      8550
weighted avg       0.94      0.96      0.95      8550



==================================================
PHASE 5: EVALUATION ON HOLDOUT SET (IS_HIGH_GROWTH)
==================================================
--- Logistic Regression ---
ROC-AUC Score: 0.7274
              precision    recall  f1-score   support

           0       0.97      0.63      0.76      7983
           1       0.12      0.71      0.21       567

    accuracy                           0.64      8550
   macro avg       0.54      0.67      0.48      8550
weighted avg       0.91      0.64      0.73      8550


--- Random Forest ---
ROC-AUC Score: 0.7436
              precision    recall  f1-score   support

           0       0.95      0.93      0.94      7983
           1       0.22      0.29      0.25       567

    accuracy                           0.88      8550
   macro avg       0.58      0.61      0.59      8550
weighted avg       0.90      0.88      0.89      8550


--- XGBoost ---
ROC-AUC Score: 0.7690
              precision    recall  f1-score   support

           0       0.97      0.74      0.84      7983
           1       0.15      0.66      0.25       567

    accuracy                           0.73      8550
   macro avg       0.56      0.70      0.54      8550
weighted avg       0.91      0.73      0.80      8550


--- Neural Network ---
ROC-AUC Score: 0.7204
              precision    recall  f1-score   support

           0       0.98      0.23      0.37      7983
           1       0.08      0.93      0.15       567

    accuracy                           0.28      8550
   macro avg       0.53      0.58      0.26      8550
weighted avg       0.92      0.28      0.36      8550
