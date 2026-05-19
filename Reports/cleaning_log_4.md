# Cleaning & Feature Engineering Log - Update 4 (Refined Methodology)

## Overview
This log documents the transition from a fixed calendar year cutoff (2021) to a **Dynamic Observation Horizon** based on company age. This ensures every startup is evaluated at the same stage of its lifecycle (Age 5), eliminating bias caused by varying maturity levels in the training data.

## 1. Data Splitting & Horizon Filtering
*   **Observation Age:** 5 years.
*   **Observation Year:** `Launch year + 5`.
*   **Dataset Filter:** Only companies that reached Age 5 by 2021 are included (`Observation_Year <= 2021`). This ensures a full 5-year post-observation window for exits and a 3-year window for high growth within the available data (up to 2026).
*   **Leaky Columns Removed:** `Launch year` and `Observation_Year` are excluded from features to prevent temporal bias.

## 2. Target Variable Definition (3-Class)
The model predicts the outcome occurring **after** the individual observation year.

| Class | Label | Definition |
| :--- | :--- | :--- |
| **0** | `NOT_EXIT_OR_HIGH_GROWTH` | No exit within 5 years AND no high growth within 3 years. |
| **1** | `IS_EXIT_WITHIN_5Y` | IPO, Acquisition, Merger, or Buyout within `[Obs, Obs + 5]`. |
| **2** | `IS_HIGH_GROWTH_WITHIN_3Y` | Employee CAGR >= 20% within `[Obs, Obs + 3]` (Minimum 10 employees at Obs). |

## 3. Feature Engineering (Point-in-Time)
All features are engineered using only data available on or before the company's specific `Observation_Year`.

### A. Funding Features
*   **Status:** `Has_Funding_By_Obs`, `Num_Rounds_By_Obs`.
*   **Timing:** `Months_To_First_Funding_By_Obs`, `Months_Since_Last_Funding_At_Obs`.
*   **Growth:** `Total_Funding_By_Obs`, `Avg_Round_Size_By_Obs`, `Round_Size_Growth_By_Obs`.
*   **Instruments:** `Has_Grant_By_Obs`, `Has_Debt_By_Obs`, `Has_VC_or_Equity_By_Obs`, and their respective shares.

### B. Investor Features (Strictly Aligned)
Investors are parsed round-by-round to ensure strict point-in-time validity.
*   **Counts:** `Num_Unique_Investors_By_Obs`, `Num_Investor_Types_By_Obs`.
*   **Behavior:** `Has_Repeat_Investor_By_Obs` (Investor participated in >1 round before Obs).
*   **Type Mapping:**
    *   `Has_Corporate_Investor_By_Obs`: Dealroom "corporate"
    *   `Has_CVC_Investor_By_Obs`: Dealroom "corporate_venture_fund"
    *   `Has_Government_Investor_By_Obs`: Dealroom "government"
    *   `Has_University_Investor_By_Obs`: Dealroom "university"
    *   `Has_Accelerator_or_Incubator_By_Obs`: Dealroom "accelerator" or "incubator"
    *   `Has_VC_Investor_By_Obs`: Dealroom "venture_capital"
    *   `Has_Angel_Investor_By_Obs`: Dealroom "angel"

### C. Employee Trajectory
*   **Snapshots:** `Emp_At_Obs`, `Emp_1Y_Before`, `Emp_2Y_Before`, `Emp_3Y_Before`.
*   **Growth Metrics:** `Employee_Growth_1Y_PreObs`, `Employee_Growth_2Y_PreObs`, `Employee_CAGR_PreObs`.
*   **Patterns:** `Employee_Slope`, `Employee_Acceleration`, `Employee_Volatility`, `Employee_Max`, `Employee_Min`.

### D. Founder & Profile Features
*   **Founders:** `Num_Founders`, `Num_Serial_Founders`, `Share_Serial_Founders`, `Num_Top_Uni_Founders`, `Share_Top_Uni_Founders`.
*   **Categorical Encoding:** `SubIndustry_Frequency`, `Country_Frequency`, `Region`.
*   **Log Transforms:** `Log_Num_Patents`, `Log_Num_Investors`, `Log_Founder_Prior_Funding`.

## 4. Missingness Indicators
Boolean flags identify missing data in critical fields at the point of observation:
*   `Missing_Employee_Data`
*   `Missing_Founder_Data`
*   `Missing_Investor_Data`
*   `Missing_SubIndustry`
*   `Missing_Founder_Education/Background`
*   `Missing_Valuation`
*   `Missing_Funding_Round/Amount/Type_Data`

## 5. Summary Statistics
*   **Total Samples:** 15,375
*   **Features:** 76
*   **Class Distribution (Train):**
    *   None (0): 90.8%
    *   Exit (1): 6.8%
    *   High Growth (2): 2.4%

## 6. Preliminary Model Results
The following results were obtained after HPO (Optuna) on the refined 3-class dataset:

| Model | Macro F1-Score | ROC-AUC (OVR) |
| :--- | :--- | :--- |
| **CatBoost** | **0.5461** | **0.8807** |
| **Random Forest** | 0.5457 | 0.8798 |
| **XGBoost** | 0.5438 | 0.8753 |
| **Neural Network** | 0.4670 | 0.8661 |
| **Logistic Regression** | 0.4647 | 0.8339 |

### Detailed Performance by Class (Test Set)

#### CatBoost (Best Overall)
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.97 | 0.78 | 0.86 |
| **1 (Exit)** | 0.19 | 0.58 | 0.28 |
| **2 (High Growth)** | 0.33 | 0.92 | 0.49 |

#### Random Forest
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.97 | 0.82 | 0.89 |
| **1 (Exit)** | 0.20 | 0.43 | 0.27 |
| **2 (High Growth)** | 0.31 | 1.00 | 0.48 |

#### XGBoost
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.93 | 0.98 | 0.95 |
| **1 (Exit)** | 0.39 | 0.12 | 0.18 |
| **2 (High Growth)** | 0.52 | 0.47 | 0.50 |

#### Neural Network
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.98 | 0.62 | 0.76 |
| **1 (Exit)** | 0.14 | 0.74 | 0.24 |
| **2 (High Growth)** | 0.28 | 0.73 | 0.40 |

#### Logistic Regression
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.97 | 0.71 | 0.82 |
| **1 (Exit)** | 0.14 | 0.51 | 0.23 |
| **2 (High Growth)** | 0.22 | 0.91 | 0.35 |

### Observations:
*   Tree-based models (CatBoost, RF, XGB) significantly outperform the linear baseline and neural network on this tabular dataset.
*   CatBoost achieved the highest overall performance, successfully handling categorical features and imbalanced classes.
*   The high ROC-AUC scores (0.83 - 0.88) indicate strong separability, though the Macro F1 reflects the challenge of predicting rare success events (Classes 1 and 2).
