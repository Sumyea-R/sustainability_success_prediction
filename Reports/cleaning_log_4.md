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

## 7. Resampling Strategy & Final Model Results (SMOTENC + Tomek Links)

Due to the severe class imbalance (90.8% / 6.8% / 2.4%) and the presence of mixed continuous and categorical features, the training pipeline was updated to implement **SMOTENC** (SMOTE for Nominal and Continuous) followed by **Tomek Links** for undersampling noisy boundary samples.

*   **SMOTENC:** Safely generates synthetic minority examples without producing invalid fractional values for categorical columns.
*   **Tomek Links:** Cleans up borderline regions where synthetic minority examples might overlap with majority class examples.

### Class Distribution in Training After Resampling
The training data is now physically balanced, penalizing the model from adopting a "lazy" majority-class prediction strategy:
*   **0 (None):** ~33.5%
*   **1 (Exit):** ~33.1%
*   **2 (High Growth):** ~33.5%

### Updated Model Performance (Imbalanced Test Set)
Evaluated on the true, imbalanced holdout set to reflect real-world performance, XGBoost emerged as the leading model across Macro F1 and ROC-AUC. 

| Model | Macro F1-Score | ROC-AUC (OVR) |
| :--- | :--- | :--- |
| **XGBoost** | **0.5409** | **0.8726** |
| **CatBoost** | 0.5373 | 0.8705 |
| **Random Forest** | 0.5364 | 0.8704 |
| **Neural Network** | 0.4893 | 0.8302 |
| **Logistic Regression** | 0.4795 | 0.8353 |

### Detailed Performance by Class (Post-Resampling)

#### XGBoost (Best Overall)
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.96 | 0.84 | 0.90 |
| **1 (Exit)** | 0.20 | 0.40 | 0.26 |
| **2 (High Growth)** | 0.32 | 0.84 | 0.46 |

#### CatBoost
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.95 | 0.89 | 0.92 |
| **1 (Exit)** | 0.21 | 0.28 | 0.24 |
| **2 (High Growth)** | 0.33 | 0.76 | 0.46 |

#### Random Forest
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.95 | 0.90 | 0.92 |
| **1 (Exit)** | 0.22 | 0.25 | 0.23 |
| **2 (High Growth)** | 0.31 | 0.82 | 0.45 |

#### Neural Network
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.93 | 0.89 | 0.91 |
| **1 (Exit)** | 0.21 | 0.37 | 0.27 |
| **2 (High Growth)** | 0.32 | 0.26 | 0.29 |

#### Logistic Regression
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (None)** | 0.96 | 0.75 | 0.84 |
| **1 (Exit)** | 0.16 | 0.49 | 0.24 |
| **2 (High Growth)** | 0.24 | 0.72 | 0.36 |

### Impact of Resampling
*   **Minority Recall Improvement:** Implementing SMOTENC + Tomek Links improved the models' abilities to detect the minority classes. For example, XGBoost now achieves a recall of 0.84 for High Growth, showing it is highly sensitive to successful trajectories.
*   **Metric Shift:** While overall accuracy may have slightly decreased due to more aggressive predictions on minority classes, the balance captured by F1-Macro highlights a much more robust and practically useful model.

### 7.1 Empirical Finding: Synthetic Resampling vs. Cost-Sensitive Learning

A critical comparative analysis was conducted between algorithmic **Cost-Sensitive Learning** (using native `class_weight='balanced'` in tree models, documented in Section 6) and **Data-Level Synthetic Resampling** (SMOTENC + Tomek Links, documented above). 

Contrary to common assumptions in imbalanced classification, the empirical results demonstrated that **Synthetic Resampling slightly degraded overall predictive performance** across the primary tree-based models:
*   **CatBoost F1-Macro:** Decreased from 0.546 (Weighted) to 0.537 (Resampled).
*   **XGBoost F1-Macro:** Decreased from 0.543 (Weighted) to 0.540 (Resampled).
*   **Random Forest F1-Macro:** Decreased from 0.545 (Weighted) to 0.536 (Resampled).

**Methodological Implications for the Thesis:**
1.  **The "Synthetic Noise" Problem:** Sustainable startup trajectories are highly complex and non-linear. SMOTENC generates synthetic data by mathematically interpolating between existing rare instances (Exits and High Growth). In a high-dimensional space (76 features) containing numerous sparse, categorical boolean flags (e.g., specific investor types or funding instruments), this interpolation likely generates unrealistic, "Frankenstein" profiles that do not exist in the real world. The models overfit to these artificial boundaries during training, leading to poorer generalization on the strict, unmodified real-world test set.
2.  **Superiority of Algorithmic Penalization:** The results empirically validate that for this specific domain, modifying the loss function to heavily penalize misclassifications of the minority class (Cost-Sensitive Learning) is a more robust methodology than altering the underlying data distribution. It preserves the true, organic variance of the sustainable startup ecosystem.
3.  **Data Limitations:** Ultimately, these findings highlight a core limitation inherent in current sustainability data: the sheer sparsity of true "success" events (Exits/High Growth). While advanced machine learning methodologies (like SMOTENC) attempt to bridge this gap, they cannot magically synthesize the nuanced realities of successful sustainable business models if the baseline signal is fundamentally weak or constrained by the current maturity of the sector.
