## **1\. Define the Target Variable**

Your research question is about long-term sustainability success. Common Target Variable Option to Consider.

### **Binary Classification**

**Target:** `Sustainable_Success` (0 / 1\)

Examples of rules:

* 1 \= Startup has **survived ≥ X years** (e.g., 5–7 years)  
* AND/OR reached **Series B+**  
* AND/OR achieved **significant revenue growth**  
* AND/OR exit (IPO / acquisition)  
* 0 \= Otherwise (failed, inactive, stagnant)

## **2\. Features to Consider**

### **A. Founding Team Characteristics**

Examples:

* Number of founders  
* % founders with prior startup experience  
* % founders with prior exits  
* % founders with technical background  
* % founders with sustainability / environmental background  
* Founder age (mean / median, if available)  
* Gender diversity indicators  
* Education level (top university indicator)

### **B. Funding History & Financial Signals**

Examples:

* Total funding amount  
* Number of funding rounds  
* Time between rounds  
* Early VC involvement (binary)  
* Presence of impact / ESG investors  
* Public vs private funding  
* Average round size  
* Funding velocity (funding per year)

Note: Avoid **data leakage**; Use only **early-stage funding features** (e.g., first 2–3 years).

**C. Sustainability & SDG Signals**

Examples:

* SDG category (one-hot encoded)  
* Number of SDGs addressed  
* Sustainability sector (clean energy, agri-tech, circular economy)

This is where this study differentiates itself from generic startup success models. (will be more clear after a comprehensive literature review)

### **D. Innovation & Intellectual Property**

Examples:

* Patent count

### **E. Market & Sector Characteristics**

Examples:

* Primary sector (one-hot encoded)  
* Sub-sector  
* B2B vs B2C  
* Capital intensity (Total\_Funding / Years\_Since\_Founding) indicator

### **F. Geography & Ecosystem**

Examples:

* Country  
* Region (EU / US / Asia)

### **G. Early Performance Signals**

Examples:

* Employee growth rate (first 2–3 years)  
* Web traffic growth  
* Revenue growth (early only)  
* Partnerships announced

Exclude features that occur **after** your success label.

## **4\. Step-by-Step Data Cleaning & Processing Pipeline**

### **Step 1: Initial Dataset Audit**

* Check:  
  * Missing values per column  
  * Data types  
  * Duplicate rows  
  * Time coverage

Remove columns that:

* Are identifiers only (URLs)   
* Are post-outcome variables  
* Contain excessive missing values (\>60–70%)

Note: Keep the IDs, Names to identify the startup, do not use them as features

### **Step 2: Handle Missing Values**

* Numerical:  
  * Median imputation (preferred for tree models)  
* Categorical:  
  * “Unknown” category  
* Binary flags:  
  * Convert NaN → 0 if logically valid

### **Step 3: Outlier Treatment**

* Log-transform:  
  * Funding amounts  
  * Valuations  
* Winsorize extreme values if needed  
* Inspect long-tailed distributions

### **Step 4: Encode Categorical Variables**

* One-hot encoding for:  
  * Sector  
  * SDG  
  * Country / region

### **Step 5: Feature Engineering**

Create **theoretically meaningful features**, e.g.:

* Funding per year  
* Founder experience score  
* Sustainability focus index  
* Early growth momentum

### **Step 6: Remove Leakage & Temporal Bias**

* Ensure all features are:  
  * Available **before** the success outcome  
* Fix a **cut-off time** (e.g., first 3 years after founding)

### **Step 7: Train-Test Split**

* Time-based split preferred:  
  * Older startups → train  
  * Newer startups → test  
* Or stratified split if time info is limited

### **Step 8: Model Training**

* Random Forest (baseline)  
* XGBoost / Gradient Boosting (main model)  
* Handle class imbalance:  
  * Class weights  
  * SMOTE

### **Step 9: SHAP Interpretability**

Use SHAP to:

* Identify top global predictors  
* Compare:  
  * Sustainability vs non-sustainability drivers  
  * Regional differences  
* Support your **contribution claims**

