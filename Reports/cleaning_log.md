use USD as currency!


Original Target Variable:
(IPO) or (M&A) or ((survived 5+ years) and (20% or more revenue growth over 3 years)).

Change to Target Variable:
Financial data such as profit is highly distorted due to Corona pandemic that lead to negative margins. Hence, I prefer employee number over revenue: the values are not as volatile and profitable startups could still see growth during the pandemic.
=>
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)).

for X:
remove all entries where Launch year < 2016
remove all entries where Launch year > 2021 (survival of 5+ years)

for y:
employee growth calculation
IPO/ACQUISITION in Last round column


ACQUISITION: identified problematic case in data
Last round: 
- ACQUISITION;BANKRUPTCY; ... or BUYOUT;BANKRUPTCY; .... no success, acquisition by lenders

Tried... 
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)) target variable and got... 
IS_SUCCESS
0    0.94984
1    0.05016
... distribution with too little success rate. I also tried to lower the CAGR rules but without better results.

Then, I tried...
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)) or (Scale-up (>50 employees as most recent value) as a proxy for long-term survival)
... and found a discrepancy in very high scale-ups (~1400) and only few CAGR (~200). I considered the low data density in Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026) column to be the cause.

So I tried the following: I calculated the dynamic CAGR over all years between the first and the last value I could find. The threshold is still 20% CAGR and >10 employees at the start. This is much more reliable, finds startups that cannot reach the >50 employee threshold but are still significantly growing. This gave me the following distribution of successful startups:

Breakdown: The metrics have a 72,25% overlap (convergent validity)
High Growth (dynamic CAGR):    1040
Successful Exits:      628
Scale-ups (>50 Emp):   1488
TOTAL SUCCESSES (1):   2288
------------------------------
Class Distribution:
IS_SUCCESS
0    0.817252
1    0.182748



After having a first understanding of success distribution, I pursued with feature selection. I evlauated which columns needed to be removed as they are post target variable values and which are not necessary for further feature engineering.
First, I removed duplicate data, focussing on keeping the newest data for a startup (sorted by last funding date and total funding descending under duplicate values) while I identified a startup as unique when it had the same name + founding year + HQ country
I kept ID and Name as index
I counted number of entries in Num_Investors, Num_Founders, Num_SDGs
I mapped Is_Serial_Founder and Is_Top_Uni_Founder to binary, 1 if at least 1x yes, else 0 => rename to Has_...
I calculated gender category (unknown, male only, diverse with at least 1 female), company age with 2026 as current year, capital intensity and average round size
I encoded multi-label columns using MultiLabelBinarizer: Industries and Sub industries, client model (b2b, b2c), SDGs, SDG core or side value, HQ Region, Top 40 countries 
I created a feature indicating if funding is undisclosed (investor but no funding) as this might be a different signal than truly having 0 funding with no investors
I created a feature "Months_Since_Last_Funding" adn "IS_Funding_Overdue" (>24 months)

Put everything together in dealroom_clean_features_final.csv
Console log with a few stats:
Entries before removing duplicates: 12520
Entries after removing duplicates: 12498
Startups without Top-20 Industry: 4.0%
Startups without Top-60 Sub-Industry: 27.1%
Distribution of gender category:
Gender_Category
-1    0.465755
 0    0.385582
 1    0.148664
Name: proportion, dtype: float64
IS_SUCCESS
0    0.817011
1    0.182989
Name: proportion, dtype: float64
--- Country Coverage Check ---
Top countries cover: 95.0%
Startups in 'long tail': 5.0%
------------------------------
ANALYSIS: Recency & Overdue Features
------------------------------
Stats for months without fallback strategy:
count    12498.000000
mean        52.464954
std         33.379692
min          0.000000
25%         25.000000
50%         47.000000
75%         77.000000
max        256.000000
Name: Months_Since_Last_Funding, dtype: float64

Number of 'unknown/fallback' (999): 0 (0.0%)
\Distribution 'Is_Funding_Overdue' (> 24 months):
Is_Funding_Overdue
1    0.75132
0    0.24868
Name: proportion, dtype: float64

Crosstab: Overdue vs. Success:
IS_SUCCESS                 0         1
Is_Funding_Overdue 
                  
0                   0.604569  0.395431
1                   0.887327  0.112673
------------------------------