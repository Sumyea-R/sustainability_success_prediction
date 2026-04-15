import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, '..', 'Data_Clean')
data_file = os.path.join(data_dir, 'dealroom_clean_features_Improved.csv')
output_file = os.path.join(data_dir, 'numerical_features_describe.csv')


# Numerical Features
df = pd.read_csv(data_file, sep=';')
df_subset = df[["Log_Funding_Total_USD", "Total_Rounds", "Log_Avg_Round_Size", "Log_Capital_Intensity", "Company_Age", "Num_Investors", "Num_Founders", "Num_SDGs", "Num_patents"]]

res = df_subset.describe().T
print(res)
# res.to_csv(output_file, index = False, sep=';')


# Countries
# Isolate the Country columns using filter
country_cols = df.filter(regex='^Country_')

# Calculate percentages and sort
country_props = (country_cols.mean() * 100).sort_values(ascending=False)


# Plot
plt.figure(figsize=(12, 5))
sns.barplot(x=country_props.values, y=country_props.index)
plt.title("Top 20 Countries by Proportion and Others")
plt.xlabel("Percentage (%)")

# Add the exact percentage text to the end of each bar
for index, value in enumerate(country_props.values):
    # Places the text slightly to the right of the bar's end
    plt.text(value + 0.3, index, f'{value:.1f}%', va='center', fontsize=10)

# Add a little padding to the right side of the x-axis so the text doesn't get cut off
plt.xlim(0, max(country_props.values) + 5) 

plt.show()


# Industries
# Isolate the Industry OHE columns
industry_cols = df.filter(regex='^Ind_')

# Calculate the percentage (mean * 100) and sort from highest to lowest
industry_percentages = (industry_cols.mean() * 100).sort_values(ascending=False)

# Clean up the labels for the chart by removing the "Ind_" prefix
industry_percentages.index = industry_percentages.index.str.replace('Ind_', '')

plt.figure(figsize=(10, 8)) 

# Create the bar chart
ax = sns.barplot(
    x=industry_percentages.values, 
    y=industry_percentages.index
)

# Add formatting and labels
plt.title("Distribution by Industry", fontsize=14, pad=15)
plt.xlabel("Percentage of Dataset (%)", fontsize=12)


# Add the exact percentage text to the end of each bar
for index, value in enumerate(industry_percentages.values):
    # Places the text slightly to the right of the bar's end
    plt.text(value + 0.3, index, f'{value:.1f}%', va='center', fontsize=10)

# Add a little padding to the right side of the x-axis so the text doesn't get cut off
plt.xlim(0, max(industry_percentages.values) + 5) 
# Ensure nothing gets cut off at the edges
plt.tight_layout()
plt.show()



# SDGs
# Isolate the SDG OHE columns
sdg_cols = df.filter(regex='^SDG_')
if 'SDG_Type_core' in sdg_cols.columns:
    sdg_cols = sdg_cols.drop(columns=['SDG_Type_core', 'SDG_Type_side']) 

# Calculate the percentage
sdg_percentages = (sdg_cols.mean() * 100).sort_values(ascending=False)

# Clean up the labels by removing the "SDG_" prefix (leaving just '1', '2', etc.)
sdg_percentages.index = sdg_percentages.index.str.replace('SDG_', '')

# Define the official SDG mapping dictionary
sdg_dict = {
    '1': 'No Poverty',
    '2': 'Zero Hunger',
    '3': 'Good Health and Well-being',
    '4': 'Quality Education',
    '5': 'Gender Equality',
    '6': 'Clean Water and Sanitation',
    '7': 'Affordable and Clean Energy',
    '8': 'Decent Work and Economic Growth',
    '9': 'Industry, Innovation and Infrastructure',
    '10': 'Reduced Inequalities',
    '11': 'Sustainable Cities and Communities',
    '12': 'Responsible Consumption and Production',
    '13': 'Climate Action',
    '14': 'Life Below Water',
    '15': 'Life on Land',
    '16': 'Peace, Justice and Strong Institutions',
    '17': 'Partnerships for the Goals'
}

# Map the numbers to the actual names using the dictionary
sdg_percentages.index = sdg_percentages.index.map(lambda x: sdg_dict.get(x, x))

plt.figure(figsize=(10, 8))
sns.barplot(x=sdg_percentages.values, y=sdg_percentages.index)
plt.title("All SDGs by Proportion")
plt.xlabel("Percentage (%)")
plt.ylabel("")

# Add the exact percentage text to the end of each bar
for index, value in enumerate(sdg_percentages.values):
    plt.text(value + 0.3, index, f'{value:.1f}%', va='center', fontsize=10)

# Add padding to the right side of the x-axis
plt.xlim(0, max(sdg_percentages.values) + 5) 

plt.tight_layout()
plt.show()