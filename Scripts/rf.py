import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')

# If ID/Name are columns, set them as index so they aren't used as features
if 'ID' in df.columns and 'Name' in df.columns:
    df = df.set_index(['ID', 'Name'])
elif 'ID' in df.columns:
    df = df.set_index('ID')


# Drop variables tied to the 2026 present-day calculation
leaky_columns = ['Months_Since_Last_Funding', 'Is_Funding_Overdue']
df = df.drop(columns=[col for col in leaky_columns if col in df.columns])


# Train on mature startups (Launch <= 2018 -> Age >= 8)
train_mask = df['Company_Age'] >= 8

# Test on younger startups (Launch 2019-2021 -> Age 5 to 7)
test_mask = (df['Company_Age'] >= 5) & (df['Company_Age'] <= 7)

df_train = df[train_mask]
df_test = df[test_mask]

print(f"Training Set (Age >= 8):  {len(df_train)} startups")
print(f"Testing Set  (Age 5-7):   {len(df_test)} startups")
print(f"Excluded     (Age < 5):   {len(df) - len(df_train) - len(df_test)} startups")

# Separate Features and Target
X_train = df_train.drop(columns=['IS_SUCCESS'])
y_train = df_train['IS_SUCCESS']

X_test = df_test.drop(columns=['IS_SUCCESS'])
y_test = df_test['IS_SUCCESS']


print("\nTraining the Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15,            
    random_state=42, 
    class_weight='balanced',  
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))