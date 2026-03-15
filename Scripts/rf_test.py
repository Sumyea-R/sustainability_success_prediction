import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import os


current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
input_file = os.path.join(data_clean_folder, "dealroom_clean_features.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')


# Drop Name if it exists
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['IS_SUCCESS'])
y = df['IS_SUCCESS']

# Ensure 'ID' is dropped if it somehow became a feature column
if 'ID' in X.columns:
    X = X.drop(columns=['ID'])

# 3. Split the data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, 
                                  random_state=42, 
                                  class_weight='balanced',
                                  n_jobs=-1)

print("Training the Random Forest model...")
rf_model.fit(X_train, y_train)

# 5. Evaluate with Default Threshold (0.50)
y_pred_default = rf_model.predict(X_test)
print("\n--- Classification Report (Default Threshold = 0.50) ---")
print(classification_report(y_test, y_pred_default))

# 6. Evaluate with Custom Threshold (0.35)
# Get the raw probabilities for Class 1 (Success)
y_probs = rf_model.predict_proba(X_test)[:, 1]

custom_threshold = 0.35
y_pred_custom = (y_probs >= custom_threshold).astype(int)

print(f"\n--- Classification Report (Custom Threshold = {custom_threshold}) ---")
print(classification_report(y_test, y_pred_custom))

# 7. Extract and Visualize Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 15 Most Important Features ---")
print(feature_importance_df.head(15))



# Plot 1: Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), hue='Feature', palette='viridis', legend=False)
plt.title('Top 20 Feature Importances - Startup Success Prediction', fontsize=16)
plt.xlabel('Relative Importance (Gini)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show(block=False) # block=False lets the second plot render at the same time

# Plot 2: Precision-Recall Tradeoff
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend(loc="center left")
plt.title("Precision vs Recall Tradeoff")
plt.grid(True)
plt.show()