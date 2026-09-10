import os
import pandas as pd

def main():
    # Setup paths
    # Resolve relative to the file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_clean_folder = os.path.join(project_root, "Data_Clean")

    features_file = os.path.join(data_clean_folder, "dealroom_clean_features_improved_improved.csv")
    labels_file = os.path.join(data_clean_folder, "dealroom_clean_labels.csv")

    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        print("Error: Clean features or labels files not found. Make sure paths are correct.")
        return

    # Load datasets
    print(f"Loading features from: {features_file}")
    df_features = pd.read_csv(features_file, sep=';')
    print(f"Loading labels from: {labels_file}")
    df_labels = pd.read_csv(labels_file, sep=';')

    # Merge on 'ID'
    df = pd.merge(df_features, df_labels, on='ID', how='inner')
    print(f"Merged dataset contains {len(df)} companies.")

    # Check if target columns are present
    exit_col = 'IS_EXIT_WITHIN_5Y_AFTER_OBS'
    growth_col = 'IS_HIGH_GROWTH_WITHIN_3Y_AFTER_OBS'
    funding_col = 'Has_Funding_By_Obs'

    for col in [exit_col, growth_col, funding_col]:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing in the dataset.")
            return

    # Filter by funding status at observation
    df_funded = df[df[funding_col] == 1]
    df_not_funded = df[df[funding_col] == 0]

    total_funded = len(df_funded)
    total_not_funded = len(df_not_funded)

    # Calculate success cases for funded startups
    exits_funded = df_funded[exit_col].sum()
    high_growth_funded = df_funded[growth_col].sum()
    # Unique successes (either exit or high growth or both)
    success_funded = df_funded[(df_funded[exit_col] == 1) | (df_funded[growth_col] == 1)].shape[0]

    # Calculate success cases for unfunded startups for comparison
    exits_not_funded = df_not_funded[exit_col].sum()
    high_growth_not_funded = df_not_funded[growth_col].sum()
    success_not_funded = df_not_funded[(df_not_funded[exit_col] == 1) | (df_not_funded[growth_col] == 1)].shape[0]

    # Print results
    print("\n" + "="*60)
    print("STARTUP SUCCESS ANALYSIS BY FUNDING STATUS (AT AGE 5)")
    print("="*60)
    print(f"TOTAL STARTUPS IN DATASET: {len(df)}")
    print(f"  - Funded by Age 5:     {total_funded} ({total_funded/len(df)*100:.2f}%)")
    print(f"  - Not Funded by Age 5: {total_not_funded} ({total_not_funded/len(df)*100:.2f}%)")
    print("-" * 60)
    
    print(f"SUCCESS METRICS FOR FUNDED STARTUPS ({total_funded} total):")
    if total_funded > 0:
        print(f"  - IPO/M&A Exits (within 5Y post-obs): {exits_funded} ({exits_funded/total_funded*100:.2f}%)")
        print(f"  - High Growth (within 3Y post-obs)  : {high_growth_funded} ({high_growth_funded/total_funded*100:.2f}%)")
        print(f"  - Total Success (Exit OR Growth)     : {success_funded} ({success_funded/total_funded*100:.2f}%)")
    else:
        print("  No funded startups found.")

    print("-" * 60)
    print(f"SUCCESS METRICS FOR NOT FUNDED STARTUPS ({total_not_funded} total):")
    if total_not_funded > 0:
        print(f"  - IPO/M&A Exits (within 5Y post-obs): {exits_not_funded} ({exits_not_funded/total_not_funded*100:.2f}%)")
        print(f"  - High Growth (within 3Y post-obs)  : {high_growth_not_funded} ({high_growth_not_funded/total_not_funded*100:.2f}%)")
        print(f"  - Total Success (Exit OR Growth)     : {success_not_funded} ({success_not_funded/total_not_funded*100:.2f}%)")
    else:
        print("  No unfunded startups found.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
