import os
import pandas as pd
import csv
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
data_clean_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Clean"))
data_raw_folder = os.path.abspath(os.path.join(current_dir, "..", "Data_Raw"))
input_file = os.path.join(data_clean_folder, "dealroom_post_labels.csv")
input_raw = os.path.join(data_raw_folder, "Dealroom_Companies_SDG.csv")
input_more_raw = os.path.join(data_raw_folder, "Dealroom_More_Companies_SDG.csv")

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file, sep=';')
df2 = df.copy()

df = df[df['IS_EXIT'] == 1]

print(df['Launch year'].describe())
print(df[df['Launch year'] <= 2016].count())

df2 = df2[df2['IS_HIGH_GROWTH'] == 1]

print(df2['Launch year'].describe())
print(df2[df2['Launch year'] <= 2016].count())

import pandas as pd
