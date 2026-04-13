import pandas
import os

current_dir = os.path.dirname(__file__)
raw_data_dir = os.path.join(current_dir, '..', 'Data_Raw')
raw_data_path = os.path.join(raw_data_dir, 'Dealroom_Companies_SDG.csv')

df = pandas.read_csv(raw_data_path)
print(df.shape)