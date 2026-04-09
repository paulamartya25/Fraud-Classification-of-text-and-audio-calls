import pandas as pd

# Check Hindi dataset
hindi_df = pd.read_csv('hindi_sms_dataset.csv')
print('Hindi SMS Dataset:')
print(f'Columns: {hindi_df.columns.tolist()}')
print(f'Label values: {hindi_df["Label"].unique()}')
print(f'Label counts:\n{hindi_df["Label"].value_counts()}')
print(f'\nFirst 5 rows:')
print(hindi_df.head())
