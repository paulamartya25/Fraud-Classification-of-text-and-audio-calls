import pandas as pd

# Check Hindi dataset
hindi_df = pd.read_csv('hindi_sms_dataset.csv')
print('=== HINDI SMS DATASET ===')
print(f'Shape: {hindi_df.shape}')
print(f'Columns: {hindi_df.columns.tolist()}')
print(f'Label values: {hindi_df["Label"].unique()}')
print(f'Label counts:')
print(hindi_df["Label"].value_counts())
print(f'\nFirst 3 rows:')
print(hindi_df.head(3))

# Check Telugu dataset
telugu_df = pd.read_csv('telugu_sms_dataset.csv')
print(f'\n=== TELUGU SMS DATASET ===')
print(f'Shape: {telugu_df.shape}')
print(f'Label values: {telugu_df["Label"].unique()}')
print(f'Label counts:')
print(telugu_df["Label"].value_counts())
