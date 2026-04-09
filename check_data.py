import pandas as pd

# Check Telugu dataset
df_telugu = pd.read_csv('telugu_call_dataset.csv')
print("=== TELUGU CALL DATASET ===")
print(f"Total records: {len(df_telugu)}")
print(f"Fraud: {(df_telugu['label'] == 'fraud').sum()}")
print(f"Real: {(df_telugu['label'] == 'real').sum()}")
print(f"Fraud %: {(df_telugu['label'] == 'fraud').sum() / len(df_telugu) * 100:.1f}%")
print()

# Check Hindi dataset
df_hindi = pd.read_csv('hindi_call_records_dataset.csv')
print("=== HINDI CALL DATASET ===")
print(f"Total records: {len(df_hindi)}")
print(f"Fraud: {(df_hindi['label'] == 'fraud').sum()}")
print(f"Genuine: {(df_hindi['label'] == 'genuine').sum()}")
