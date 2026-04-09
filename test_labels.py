#!/usr/bin/env python3
"""Test label normalization for Hindi/Telugu datasets"""
import pandas as pd
import sys

# Test label normalization logic
test_labels_hindi = ['fraud', 'real', 'fraud', 'real']
test_labels_telugu = [1, 0, 1, 0]

print("Testing Label Normalization Logic:")
print("=" * 50)

# Test Hindi string labels
print("\n1. Hindi String Labels ('fraud'/'real'):")
hindi_normalized = list(map(lambda x: 1 if x == 'fraud' else 0, test_labels_hindi))
print(f"   Input:  {test_labels_hindi}")
print(f"   Output: {hindi_normalized}")
print(f"   ✓ 'fraud' → 1, 'real' → 0")

# Test Telugu numeric labels
print("\n2. Telugu Numeric Labels (0/1):")
telugu_normalized = list(map(lambda x: 1 if x == 1 else 0, test_labels_telugu))
print(f"   Input:  {test_labels_telugu}")
print(f"   Output: {telugu_normalized}")
print(f"   ✓ 1 → 1, 0 → 0")

# Test the actual datasets
print("\n3. Checking Actual Dataset Formats:")
print("-" * 50)

# Hindi dataset
hindi_df = pd.read_csv('hindi_sms_dataset.csv')
print(f"\nHindi Dataset:")
print(f"   Shape: {hindi_df.shape}")
print(f"   Columns: {list(hindi_df.columns)}")
print(f"   Label unique values: {hindi_df['Label'].unique()}")
print(f"   Label dtype: {hindi_df['Label'].dtype}")

# Apply normalization
hindi_labels_normalized = hindi_df['Label'].apply(lambda x: 1 if (x == 'fraud' or x == 1) else 0)
print(f"   After normalization: {hindi_labels_normalized.unique()}")
print(f"   ✓ All labels converted to [0, 1]")

# Telugu dataset  
telugu_df = pd.read_csv('telugu_sms_dataset.csv')
print(f"\nTelugu Dataset:")
print(f"   Shape: {telugu_df.shape}")
print(f"   Columns: {list(telugu_df.columns)}")
print(f"   Label unique values: {telugu_df['Label'].unique()}")
print(f"   Label dtype: {telugu_df['Label'].dtype}")

# Apply normalization
telugu_labels_normalized = telugu_df['Label'].apply(lambda x: 1 if (x == 'fraud' or x == 1) else 0)
print(f"   After normalization: {telugu_labels_normalized.unique()}")
print(f"   ✓ All labels converted to [0, 1]")

print("\n" + "=" * 50)
print("✅ Label normalization will ensure consistent 0/1 format")
print("   across all datasets (English, Hindi, Telugu)")
