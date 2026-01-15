#!/usr/bin/env python
"""Verify the generated dataset."""
import pandas as pd

print("=" * 70)
print("BLITZ DATASET VERIFICATION")
print("=" * 70)

df = pd.read_csv('data/processed/blitz_data_cleaned.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

print(f"\n Blitz Class Distribution:")
dist = df['blitz'].value_counts().sort_index()
for label, count in dist.items():
    pct = 100 * count / len(df)
    print(f"  Class {label}: {count:6d} ({pct:5.1f}%)")

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("  None!")

print(f"\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - Dataset ready for modeling!")
print("=" * 70)
