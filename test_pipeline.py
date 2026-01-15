#!/usr/bin/env python
"""Quick test of the data pipeline."""
import sys
sys.path.insert(0, '.')

from src.data.blitz_pipeline import data_acquisition

print("Running full blitz data pipeline...")
print("=" * 60)

try:
    # Run pipeline for 2022 season only (faster)
    print("\nLoading 2022 NFL play-by-play data...")
    data, metadata = data_acquisition([2022])
    
    print(f"\n✓ Successfully loaded {len(data)} plays")
    print(f"\n✓ Columns ({len(data.columns)}): {list(data.columns)}")
    print(f"\n✓ Data saved to: {metadata.get('data_file', 'N/A')}")
    print(f"\n✓ Class Distribution:\n{metadata.get('class_distribution', 'N/A')}")
    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

