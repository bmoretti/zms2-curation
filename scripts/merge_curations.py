"""
Merge multiple curation files for nuclei that represent the same biological entity.

Useful when tracking errors cause a single biological nucleus to have multiple IDs
across time. Curate each ID separately, then merge them with this script.

Usage:
    python merge_curations.py 75231 75233
    python merge_curations.py 75231 75233 75240  # Can merge any number

Output:
    Creates: manual_points_df_{first_id}_merged.csv
"""

import pandas as pd
import sys
from pathlib import Path


# ============================================================================
# CONFIG - Edit this to match your curate_nucleus_spots.py output directory
# ============================================================================
out_dir = r'/hpc/projects/group.royer/people/bruno.moretti/data/trulive/2024-09-26/sample_3_dapt_20_uM/2024-09-26_191733/spots/manual_curation'


def merge_curations(nucleus_ids):
    """
    Merge curation files for multiple nucleus IDs.

    Args:
        nucleus_ids: List of nucleus IDs to merge

    Returns:
        Merged DataFrame
    """
    output_dir = Path(out_dir)

    # Check all files exist
    missing = []
    for nid in nucleus_ids:
        filepath = output_dir / f'manual_points_df_{nid}.csv'
        if not filepath.exists():
            missing.append(nid)

    if missing:
        print(f"Error: Curation files not found for nucleus IDs: {missing}")
        print(f"Expected location: {output_dir}")
        return None

    # Load and merge
    print(f"\nMerging {len(nucleus_ids)} curation files...")
    dfs = []

    for nid in nucleus_ids:
        filepath = output_dir / f'manual_points_df_{nid}.csv'
        df = pd.read_csv(filepath)
        print(f"  {nid}: {len(df)} points (t={df['t'].min():.0f}-{df['t'].max():.0f})")
        dfs.append(df)

    # Concatenate
    merged_df = pd.concat(dfs, ignore_index=True)

    # Update nucleus_id column to reflect all IDs
    nucleus_ids_str = ','.join(map(str, nucleus_ids))

    # Handle both old format (nucleus_id) and potential new format (nucleus_ids)
    if 'nucleus_id' in merged_df.columns:
        merged_df['original_nucleus_id'] = merged_df['nucleus_id']
        merged_df['nucleus_id'] = nucleus_ids[0]  # Primary ID

    merged_df['merged_nucleus_ids'] = nucleus_ids_str

    # Sort by time for cleaner output
    merged_df = merged_df.sort_values('t').reset_index(drop=True)

    # Save
    primary_id = nucleus_ids[0]
    output_path = output_dir / f'manual_points_df_{primary_id}_merged.csv'
    merged_df.to_csv(output_path, index=False)

    print(f"\n✓ Merged {len(merged_df)} total points")
    print(f"  Time range: t={merged_df['t'].min():.0f}-{merged_df['t'].max():.0f}")
    print(f"  Nucleus IDs: {nucleus_ids_str}")
    print(f"  Saved to: {output_path}")

    return merged_df


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python merge_curations.py <nucleus_id_1> <nucleus_id_2> [nucleus_id_3] ...")
        print("\nExample:")
        print("  python merge_curations.py 75231 75233")
        print("  python merge_curations.py 75231 75233 75240")
        sys.exit(1)

    # Parse nucleus IDs from command line
    try:
        nucleus_ids = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("Error: All arguments must be integer nucleus IDs")
        sys.exit(1)

    if len(nucleus_ids) < 2:
        print("Error: Need at least 2 nucleus IDs to merge")
        sys.exit(1)

    print("="*70)
    print("MERGE CURATIONS")
    print("="*70)
    print(f"Merging nucleus IDs: {nucleus_ids}")
    print(f"Output directory: {out_dir}")

    result = merge_curations(nucleus_ids)

    if result is None:
        sys.exit(1)

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
