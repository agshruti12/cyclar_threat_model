from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Concatenate feature CSVs")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    feat_dir = Path(args.features_dir)

    csv_paths = sorted(feat_dir.glob("*.csv"))

    if not csv_paths:
        raise RuntimeError(f"No CSVs found in {feat_dir}")

    dfs = []

    for path in csv_paths:
        print(f"Loading {path.name}")
        df = pd.read_csv(path)

        # Optional: add source file info
        df["source_file"] = path.name

        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    final_df.to_csv(args.out, index=False)

    print(f"Saved combined dataset -> {args.out}")
    print(f"Total rows: {len(final_df)}")


if __name__ == "__main__":
    main()