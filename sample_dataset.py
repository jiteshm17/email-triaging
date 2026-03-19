"""
Sample N random rows per main_label from a tagged emails dataset and save to CSV.

Usage:
    python sample_labeled_emails.py <input_path> [--data-dir DIR] [--n N] [--seed SEED]

Required:
    input_path          Path to the tagged emails CSV (e.g. data/gmail_tagged_emails.csv)

Optional:
    --data-dir DIR      Directory where the output CSV is saved (default: data)
    --n N               Rows to sample per label (default: 20)
    --seed SEED         Random seed (default: 42; try 0 for a completely different sample)
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

COLUMNS = ["id", "date", "from", "subject", "body_text", "category", "reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample N rows per label from a tagged emails CSV.")
    parser.add_argument("input_path", help="Path to the input tagged emails CSV")
    parser.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--n", type=int, default=20, help="Rows to sample per label (default: 20)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42; use 0 for a different sample)"
    )
    return parser.parse_args()


def derive_output_path(input_path: str, data_dir: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(data_dir, f"{base}_sampled.csv")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    df = pd.read_csv(args.input_path)
    available = [c for c in COLUMNS if c in df.columns]
    missing = set(COLUMNS) - set(available)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    df = df[available].copy()

    sampled = (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(n=min(args.n, len(g)), random_state=args.seed))
        .reset_index(drop=True)
    )

    output_path = derive_output_path(args.input_path, args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)
    sampled.to_csv(output_path, index=False)
    print(f"Saved {len(sampled)} rows ({sampled['category'].nunique()} labels) → {output_path}")


if __name__ == "__main__":
    main()
