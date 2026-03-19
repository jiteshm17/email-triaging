"""
Fetch emails that already have a custom Gmail label (Label_*) and save to CSV.
Used to build a labelled dataset for prompt evaluation and optimisation.

Usage:
    python build_dataset.py                     # scan 5000 recent, save to data/
    python build_dataset.py --max 2000          # scan fewer emails
    python build_dataset.py --query "in:anywhere" --out data/all_labeled.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm

from utils.gmail import (
    get_service,
    get_label_map,
    list_message_ids,
    parse_message,
    safe_get_message,
)

DATA_DIR = "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labelled email dataset from Gmail.")
    parser.add_argument(
        "--max", type=int, default=5000, metavar="N",
        help="Number of recent emails to scan (default: 5000)",
    )
    parser.add_argument(
        "--query", default="in:inbox",
        help='Gmail search query (default: "in:inbox"; use "in:anywhere" for all mail)',
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output CSV path (default: auto-generated with date and count, e.g. data/gmail_tagged_19Mar2026_320.csv)",
    )
    return parser.parse_args()


def get_custom_label_ids(id_to_name: dict[str, str]) -> list[str]:
    """Return label IDs that are user-created (Gmail IDs starting with 'Label_')."""
    return [lid for lid in id_to_name if lid.startswith("Label_")]


def fetch_labeled_emails(
    service, id_to_name: dict, custom_label_ids: list[str], query: str, max_recent: int
) -> pd.DataFrame:
    ids = list_message_ids(service, q=query, max_fetch=max_recent)
    custom_set = set(custom_label_ids)
    rows = []
    for mid in tqdm(ids, desc="Fetching and filtering by label"):
        rec = safe_get_message(service, mid)
        label_ids = rec.get("labelIds") or []
        if not custom_set.intersection(label_ids):
            continue
        row = parse_message(rec, id_to_name)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    service = get_service()
    id_to_name = get_label_map(service)
    custom_label_ids = get_custom_label_ids(id_to_name)
    print(f"Found {len(custom_label_ids)} custom labels. Scanning up to {args.max} emails...")

    df = fetch_labeled_emails(service, id_to_name, custom_label_ids, args.query, args.max)

    out_path = args.out or (
        f"{DATA_DIR}/gmail_tagged_{datetime.now().strftime('%d%b%Y')}_{len(df)}.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} labeled emails to {out_path}")


if __name__ == "__main__":
    main()
