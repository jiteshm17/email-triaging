"""
Fetch recent inbox emails (read + unread), classify with Ollama, apply Gmail labels.
Useful for a one-off backfill rather than the daily unread-only run.

Usage:
    python tag_recent.py                   # tag 1000 most recent inbox emails
    python tag_recent.py --max 500         # tag 500 most recent
    python tag_recent.py --dry-run         # classify only, skip Gmail updates
    python tag_recent.py --model llama3:8b # use a different Ollama model
"""

from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime
from typing import get_args

import pandas as pd
from tqdm.auto import tqdm

from utils.gmail import (
    get_service,
    get_label_map,
    list_message_ids,
    parse_message,
    safe_get_message,
)
from utils.classifier import get_ollama_client, classify_email, DEFAULT_MODEL
from utils.prompts import Category

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
_DATE = datetime.now().strftime("%d%b%Y")   # e.g. 19Mar2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify recent Gmail emails with Ollama.")
    parser.add_argument(
        "--max", type=int, default=1000, metavar="N",
        help="Number of recent emails to fetch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify only — skip applying labels and marking as read in Gmail",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, metavar="MODEL",
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def _check_labels(name_to_id: dict) -> None:
    categories = list(get_args(Category))
    present = [c for c in categories if c in name_to_id]
    missing = [c for c in categories if c not in name_to_id]
    logger.info("Gmail labels found (%d/%d): %s", len(present), len(categories), present)
    if missing:
        logger.warning(
            "Gmail labels MISSING (%d) — classified but NOT applied to inbox: %s",
            len(missing), missing,
        )


def fetch_emails(service, id_to_name: dict, max_emails: int) -> pd.DataFrame:
    ids = list_message_ids(service, q="in:inbox", max_fetch=max_emails)
    logger.info("Fetching %d recent emails", len(ids))
    rows = []
    for mid in tqdm(ids, desc="Downloading emails"):
        rec = safe_get_message(service, mid)
        row = parse_message(rec, id_to_name)
        rows.append(row)
    return pd.DataFrame(rows)


def run_classification(df: pd.DataFrame, client, model: str) -> pd.DataFrame:
    tqdm.pandas(desc="Classifying emails")

    def classify_row(row):
        cat, reason = classify_email(client, row["subject"], row["body_text"], model=model)
        return pd.Series({"category": cat, "reason": reason})

    out = df.progress_apply(classify_row, axis=1)
    df["category"] = out["category"]
    df["reason"] = out["reason"]
    return df


def apply_labels_to_gmail(service, df: pd.DataFrame, name_to_id: dict) -> tuple[int, int, list]:
    applied, skipped, errors = 0, 0, []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Applying to Gmail"):
        msg_id = row.get("id")
        category = row.get("category")
        if pd.isna(category) or not category:
            skipped += 1
            continue
        label_id = name_to_id.get(category)
        if not label_id:
            skipped += 1
            continue
        try:
            body = {"addLabelIds": [label_id]}
            service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
            applied += 1
        except Exception as e:
            errors.append({"id": msg_id, "category": category, "error": str(e)})
    return applied, skipped, errors


def main() -> None:
    args = parse_args()

    raw_csv = f"{DATA_DIR}/gmail_recent_{_DATE}_{args.max}.csv"

    service = get_service()
    id_to_name = get_label_map(service)
    name_to_id = {name: lid for lid, name in id_to_name.items()}

    _check_labels(name_to_id)

    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(raw_csv):
        logger.info("Using cached emails from %s (delete to re-fetch)", raw_csv)
        df = pd.read_csv(raw_csv)
    else:
        df = fetch_emails(service, id_to_name, args.max)
        df.to_csv(raw_csv, index=False)
        logger.info("Saved %d raw emails to %s", len(df), raw_csv)

    # Include actual email count in tagged filename so re-runs don't overwrite.
    tagged_csv = f"{DATA_DIR}/gmail_recent_{_DATE}_{len(df)}_tagged.csv"
    client = get_ollama_client()
    df = run_classification(df, client, args.model)
    df.to_csv(tagged_csv, index=False)
    logger.info("Saved %d tagged emails to %s", len(df), tagged_csv)

    if not args.dry_run:
        applied, skipped, errors = apply_labels_to_gmail(service, df, name_to_id)
        logger.info("Applied label: %d | Skipped: %d", applied, skipped)
        if errors:
            logger.warning("Errors: %d", len(errors))
            for e in errors[:5]:
                logger.warning("  %s", e)
    else:
        logger.info("Dry run — skipped applying labels to Gmail.")


if __name__ == "__main__":
    main()
