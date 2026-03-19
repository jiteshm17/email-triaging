"""
Daily job: fetch unread Primary-inbox emails, classify with Ollama, apply Gmail labels.

By default only the Primary tab is fetched (matches the badge count you see in Gmail).
Pass --all-tabs to also include Promotions, Social, and Updates.

Usage:
    python run.py                    # unread Primary-tab emails, no cap
    python run.py --all-tabs         # unread emails from ALL inbox tabs
    python run.py --max 200          # cap at 200 emails
    python run.py --dry-run          # classify only, skip applying labels to Gmail
    python run.py --model llama3:8b  # use a different Ollama model

Raw emails are cached per-scope (primary / all) and date, so re-running skips the
download and only redoes classification — useful when tweaking the prompt.
Delete the matching data/gmail_unread_*.csv to force a fresh fetch.
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


def _csv_paths(count: int, all_tabs: bool) -> tuple[str, str]:
    """Return (raw_path, tagged_path).
    Scope (primary vs all) is encoded in the name so both variants can coexist."""
    scope = "all" if all_tabs else "primary"
    raw = f"{DATA_DIR}/gmail_unread_{scope}_{_DATE}.csv"
    tagged = f"{DATA_DIR}/gmail_unread_{scope}_{_DATE}_{count}_tagged.csv"
    return raw, tagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify unread Gmail emails with Ollama.")
    parser.add_argument(
        "--max", type=int, default=None, metavar="N",
        help="Cap the number of unread emails to fetch (default: no limit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify only — skip applying labels and marking as read in Gmail",
    )
    parser.add_argument(
        "--all-tabs", action="store_true",
        help="Fetch unread emails from all inbox tabs (Promotions, Social, Updates too). "
             "Default: Primary tab only.",
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


def fetch_emails(service, id_to_name: dict, max_emails: int | None, query: str) -> pd.DataFrame:
    ids = list_message_ids(service, q=query, max_fetch=max_emails)
    logger.info("Found %d unread emails to process", len(ids))
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
            body = {"addLabelIds": [label_id], "removeLabelIds": ["UNREAD"]}
            service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
            applied += 1
        except Exception as e:
            errors.append({"id": msg_id, "category": category, "error": str(e)})
    return applied, skipped, errors


def main() -> None:
    args = parse_args()

    service = get_service()
    id_to_name = get_label_map(service)
    name_to_id = {name: lid for lid, name in id_to_name.items()}

    _check_labels(name_to_id)

    query = "in:inbox is:unread" if args.all_tabs else "in:inbox is:unread category:primary"
    logger.info("Fetching: %s", query)

    # Raw cache encodes scope + date so primary vs all-tabs runs don't share a file.
    raw_csv, _ = _csv_paths(0, args.all_tabs)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(raw_csv):
        logger.info("Using cached emails from %s (delete to re-fetch)", raw_csv)
        df = pd.read_csv(raw_csv)
    else:
        df = fetch_emails(service, id_to_name, args.max, query)
        df.to_csv(raw_csv, index=False)
        logger.info("Saved %d raw emails to %s", len(df), raw_csv)

    # Tagged output includes count so re-runs never overwrite each other.
    _, tagged_csv = _csv_paths(len(df), args.all_tabs)
    client = get_ollama_client()
    df = run_classification(df, client, args.model)
    df.to_csv(tagged_csv, index=False)
    logger.info("Saved %d tagged emails to %s", len(df), tagged_csv)

    if not args.dry_run:
        applied, skipped, errors = apply_labels_to_gmail(service, df, name_to_id)
        logger.info("Applied label + mark read: %d | Skipped: %d", applied, skipped)
        if errors:
            logger.warning("Errors: %d", len(errors))
            for e in errors[:5]:
                logger.warning("  %s", e)
    else:
        logger.info("Dry run — skipped applying labels to Gmail.")


if __name__ == "__main__":
    main()
