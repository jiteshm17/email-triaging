"""
Daily job: fetch unread Primary-inbox emails, classify with Ollama, apply Gmail labels.

By default only the Primary tab is fetched (matches the badge count you see in Gmail).
Pass --all-tabs to also include Promotions, Social, and Updates.

Usage:
    python run.py                    # unread Primary-tab emails, no cap
    python run.py --all-tabs         # unread emails from ALL inbox tabs
    python run.py --max 200          # cap at 200 emails
    python run.py --dry-run          # classify only, skip applying labels to Gmail
    python run.py --cache            # skip re-downloading if today's CSV already exists
    python run.py --model llama3:8b  # use a different Ollama model

By default emails are always re-fetched from Gmail. Pass --cache to reuse today's
existing CSV (useful when tweaking the prompt without waiting for a fresh download).
"""

from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime
from typing import get_args

import pandas as pd
from tqdm.auto import tqdm

import urllib.request

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


def _csv_path(all_tabs: bool) -> str:
    """Single output file per scope+date. Classification columns are written in-place."""
    scope = "all" if all_tabs else "primary"
    return f"{DATA_DIR}/gmail_unread_{scope}_{_DATE}.csv"


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
        "--cache", action="store_true",
        help="Load from an existing CSV for today instead of re-fetching from Gmail. "
             "Useful when tweaking the prompt without re-downloading.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, metavar="MODEL",
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def _check_ollama(base_url: str = "http://localhost:11434") -> None:
    """Warn early if Ollama is unreachable — avoids silent None classifications."""
    try:
        urllib.request.urlopen(base_url, timeout=3)
    except Exception as e:
        logger.warning(
            "Ollama not reachable at %s (%s). "
            "All classifications will return None. "
            "If running inside WSL2, see README for mirrored-networking setup.",
            base_url, e,
        )


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


def display_summary(df: pd.DataFrame) -> None:
    """Print a human-readable classification report to stdout / logs."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  CLASSIFICATION RESULTS  ({len(df)} emails)")
    print(sep)

    counts = df["category"].value_counts()
    print("\nCategory breakdown:")
    for cat, n in counts.items():
        bar = "█" * n
        print(f"  {cat:<35} {n:>4}  {bar}")

    display_cols = [c for c in ("sender", "from", "subject", "category", "reason") if c in df.columns]
    print(f"\nFull results ({', '.join(display_cols)}):")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(df[display_cols].to_string(index=True))
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.width")
    print(sep + "\n")


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

    service = get_service()
    id_to_name = get_label_map(service)
    name_to_id = {name: lid for lid, name in id_to_name.items()}

    _check_labels(name_to_id)
    _check_ollama()

    query = "in:inbox is:unread" if args.all_tabs else "in:inbox is:unread category:primary"
    logger.info("Fetching: %s", query)

    csv_path = _csv_path(args.all_tabs)
    os.makedirs(DATA_DIR, exist_ok=True)
    if args.cache and os.path.exists(csv_path):
        logger.info("Cache hit — loading emails from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        if args.cache:
            logger.info("Cache miss (file not found) — fetching from Gmail")
        df = fetch_emails(service, id_to_name, args.max, query)
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d raw emails to %s", len(df), csv_path)

    client = get_ollama_client()
    df = run_classification(df, client, args.model)

    display_summary(df)

    df.to_csv(csv_path, index=False)
    logger.info("Saved %d emails (with classifications) to %s", len(df), csv_path)

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
