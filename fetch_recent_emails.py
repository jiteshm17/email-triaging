"""
Fetch recent Gmail messages, tag each with a category using a local LLM (Ollama), and optionally apply labels in Gmail.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm.auto import tqdm

from gmail_utils import (
    get_service,
    get_label_map,
    list_message_ids,
    parse_message,
    safe_get_message,
)
from classifier import get_ollama_client, classify_email, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Config (adjust as needed) ---
MAX_EMAILS = 500
QUERY = "in:inbox"
RAW_CSV_PATH = "gmail_recent_500.csv"
TAGGED_CSV_PATH = "gmail_recent_500_tagged.csv"
APPLY_TO_GMAIL = True  # Set False to only save CSVs and skip Gmail label/read updates


def fetch_emails(service, id_to_name: dict, query: str, max_emails: int) -> pd.DataFrame:
    """Fetch message IDs, download each message, parse into rows, return DataFrame."""
    ids = list_message_ids(service, q=query, label_ids=None, max_fetch=max_emails)
    rows = []
    for mid in tqdm(ids, desc="Downloading emails"):
        rec = safe_get_message(service, mid)
        row = parse_message(rec, id_to_name)
        rows.append(row)
    return pd.DataFrame(rows)


def run_classification(df: pd.DataFrame, client, model: str) -> pd.DataFrame:
    """Add category and reason columns using the LLM. Mutates and returns df."""
    tqdm.pandas(desc="Classifying emails")

    def classify_row(row):
        cat, reason = classify_email(
            client, row["subject"], row["body_text"], model=model
        )
        return pd.Series({"category": cat, "reason": reason})

    out = df.progress_apply(classify_row, axis=1)
    df["category"] = out["category"]
    df["reason"] = out["reason"]
    return df


def apply_labels_to_gmail(service, df: pd.DataFrame, name_to_id: dict) -> tuple[int, int, list]:
    """Apply category label and mark as read for each row. Returns (applied, skipped, errors)."""
    applied = 0
    skipped = 0
    errors = []
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
    service = get_service()
    id_to_name = get_label_map(service)
    name_to_id = {name: lid for lid, name in id_to_name.items()}

    df = fetch_emails(service, id_to_name, QUERY, MAX_EMAILS)
    df.to_csv(RAW_CSV_PATH, index=False)

    client = get_ollama_client()
    df = run_classification(df, client, DEFAULT_MODEL)
    df.to_csv(TAGGED_CSV_PATH, index=False)

    if APPLY_TO_GMAIL:
        applied, skipped, errors = apply_labels_to_gmail(service, df, name_to_id)
        logger.info("Applied label + mark read: %d | Skipped: %d", applied, skipped)
        if errors:
            logger.warning("Errors: %d", len(errors))
            for e in errors[:5]:
                logger.warning("  %s", e)


if __name__ == "__main__":
    main()
