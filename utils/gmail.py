"""Gmail API helpers: auth, listing messages, fetching and parsing email content."""

from __future__ import annotations

import os
import re
import time
import base64
from datetime import datetime, timezone

from dateutil import tz
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def get_service():
    """Build and return authenticated Gmail API service."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                os.remove("token.json")
                return get_service()
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def decode_b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def extract_text_from_payload(payload: dict) -> str:
    if not payload:
        return ""
    mime = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")
    if mime == "text/plain" and data:
        return decode_b64url(data).decode("utf-8", errors="replace")
    if mime == "text/html" and data:
        html = decode_b64url(data).decode("utf-8", errors="replace")
        return html_to_text(html)
    parts = payload.get("parts", [])
    if parts:
        texts = [extract_text_from_payload(p) for p in parts]
        return "\n".join(t for t in texts if t)
    return ""


def header_value(headers: list[dict], name: str) -> str | None:
    for h in headers or []:
        if h.get("name", "").lower() == name.lower():
            return h.get("value")
    return None


def to_local_iso(ms_since_epoch: str | int, tz_name: str = "Asia/Kolkata") -> str:
    dt = datetime.fromtimestamp(
        int(ms_since_epoch) / 1000.0, tz=timezone.utc
    ).astimezone(tz.gettz(tz_name))
    return dt.isoformat(timespec="seconds")


def get_label_map(service) -> dict[str, str]:
    """Return mapping of Gmail label id -> name."""
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    return {lab["id"]: lab["name"] for lab in labels}


def list_message_ids(
    service,
    q: str = "",
    label_ids: list[str] | None = None,
    max_fetch: int | None = None,
) -> list[str]:
    """Return message IDs matching the query. Pass max_fetch=None to fetch all."""
    ids = []
    page_token = None
    label_ids = label_ids or []
    while True:
        batch = min(500, max_fetch - len(ids)) if max_fetch is not None else 500
        resp = service.users().messages().list(
            userId="me",
            q=q,
            labelIds=label_ids,
            pageToken=page_token,
            maxResults=batch,
        ).execute()
        for m in resp.get("messages", []):
            ids.append(m["id"])
            if max_fetch is not None and len(ids) >= max_fetch:
                return ids
        page_token = resp.get("nextPageToken")
        if not page_token:
            return ids


def get_message_minimal(service, msg_id: str) -> dict:
    return service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
        fields="id,labelIds,internalDate,payload(headers,body,parts,mimeType)",
    ).execute()


def parse_message(record: dict, id_to_name: dict[str, str]) -> dict:
    payload = record.get("payload", {})
    headers = payload.get("headers", [])
    frm = header_value(headers, "From") or ""
    subj = header_value(headers, "Subject") or ""
    internal_date = record.get("internalDate")
    when = to_local_iso(internal_date) if internal_date else ""
    body_text = extract_text_from_payload(payload)
    body_text = re.sub(r"\s+\n", "\n", body_text).strip()
    labels = [id_to_name.get(lid, lid) for lid in record.get("labelIds", [])]
    return {
        "id": record.get("id"),
        "date": when,
        "from": frm,
        "subject": subj,
        "body_text": body_text,
        "labels": labels,
    }


def safe_get_message(service, mid: str, retries: int = 5, sleep_s: float = 2.0) -> dict:
    for attempt in range(retries):
        try:
            return get_message_minimal(service, mid)
        except HttpError as e:
            if e.resp.status in (403, 429):
                time.sleep(sleep_s * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Failed to fetch message {mid} after {retries} retries.")
