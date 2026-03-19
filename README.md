# Email Triaging

Fetches recent emails from Gmail and tags each with a category using a local LLM (Ollama). Applies the category as a Gmail label

## What it does

1. **Fetch** – Connects to Gmail (OAuth), lists recent messages in the inbox, and downloads each message’s headers and body.
2. **Tag** – Sends subject + body to Ollama (OpenAI-compatible API) and gets a single category and short reason per email.
3. **Save** – Writes a raw CSV of fetched emails and a tagged CSV with `category` and `reason`.
4. **Apply** – For each tagged email, adds the category as a Gmail label

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Gmail OAuth

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable the Gmail API.
2. Create OAuth 2.0 credentials (Desktop app), download as **`credentials.json`**, and place it in the project root.
3. On the **first run**, a browser window will open for you to sign in. After authorising, a **`token.json`** is created automatically — this stores your access token so you don't need to sign in again.

> `credentials.json` is the app identity (safe to keep private, never changes).  
> `token.json` is your personal access token (auto-refreshed when it expires).  
> Both are listed in `.gitignore` and will never be committed.

### 3. Ollama

Install [Ollama](https://ollama.ai), then pull the default model:

```bash
ollama pull qwen2.5:14b
```

### 4. Gmail labels

In Gmail Settings → Labels, create labels whose names **exactly match** the category names defined in `utils/prompts.py` (e.g. `TRANSACTION_ALERT`, `OTP_AND_VERIFICATION`, `ADS`). The script prints which labels are missing on startup — any missing categories are still classified but the label won't be applied in Gmail.

---

## Usage

### Daily run — tag all unread emails

```bash
python run.py
```

Fetches every unread inbox email, classifies with Ollama, applies labels, marks as read.

| Flag | Default | Description |
|---|---|---|
| `--max N` | None (all) | Cap the number of unread emails |
| `--dry-run` | off | Classify only, skip Gmail updates |
| `--model MODEL` | `qwen2.5:14b` | Ollama model to use |

- `fetch_recent_emails.py` – Entry point: fetch → classify → save → optionally apply labels.
- `gmail_utils.py` – Gmail API auth, listing/fetching messages, parsing payloads.
- `classifier.py` – Ollama client and `classify_email()` using structured output.
- `prompts.py` – System prompt and Pydantic schema (categories and `reason_short`).
   ```
