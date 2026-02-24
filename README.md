# Email Triaging

Fetches recent emails from Gmail and tags each with a category using a local LLM (Ollama). Optionally applies the category as a Gmail label and marks messages as read.

## What it does

1. **Fetch** – Connects to Gmail (OAuth), lists recent messages in the inbox, and downloads each message’s headers and body.
2. **Tag** – Sends subject + body to Ollama (OpenAI-compatible API) and gets a single category and short reason per email.
3. **Save** – Writes a raw CSV of fetched emails and a tagged CSV with `category` and `reason`.
4. **Apply (optional)** – For each tagged email, adds the category as a Gmail label and marks the message as read (same idea as an n8n “Add label → Mark as read” flow).

## Setup

1. **Python** – Use a venv and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Gmail** – In Google Cloud Console, create OAuth credentials (Desktop app), download as `credentials.json`, and place it in the project root. On first run you’ll sign in in the browser; a `token.json` will be created.

3. **Ollama** – Install [Ollama](https://ollama.ai), then pull the model used by the script (default: `qwen2.5:14b`):
   ```bash
   ollama pull qwen2.5:14b
   ```

4. **Gmail labels** – In Gmail, create labels whose **names** exactly match the categories (e.g. `TRANSACTION_ALERT`, `ORDERS_SUBSCRIPTIONS`, `OTHERS`). Only messages whose predicted category matches an existing label will get that label applied.

## Run

From the project root:

```bash
python fetch_recent_emails.py
```

Config is at the top of `fetch_recent_emails.py`: `MAX_EMAILS`, `QUERY`, output paths, and `APPLY_TO_GMAIL` (set to `False` to only generate CSVs and skip Gmail updates).

## Project layout

- `fetch_recent_emails.py` – Entry point: fetch → classify → save → optionally apply labels.
- `gmail_utils.py` – Gmail API auth, listing/fetching messages, parsing payloads.
- `classifier.py` – Ollama client and `classify_email()` using structured output.
- `prompts.py` – System prompt and Pydantic schema (categories and `reason_short`).

## Pushing to GitHub

The repo is already a Git repository. To push to GitHub:

1. Create a new repository on GitHub (no need to add a README or .gitignore if you already have them locally).
2. Add the remote and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/EMAIL_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

Keep `credentials.json` and `token.json` out of version control (they are listed in `.gitignore`).
