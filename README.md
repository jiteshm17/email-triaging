# Email Triaging

A personal tool to keep Gmail organised. Fetches emails, classifies each into one of 16 categories using a local LLM (Ollama), and applies the matching Gmail label while marking the message as read — no third-party cloud services involved.

---

## What it does

1. **Fetch** – Connects to Gmail via OAuth and downloads email headers + body text.
2. **Classify** – Sends subject + body to a locally-running Ollama model; gets back a category and a short reason.
3. **Save** – Writes raw and tagged CSVs to `data/` for inspection.
4. **Apply** – Adds the predicted category as a Gmail label and marks the message as read (same effect as an n8n "Add label → Mark as read" flow).

Raw emails are **cached** in `data/` so re-running only redoes classification — useful when tweaking the prompt without re-downloading.

---

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

```bash
python run.py --max 200          # process at most 200 unread emails
python run.py --dry-run          # test without touching Gmail
python run.py --model llama3:8b  # use a different model
```

To force a fresh download (e.g. to pick up new emails after a dry run):
```bash
del data\gmail_unread.csv        # Windows
rm data/gmail_unread.csv         # macOS/Linux
python run.py
```

---

### Backfill — tag recent emails (read + unread)

```bash
python tag_recent.py
```

Fetches recent inbox emails regardless of read status. Useful for a one-off backfill.

| Flag | Default | Description |
|---|---|---|
| `--max N` | `1000` | Number of recent emails to fetch |
| `--dry-run` | off | Classify only, skip Gmail updates |
| `--model MODEL` | `qwen2.5:14b` | Ollama model to use |

```bash
python tag_recent.py --max 500
python tag_recent.py --max 2000 --dry-run
```

---

### Build a labelled dataset (for prompt optimisation)

```bash
python build_dataset.py
```

Scans recent emails, keeps only those that already have a custom Gmail label applied, and saves them as a CSV. This is the training data used for prompt optimisation.

| Flag | Default | Description |
|---|---|---|
| `--max N` | `5000` | Emails to scan |
| `--query Q` | `in:inbox` | Gmail query (e.g. `in:anywhere`) |
| `--out PATH` | `data/gmail_tagged_emails.csv` | Output path |

---

### Sample a subset for labelling / review

```bash
python sample_dataset.py data/gmail_tagged_emails.csv
```

Samples N rows per category for manual review or prompt evaluation.

| Flag | Default | Description |
|---|---|---|
| `--n N` | `20` | Rows per category |
| `--seed SEED` | `42` | Random seed (use `0` for a completely different sample) |
| `--data-dir DIR` | `data` | Output directory |

Output is saved as `<input_name>_sampled.csv` in the data directory.

---

### Optimise the prompt with DSPy

```bash
python optimize.py
```

Uses [DSPy MIPROv2](https://dspy.ai) to search for a better prompt instruction by evaluating candidate rewrites against human-corrected labels. Requires two CSVs: a features file and a ground-truth labels file (see `data/`).

| Flag | Default | Description |
|---|---|---|
| `--sample CSV` | `data/gmail_tagged_sample.csv` | Features CSV |
| `--ground-truth CSV` | `data/gmail_tagged_sample_updated.csv` | Human-corrected labels CSV |
| `--out-dir DIR` | `data/optimized` | Where to save the result |
| `--model MODEL` | `qwen2.5:14b` | Ollama model |
| `--effort LEVEL` | `light` | `light` (~70–90 min) / `medium` (~2–3 h) / `heavy` |

> **Note:** Ollama processes one request at a time on a single GPU. More threads don't help — the default of 1 is optimal.

The optimised program is saved to `data/optimized/optimized_classifier.json`. The winning instruction is also printed at the end so you can copy it back into `utils/prompts.py`.

---

## Project layout

```
run.py               # Daily job: tag all unread emails
tag_recent.py        # One-off backfill: tag N recent emails (read + unread)
build_dataset.py     # Fetch already-labelled emails to build training data
sample_dataset.py    # Sample N rows per category for review / optimisation
optimize.py          # DSPy MIPROv2 prompt optimisation

utils/
  prompts.py         # System prompt, category list, Pydantic schema
  gmail.py           # Gmail API auth, listing, fetching, parsing
  classifier.py      # Ollama client + classify_email()

notebooks/
  eda.ipynb          # Label distribution analysis and dataset preparation

data/                # All data files (gitignored — may contain personal email content)
  gmail_unread.csv          # Cached raw unread emails (run.py)
  gmail_unread_tagged.csv   # Classified unread emails
  gmail_tagged_emails.csv   # Labelled dataset (build_dataset.py)
  optimized/                # Saved optimised DSPy programs
```

---

## Contributing

Pull requests, bug reports, and feature suggestions are welcome. If you extend the category list or improve the prompt, feel free to open a PR — this is a small personal project but happy to make it useful for others too.
