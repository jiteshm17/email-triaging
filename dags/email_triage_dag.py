"""
Airflow DAG: classify unread Primary-tab Gmail emails daily at 8 PM.

Runs `python run.py` from the project directory inside WSL2.
The project lives on the Windows filesystem, mounted at /mnt/c/ inside WSL2.

To use a different schedule, change the `schedule` parameter (cron format):
    "0 20 * * *"  →  every day at 20:00 (8 PM)
    "0 8 * * *"   →  every day at 08:00 (8 AM)
    "0 20 * * 1-5"→  weekdays only at 8 PM
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Edit these two paths to match your setup inside WSL2.
# Your Windows path  C:\Users\Jitesh\Desktop\Email Triaging
# becomes the WSL2 path  /mnt/c/Users/Jitesh/Desktop/Email Triaging
# ---------------------------------------------------------------------------
PROJECT_DIR = "/mnt/c/Users/Jitesh/Desktop/Email Triaging"

# Path to the Python interpreter that has all project dependencies installed.
# If you used a venv inside WSL2 (recommended), point to its python binary:
#   PYTHON = f"{PROJECT_DIR}/.venv/bin/python"
# Otherwise fall back to the system python3:
PYTHON = "python3"
# ---------------------------------------------------------------------------

default_args = {
    "owner": "jitesh",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="email_triage",
    description="Classify unread Primary-inbox emails with Ollama and apply Gmail labels.",
    schedule="0 20 * * *",          # 8 PM every day (local time of the Airflow server)
    start_date=datetime(2026, 3, 20),
    catchup=False,                  # don't backfill missed runs
    default_args=default_args,
    tags=["email", "ollama"],
) as dag:

    classify_and_tag = BashOperator(
        task_id="classify_and_tag",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'{PYTHON} run.py'
            # Add flags here if needed, e.g.:
            #   ' --max 300'
            #   ' --dry-run'
            #   ' --all-tabs'
        ),
        # Stream all output (classification table, progress bars) to Airflow logs.
        append_env=True,
    )
