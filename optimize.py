"""
Optimize the email classification prompt using DSPy MIPROv2.

Ground truth labels: gmail_tagged_sample_updated.csv  (human-corrected)
Email features:      gmail_tagged_sample.csv           (id, subject, body_text, …)
Both are joined on 'id'. The old LLM-assigned label is discarded.

MIPROv2 proposes and evaluates new prompt instructions + few-shot examples and
picks the combination that maximises exact-match accuracy on the held-out val set.

--- max_tokens ---
This is the OUTPUT token budget per LM call (not input).
The updated prompt asks for analysis + category + reason_short, which is roughly
80-180 tokens. 300 gives headroom without wasting context.

--- num_threads and single GPU ---
Ollama runs one request at a time on a single GPU. Sending multiple concurrent
requests just queues them — it doesn't speed anything up and adds overhead.
Use NUM_THREADS = 1.

--- Runtime estimate (qwen2.5:14b on 8 GB VRAM, ~3-5 s/call) ---
  "light"  : ~5  optimizer trials ≈  30–60 min   ← recommended starting point
  "medium" : ~15 optimizer trials ≈  90–180 min
  "heavy"  : exhaustive           ≈  several hours
Start with "light". If accuracy is already good, stop there.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import get_args

import dspy
import pandas as pd
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

from utils.prompts import Category, SYSTEM_PROMPT

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_BODY_CHARS = 1500   # truncate long bodies to stay within context
TRAIN_RATIO = 0.75
SEED = 42

# ── MIPROv2 iteration budget ──────────────────────────────────────────────────
# auto="light" lets DSPy pick sensible defaults (~6 instruction candidates,
# ~13 Bayesian trials). Each trial scores the full val set, so at ~5 s/call
# on a single GPU expect roughly 70–90 min total.
# compile() returns ONLY the single best program found across all trials.
#
# If you want to tune manually later, replace auto="light" with explicit args:
#   NUM_CANDIDATE_INSTRUCTIONS = 6   # instruction variants to propose
#   MAX_FEW_SHOT_DEMOS         = 2   # few-shot examples added to the prompt
#   NUM_TRIALS                 = 13  # search trials (25 = "medium", 50 = "heavy")
# and pass them directly to MIPROv2(...) / compile(...) below.

# Single GPU → Ollama serialises all requests; more threads just queue and add overhead
NUM_THREADS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise the email classification prompt with DSPy MIPROv2.")
    parser.add_argument(
        "--sample", default=f"{DATA_DIR}/gmail_tagged_sample.csv", metavar="CSV",
        help="Labelled sample CSV (features)",
    )
    parser.add_argument(
        "--ground-truth", default=f"{DATA_DIR}/gmail_tagged_sample_updated.csv", metavar="CSV",
        help="Human-corrected labels CSV (id, main_label, reason)",
    )
    parser.add_argument(
        "--out-dir", default=f"{DATA_DIR}/optimized", metavar="DIR",
        help=f"Directory to save the optimised program (default: {DATA_DIR}/optimized)",
    )
    parser.add_argument(
        "--model", default="qwen2.5:14b", metavar="MODEL",
        help="Ollama model (default: qwen2.5:14b)",
    )
    parser.add_argument(
        "--effort", default="light", choices=["light", "medium", "heavy"],
        help="MIPROv2 search effort — more effort = longer runtime (default: light)",
    )
    return parser.parse_args()

# Derive categories directly from prompts.py so this file never drifts out of sync
CATEGORIES: list[str] = list(get_args(Category))
CATEGORIES_DESC = ", ".join(CATEGORIES)

# ── DSPy Signature ────────────────────────────────────────────────────────────
# The docstring becomes the LM instruction. We set it to SYSTEM_PROMPT after
# class creation so it stays in sync with prompts.py.

class EmailClassify(dspy.Signature):
    subject: str = dspy.InputField(desc="Email subject line")
    body_text: str = dspy.InputField(desc="Email body text (may be truncated)")
    reason_short: str = dspy.OutputField(desc="≤12 words explaining why this category fits best")
    category: str = dspy.OutputField(desc=f"Exactly one of: {CATEGORIES_DESC}")

EmailClassify.__doc__ = SYSTEM_PROMPT.strip()


# ── Module ────────────────────────────────────────────────────────────────────

class EmailClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(EmailClassify)

    def forward(self, subject: str, body_text: str) -> dspy.Prediction:
        return self.classify(
            subject=subject,
            body_text=(body_text or "")[:MAX_BODY_CHARS],
        )


# ── Metric ────────────────────────────────────────────────────────────────────

def category_accuracy(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Exact match on the category field (case-insensitive, stripped)."""
    return (
        str(example.category).strip().upper()
        == str(getattr(pred, "category", "")).strip().upper()
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def _print_split_distribution(trainset: list, valset: list) -> None:
    from collections import Counter
    train_counts = Counter(e.category for e in trainset)
    val_counts   = Counter(e.category for e in valset)
    all_cats = sorted(set(train_counts) | set(val_counts))
    print(f"\n{'Category':<35} {'Train':>6} {'Val':>6}")
    print("-" * 50)
    missing_from_train = []
    for cat in all_cats:
        t, v = train_counts.get(cat, 0), val_counts.get(cat, 0)
        flag = "  ← MISSING FROM TRAIN" if t == 0 else ""
        print(f"{cat:<35} {t:>6} {v:>6}{flag}")
        if t == 0:
            missing_from_train.append(cat)
    print("-" * 50)
    print(f"{'TOTAL':<35} {sum(train_counts.values()):>6} {sum(val_counts.values()):>6}\n")
    if missing_from_train:
        print(f"WARNING: {len(missing_from_train)} class(es) have no training examples: {missing_from_train}")
        print("Consider re-running with a different SEED or increasing the dataset size.\n")


def load_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    updated = pd.read_csv(args.ground_truth)
    updated.columns = updated.columns.str.strip()

    sample = sample.drop(columns=["main_label"], errors="ignore")
    merged = sample.merge(
        updated[["id", "main_label", "reason"]].rename(
            columns={"main_label": "category", "reason": "gt_reason"}
        ),
        on="id",
        how="inner",
    )
    print(f"Merged dataset: {len(merged)} rows, {merged['category'].nunique()} unique labels")

    examples = [
        dspy.Example(
            subject=str(row.get("subject", "") or ""),
            body_text=str(row.get("body_text", "") or ""),
            category=str(row["category"]).strip(),
        ).with_inputs("subject", "body_text")
        for _, row in merged.iterrows()
    ]

    random.seed(SEED)
    random.shuffle(examples)
    n_train = int(len(examples) * TRAIN_RATIO)
    trainset, valset = examples[:n_train], examples[n_train:]
    _print_split_distribution(trainset, valset)
    return trainset, valset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    lm = dspy.LM(
        f"ollama/{args.model}",
        api_base=OLLAMA_BASE_URL,
        max_tokens=300,     # output tokens only; covers category + reason_short
    )
    dspy.configure(lm=lm)

    trainset, valset = load_dataset()
    print(f"\nTrain: {len(trainset)} | Val: {len(valset)}")

    evaluator = Evaluate(
        devset=valset,
        metric=category_accuracy,
        num_threads=NUM_THREADS,
        display_progress=True,
    )

    # Baseline (unoptimised)
    # evaluator() returns an EvaluationResult object in newer DSPy; convert to float
    baseline_score = float(evaluator(EmailClassifier()))
    print(f"\nBaseline accuracy: {baseline_score:.2f}%")

    # ── MIPROv2 ───────────────────────────────────────────────────────────────
    # Phase 1: proposes NUM_CANDIDATE_INSTRUCTIONS rewordings of the prompt.
    # Phase 2: NUM_TRIALS Bayesian search trials, each scoring one (instruction,
    #          few-shot examples) combination on the full val set.
    # Returns the single best-performing program found across all trials.
    #
    # To switch to GRPO (RL-based reward optimisation) instead:
    #   from dspy.teleprompt import GRPO
    #   optimizer = GRPO(metric=category_accuracy, num_threads=NUM_THREADS)
    #   optimized = optimizer.compile(EmailClassifier(), trainset=trainset)
    optimizer = MIPROv2(
        metric=category_accuracy,
        auto=args.effort,
        num_threads=NUM_THREADS,
    )
    optimized = optimizer.compile(
        EmailClassifier(),
        trainset=trainset,
        valset=valset,
        # Skips the "this will make many LM calls, continue?" confirmation prompt
        requires_permission_to_run=False,
    )

    optimized_score = float(evaluator(optimized))
    print(f"Optimised accuracy: {optimized_score:.2f}%  (baseline: {baseline_score:.2f}%)")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = f"{args.out_dir}/optimized_classifier.json"
    optimized.save(save_path)
    print(f"\nSaved optimised program to {save_path}")

    # Print the winning instruction so you can paste it back into prompts.py
    try:
        instr = optimized.classify.extended_signature.instructions
        print("\n── Optimised instruction ──────────────────────────────────────────")
        print(instr)
        print("──────────────────────────────────────────────────────────────────")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
