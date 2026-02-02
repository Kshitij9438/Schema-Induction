"""
Synthetic sentence generator for Neural Schema Induction.

Purpose:
- Generate controlled, role-stable but surface-diverse sentences
- Single domain: expenses
- No labels, no annotations, no metadata
- Output: plain text sentences (one per line)

This data is meant to teach ROLE CONSISTENCY, not realism.
"""

import random
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

NUM_SENTENCES = 1000
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "raw" / "sentences.txt"

random.seed(42)

# -----------------------------
# Vocabulary pools
# -----------------------------

AMOUNTS = [
    "100", "250", "450", "500", "750",
    "1200", "1500", "2000", "3200", "5000"
]

ITEMS = [
    "lunch", "dinner", "breakfast",
    "rent", "groceries", "shoes",
    "books", "coffee", "snacks"
]

PEOPLE = [
    "Rahul", "Amit", "Riya", "Neha",
    "Ankit", "Priya", "Suman"
]

TIMES = [
    "today", "yesterday", "last night",
    "this morning", "in the evening"
]

# -----------------------------
# Sentence templates
# -----------------------------

TEMPLATES = [

    # -----------------------------
    # Canonical templates (baseline)
    # -----------------------------

    lambda a, i, p, t: f"I paid {a} for {i}",
    lambda a, i, p, t: f"{i.capitalize()} cost me {a}",
    lambda a, i, p, t: f"{a} was spent on {i}",

    lambda a, i, p, t: f"I paid {a} for {i} with {p}",
    lambda a, i, p, t: f"{p} and I spent {a} on {i}",

    lambda a, i, p, t: f"I paid {a} for {i} {t}",
    lambda a, i, p, t: f"{t.capitalize()}, I spent {a} on {i}",

    lambda a, i, p, t: f"I paid {a} for {i} with {p} {t}",
    lambda a, i, p, t: f"{p} and I spent {a} on {i} {t}",

    lambda a, i, p, t: f"I spent {a}",

    # -----------------------------
    # Surface-form stress templates
    # -----------------------------

    # Inverted / reordered
    lambda a, i, p, t: f"{a} for {i} I paid",
    lambda a, i, p, t: f"{i} was {a}",
    lambda a, i, p, t: f"{a} lunch payment",

    # Telegraphed / minimal
    lambda a, i, p, t: f"{a} {i}",
    lambda a, i, p, t: f"{i} {a}",

    # Verb dropped / implied
    lambda a, i, p, t: f"{i} : {a}",
    lambda a, i, p, t: f"{a} spent {i}",

    # Person-leading
    lambda a, i, p, t: f"{p} {a} {i}",
    lambda a, i, p, t: f"{p} paid {a}",

    # Time-leading
    lambda a, i, p, t: f"{t} {a} {i}",
    lambda a, i, p, t: f"{t} {i} {a}",
]

# -----------------------------
# Generator logic
# -----------------------------

def generate_sentence() -> str:
    amount = random.choice(AMOUNTS)
    item = random.choice(ITEMS)
    person = random.choice(PEOPLE)
    time = random.choice(TIMES)

    template = random.choice(TEMPLATES)
    return template(amount, item, person, time)


def main():
    sentences = []

    for _ in range(NUM_SENTENCES):
        sentence = generate_sentence()
        sentences.append(sentence)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"✅ Generated {len(sentences)} sentences")
    print(f"📄 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
