"""
Pair generation for contrastive role learning.

This module implements the Signal Specification exactly as defined in docs/SIGNAL_SPEC.md.

Output:
- A list of training triples:
    (anchor_token, positive_token, negative_token, weight)

Pairs are NOT labels.
Pairs are weak geometric constraints.
"""

from pathlib import Path
from collections import defaultdict
import random
import re

# -----------------------------
# Paths
# -----------------------------

RAW_SENTENCES_PATH = Path(__file__).resolve().parents[1] / "raw" / "sentences.txt"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "pairs" / "pairs.txt"

random.seed(42)

# -----------------------------
# Simple utilities
# -----------------------------

DIGIT_PATTERN = re.compile(r"^\d+$")

FUNCTION_WORDS = {
    "i", "for", "with", "on", "and", "was", "is", "me",
    "the", "a", "an", "to", "of", "in"
}


def is_numeric(token: str) -> bool:
    return bool(DIGIT_PATTERN.match(token))


def tokenize(sentence: str) -> list[str]:
    # Very simple tokenizer by design (do NOT over-engineer)
    return sentence.strip().replace(",", "").split()


def position_bucket(index: int, length: int) -> str:
    """Coarse position bucket."""
    ratio = index / max(length - 1, 1)
    if ratio < 0.33:
        return "start"
    elif ratio < 0.66:
        return "middle"
    return "end"


# -----------------------------
# Load & preprocess data
# -----------------------------

def load_sentences() -> list[list[str]]:
    sentences = []
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line.lower())
            if tokens:
                sentences.append(tokens)
    return sentences


# -----------------------------
# Index construction
# -----------------------------

def build_token_index(sentences):
    """
    Build token occurrence index:
    token -> list of (sentence_id, position, context)
    """
    index = defaultdict(list)

    for sid, tokens in enumerate(sentences):
        for i, tok in enumerate(tokens):
            context = (
                tokens[i - 1] if i - 1 >= 0 else None,
                tokens[i + 1] if i + 1 < len(tokens) else None,
            )
            index[tok].append((sid, i, context))

    return index


# -----------------------------
# Positive pair generation
# -----------------------------

def positive_identity_pairs(index):
    """
    Category A — Identity Anchors
    Applied only to schema-carrying tokens.
    """
    pairs = []
    for tok, occurrences in index.items():
        if tok in FUNCTION_WORDS:
            continue

        # Exclude verbs and temporal glue
        if tok in {"paid", "spent", "cost", "this", "last"}:
            continue

        # Anchor only if token appears multiple times
        if len(occurrences) > 1:
            pairs.append((tok, tok, 1.0))

    return pairs

def positive_numeric_context_pairs(index):
    """
    Category B — Context-similar numeric tokens
    """
    pairs = []
    numeric_tokens = [t for t in index if is_numeric(t)]

    for i, t1 in enumerate(numeric_tokens):
        for t2 in numeric_tokens[i + 1 :]:
            for (_, pos1, ctx1) in index[t1]:
                for (_, pos2, ctx2) in index[t2]:
                    if ctx1 == ctx2:
                        pairs.append((t1, t2, 0.9))
                        break
    return pairs


def positive_frame_entity_pairs(index):
    """
    Category C — Frame-equivalent entities (e.g. 'with Rahul', 'with Amit')
    """
    pairs = []
    candidates = [
        tok for tok in index
        if tok not in FUNCTION_WORDS and not is_numeric(tok)
    ]

    for i, t1 in enumerate(candidates):
        for t2 in candidates[i + 1 :]:
            for (_, pos1, ctx1) in index[t1]:
                for (_, pos2, ctx2) in index[t2]:
                    if ctx1 == ctx2 and ctx1[0] in {"with"}:
                        pairs.append((t1, t2, 0.7))
                        break
    return pairs


# -----------------------------
# Negative pair generation
# -----------------------------

def negative_cross_type_pairs(index):
    """
    Category A — Cross-type separation
    """
    pairs = []
    tokens = list(index.keys())

    for t1 in tokens:
        for t2 in tokens:
            if t1 == t2:
                continue
            if is_numeric(t1) and not is_numeric(t2):
                pairs.append((t1, t2, 1.0))
    return pairs


def negative_function_pairs(index):
    """
    Category B — Function vs content
    """
    pairs = []
    for f in FUNCTION_WORDS:
        if f not in index:
            continue
        for tok in index:
            if tok != f and not tok in FUNCTION_WORDS:
                pairs.append((f, tok, 0.8))
    return pairs


# -----------------------------
# Pair assembly
# -----------------------------

def build_pairs():
    sentences = load_sentences()
    index = build_token_index(sentences)

    positives = []
    positives += positive_identity_pairs(index)
    positives += positive_numeric_context_pairs(index)
    positives += positive_frame_entity_pairs(index)

    negatives = []
    negatives += negative_cross_type_pairs(index)
    negatives += negative_function_pairs(index)

    # De-duplicate
    positive_set = set((a, b) for a, b, _ in positives)
    negative_set = set((a, b) for a, b, _ in negatives)

    # Safety rule: no overlap
    final_pairs = []

    for a, b, w in positives:
        if (a, b) not in negative_set:
            final_pairs.append(("pos", a, b, w))

    for a, b, w in negatives:
        if (a, b) not in positive_set:
            final_pairs.append(("neg", a, b, w))

    return final_pairs


# -----------------------------
# Save output
# -----------------------------

def main():
    pairs = build_pairs()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for kind, a, b, w in pairs:
            f.write(f"{kind}\t{a}\t{b}\t{w}\n")

    print(f"✅ Generated {len(pairs)} training pairs")
    print(f"📄 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
