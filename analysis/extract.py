"""
Schema-free role-based extractor with policy layer.

This version adds a principled inference-time policy:
- type sanity
- subword hygiene
- role sparsity

NO learning is modified.
"""

from pathlib import Path
from collections import defaultdict
import json
import re

import torch
import numpy as np

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead
from analysis.role_bindings import CLUSTER_CENTROIDS, CLUSTER_ROLE_MAP


# -----------------------------
# Paths & device
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "analysis" / "role_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Policy configuration
# -----------------------------

ROLE_CONF_THRESHOLDS = {
    "amount": 0.85,
    "item": 0.70,
    "person": 0.75,
    "time": 0.65,   # time is fuzzier by nature
}


NUMERIC_PATTERN = re.compile(r"^\d+$")
FUNCTION_WORDS = {
    "i", "me", "the", "and", "for", "on", "with", "was", "is", "this", "today", "yesterday"
}
TIME_TOKENS = {
    "today", "yesterday", "tonight", "tomorrow",
    "morning", "evening", "night",
    "last", "now"
}


# -----------------------------
# Helpers
# -----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest_role(embedding: np.ndarray):
    best_role, best_score = None, -1.0

    for cid, role in CLUSTER_ROLE_MAP.items():
        centroid = CLUSTER_CENTROIDS[cid]
        score = cosine_similarity(embedding, centroid)
        if score > best_score:
            best_score = score
            best_role = role

    return best_role, best_score


def passes_policy(token: str, role: str, confidence: float) -> bool:
    # Confidence gate
    if confidence < ROLE_CONF_THRESHOLDS.get(role, 1.0):
        return False

    # Subword hygiene
    if token.startswith("##"):
        return False

    # Type sanity
    if role == "amount":
        return bool(NUMERIC_PATTERN.match(token))
    if role == "time":
        # Temporal plausibility check
        if token in TIME_TOKENS:
            return True
        return False

    if role in {"item", "person"}:
        if token in FUNCTION_WORDS:
            return False
        return token.isalpha()

    return True


# -----------------------------
# Main extraction
# -----------------------------

def extract(text: str) -> dict:
    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    role_head = RoleProjectionHead(input_dim=encoder.hidden_size).to(DEVICE)
    role_head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    role_head.eval()

    candidates = defaultdict(list)

    with torch.no_grad():
        token_embs, _ = encoder([text])
        role_embs = role_head(token_embs)

        tokens = encoder.tokenizer.tokenize(text)

        for tok, emb in zip(tokens, role_embs[0]):
            role, confidence = nearest_role(emb.cpu().numpy())

            if role is None:
                continue

            if not passes_policy(tok, role, confidence):
                continue

            candidates[role].append(
                {
                    "token": tok,
                    "confidence": confidence,
                }
            )

    # Role sparsity: keep best per role
    final = {}
    meta = {}

    # ---- Core schema roles ----
    for role in {"amount", "item"}:
        items = candidates.get(role)
        if items:
            best = max(items, key=lambda x: x["confidence"])
            final[role] = best["token"]

    # ---- Optional metadata ----
    time_items = candidates.get("time")
    if time_items:
        best = max(time_items, key=lambda x: x["confidence"])
        meta["time"] = best["token"]

    if meta:
        final["meta"] = meta


    return final


# -----------------------------
# CLI demo
# -----------------------------

if __name__ == "__main__":
    examples = [
        "I paid 450 for lunch with Rahul yesterday",
        "450 lunch",
        "Rahul 1200 rent",
        "Coffee cost me 250 today",
        "yesterday 500 coffee",
        "I paid 450 for lunch yesterday",
        "yesterday 500 coffee",
        "Coffee 250 today",
        "last night 1200 rent",
    ]

    for text in examples:
        print("\nTEXT:", text)
        print(json.dumps(extract(text), indent=2))
