"""
Schema-free role-based extractor.

Purpose:
- Take raw text input
- Assign tokens to discovered roles
- Produce structured output (JSON-like dict)

This file:
- Does NOT train
- Does NOT cluster
- Does NOT assume schema during learning

It only uses:
- trained role_head
- discovered role clusters
- post-hoc role bindings
"""

from pathlib import Path
from collections import defaultdict
import json

import torch
import numpy as np

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead
from analysis.role_bindings import (
    TOKEN_TO_CLUSTER,
    CLUSTER_CENTROIDS,
    CLUSTER_ROLE_MAP,
    get_role_for_token,
)

# -----------------------------
# Paths & device
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "analysis" / "role_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Helper functions
# -----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest_role(embedding: np.ndarray):
    """
    Assign role by nearest cluster centroid.

    Returns:
        (role_name, confidence) or (None, None)
    """
    best_role = None
    best_score = -1.0

    for cid, role in CLUSTER_ROLE_MAP.items():
        centroid = CLUSTER_CENTROIDS[cid]
        score = cosine_similarity(embedding, centroid)

        if score > best_score:
            best_score = score
            best_role = role

    return best_role, best_score


# -----------------------------
# Main extraction logic
# -----------------------------

def extract(text: str) -> dict:
    """
    Extract structured information from raw text.

    Args:
        text: input sentence

    Returns:
        dict with role -> extracted values
    """

    # Load models
    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    role_head = RoleProjectionHead(input_dim=encoder.hidden_size).to(DEVICE)
    role_head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    role_head.eval()

    results = defaultdict(list)

    with torch.no_grad():
        token_embs, _ = encoder([text])
        role_embs = role_head(token_embs)

        tokens = encoder.tokenizer.tokenize(text)

        for tok, emb in zip(tokens, role_embs[0]):
            emb_np = emb.cpu().numpy()

            role, confidence = nearest_role(emb_np)

            if role is None:
                continue

            # Optional confidence threshold (conservative)
            if confidence < 0.4:
                continue

            results[role].append(
                {
                    "token": tok,
                    "confidence": round(confidence, 3),
                }
            )

    return dict(results)


# -----------------------------
# CLI usage
# -----------------------------

if __name__ == "__main__":
    examples = [
        "I paid 450 for lunch with Rahul yesterday",
        "Spent 1200 on rent",
        "Coffee cost me 250 today",
    ]

    for text in examples:
        print("\nTEXT:", text)
        output = extract(text)
        print(json.dumps(output, indent=2))
