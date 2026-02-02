"""
Training loop for role-based contrastive learning.

Correct gradient-safe version.
"""

from pathlib import Path
from collections import defaultdict
import random

import torch
from torch import optim

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead
from training.loss import TripletContrastiveLoss


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_SENTENCES_PATH = ROOT / "data" / "raw" / "sentences.txt"
PAIRS_PATH = ROOT / "data" / "pairs" / "pairs.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
# -----------------------------

def load_sentences():
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_pairs():
    positives, negatives = [], []
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            kind, a, b, w = line.strip().split("\t")
            if kind == "pos":
                positives.append((a, b, float(w)))
            else:
                negatives.append((a, b, float(w)))
    return positives, negatives


def build_token_index(sentences, tokenizer):
    """
    token -> list of (sentence, token_position)
    """
    index = defaultdict(list)

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        for i, tok in enumerate(tokens):
            index[tok].append((sent, i))

    return index


# -----------------------------
# Training loop
# -----------------------------

def train(epochs=10, triplets_per_epoch=256, lr=1e-3):
    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    positives, negatives = load_pairs()

    print(f"📄 Loaded {len(sentences)} sentences")
    print(f"➕ {len(positives)} positive pairs")
    print(f"➖ {len(negatives)} negative pairs")

    encoder = TokenEncoder().to(DEVICE)
    role_head = RoleProjectionHead(input_dim=encoder.hidden_size).to(DEVICE)

    encoder.eval()          # frozen
    role_head.train()       # trainable

    loss_fn = TripletContrastiveLoss()
    optimizer = optim.Adam(role_head.parameters(), lr=lr)

    # Build token → (sentence, position) index ONCE
    token_index = build_token_index(sentences, encoder.tokenizer)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        batch_anchor, batch_pos, batch_neg, weights = [], [], [], []

        for _ in range(triplets_per_epoch):
            a, p, w = random.choice(positives)
            _, n, _ = random.choice(negatives)

            if a not in token_index or p not in token_index or n not in token_index:
                continue

            # Sample occurrences
            sent_a, pos_a = random.choice(token_index[a])
            sent_p, pos_p = random.choice(token_index[p])
            sent_n, pos_n = random.choice(token_index[n])

            # Forward pass (fresh graph)
            emb_a, _ = encoder([sent_a])
            emb_p, _ = encoder([sent_p])
            emb_n, _ = encoder([sent_n])

            role_a = role_head(emb_a)[0, pos_a]
            role_p = role_head(emb_p)[0, pos_p]
            role_n = role_head(emb_n)[0, pos_n]

            batch_anchor.append(role_a)
            batch_pos.append(role_p)
            batch_neg.append(role_n)
            weights.append(w)

        if not batch_anchor:
            print("⚠️ No valid triplets, skipping epoch")
            continue

        anchor_t = torch.stack(batch_anchor).to(DEVICE)
        pos_t = torch.stack(batch_pos).to(DEVICE)
        neg_t = torch.stack(batch_neg).to(DEVICE)
        weight_t = torch.tensor(weights, device=DEVICE)

        loss = loss_fn(anchor_t, pos_t, neg_t, weight=weight_t)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # Save trained role head
    ckpt = ROOT / "analysis" / "role_head.pt"
    torch.save(role_head.state_dict(), ckpt)
    print(f"💾 Saved role head to: {ckpt}")
    print("✅ Training complete")


if __name__ == "__main__":
    train()
