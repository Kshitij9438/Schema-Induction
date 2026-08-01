"""
Training loop for role-based contrastive learning.

This implementation trains the RoleProjectionHead using weak
contrastive supervision while keeping the encoder frozen.

Design goals:
- Single source of truth for tokenization (TokenEncoder)
- Gradient-safe
- Token/embedding alignment guaranteed
- Minimal research assumptions
"""

from pathlib import Path
from collections import defaultdict
import random

import torch
from torch import optim

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead
from training.loss import TripletContrastiveLoss


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_SENTENCES_PATH = ROOT / "data" / "raw" / "sentences.txt"
PAIRS_PATH = ROOT / "data" / "pairs" / "pairs.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def load_sentences():
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_pairs():
    positives = []
    negatives = []

    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            kind, a, b, w = line.strip().split("\t")

            if kind == "pos":
                positives.append((a, b, float(w)))
            else:
                negatives.append((a, b, float(w)))

    return positives, negatives


def build_token_index(sentences, encoder):
    """
    Build an index using the encoder's tokenizer.

    Returns
    -------
    token -> list[(sentence, tokenizer_position)]
    """

    index = defaultdict(list)

    encoder.eval()

    with torch.no_grad():
        for sentence in sentences:

            tokens, _ = encoder.encode_with_tokens(sentence)

            for position, token in enumerate(tokens):
                index[token].append((sentence, position))

    return index


# --------------------------------------------------
# Training
# --------------------------------------------------


def train(
    epochs=10,
    triplets_per_epoch=256,
    lr=1e-3,
):
    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    positives, negatives = load_pairs()

    print(f"📄 Loaded {len(sentences)} sentences")
    print(f"➕ {len(positives)} positive pairs")
    print(f"➖ {len(negatives)} negative pairs")

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()  # Frozen

    role_head = RoleProjectionHead(
        input_dim=encoder.hidden_size
    ).to(DEVICE)

    role_head.train()

    loss_fn = TripletContrastiveLoss()

    optimizer = optim.Adam(
        role_head.parameters(),
        lr=lr,
    )

    # --------------------------------------------------
    # Build token occurrence index
    # --------------------------------------------------

    token_index = build_token_index(
        sentences,
        encoder,
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        batch_anchor = []
        batch_positive = []
        batch_negative = []
        batch_weights = []

        for _ in range(triplets_per_epoch):

            anchor_token, positive_token, weight = random.choice(
                positives
            )

            _, negative_token, _ = random.choice(
                negatives
            )

            if (
                anchor_token not in token_index
                or positive_token not in token_index
                or negative_token not in token_index
            ):
                continue

            # ------------------------------------------
            # Sample token occurrences
            # ------------------------------------------

            sent_a, pos_a = random.choice(
                token_index[anchor_token]
            )

            sent_p, pos_p = random.choice(
                token_index[positive_token]
            )

            sent_n, pos_n = random.choice(
                token_index[negative_token]
            )

            # ------------------------------------------
            # Encode using the canonical tokenizer
            # ------------------------------------------

            _, emb_a = encoder.encode_with_tokens(sent_a)
            _, emb_p = encoder.encode_with_tokens(sent_p)
            _, emb_n = encoder.encode_with_tokens(sent_n)

            # Safety checks

            assert pos_a < emb_a.size(0)
            assert pos_p < emb_p.size(0)
            assert pos_n < emb_n.size(0)

            # Projection head expects batch dimension

            role_a = role_head(
                emb_a.unsqueeze(0)
            )[0, pos_a]

            role_p = role_head(
                emb_p.unsqueeze(0)
            )[0, pos_p]

            role_n = role_head(
                emb_n.unsqueeze(0)
            )[0, pos_n]

            batch_anchor.append(role_a)
            batch_positive.append(role_p)
            batch_negative.append(role_n)
            batch_weights.append(weight)

        if not batch_anchor:
            print("⚠️ No valid triplets found.")
            continue

        anchor_tensor = torch.stack(batch_anchor)
        positive_tensor = torch.stack(batch_positive)
        negative_tensor = torch.stack(batch_negative)

        weight_tensor = torch.tensor(
            batch_weights,
            device=DEVICE,
        )

        loss = loss_fn(
            anchor_tensor,
            positive_tensor,
            negative_tensor,
            weight=weight_tensor,
        )

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    checkpoint_path = ROOT / "analysis" / "role_head.pt"

    torch.save(
        role_head.state_dict(),
        checkpoint_path,
    )

    print(f"💾 Saved role head to: {checkpoint_path}")
    print("✅ Training complete")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    train()