"""
Dump token-level role embeddings for analysis.

Responsibility:
- Load sentences
- Run encoder + trained role head
- Collect (token, role_embedding)
- Save to disk for downstream clustering / visualization

This file does NOT:
- train
- cluster
- visualize
"""

from pathlib import Path
from collections import defaultdict
import torch

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_SENTENCES_PATH = PROJECT_ROOT / "data" / "raw" / "sentences.txt"
CHECKPOINT_PATH = PROJECT_ROOT / "analysis" / "role_head.pt"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "role_embeddings.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
# -----------------------------

def load_sentences():
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -----------------------------
# Main logic
# -----------------------------

def dump_embeddings():
    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    print(f"📄 Loaded {len(sentences)} sentences")

    # Load encoder (frozen)
    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    # Load trained role head
    role_head = RoleProjectionHead(
        input_dim=encoder.hidden_size
    ).to(DEVICE)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"❌ Role head checkpoint not found at {CHECKPOINT_PATH}\n"
            f"Did you train the model before dumping embeddings?"
        )

    role_head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    role_head.eval()

    token_embeddings = defaultdict(list)

    with torch.no_grad():
        for sentence in sentences:
            token_embs, mask = encoder([sentence])
            role_embs = role_head(token_embs)

            tokens = encoder.tokenizer.tokenize(sentence)

            for tok, emb in zip(tokens, role_embs[0]):
                token_embeddings[tok].append(emb.cpu())

    # Convert lists to tensors
    final_dump = {
        tok: torch.stack(embs)
        for tok, embs in token_embeddings.items()
    }

    torch.save(final_dump, OUTPUT_PATH)

    print(f"✅ Dumped role embeddings for {len(final_dump)} unique tokens")
    print(f"📦 Saved to: {OUTPUT_PATH}")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    dump_embeddings()
