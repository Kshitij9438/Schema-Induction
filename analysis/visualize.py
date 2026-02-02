"""
Visualization of learned role embeddings.

This script:
- Loads trained role_head
- Runs encoder + role_head on corpus
- Collects token-level role embeddings
- Reduces dimensionality (PCA)
- Plots tokens in 2D space

This is for HUMAN INSPECTION, not metrics.
"""

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_SENTENCES_PATH = ROOT / "data" / "raw" / "sentences.txt"
CHECKPOINT_PATH = ROOT / "analysis" / "role_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
# -----------------------------

def load_sentences():
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -----------------------------
# Main visualization logic
# -----------------------------

def visualize():
    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    print(f"📄 Loaded {len(sentences)} sentences")

    # Load models
    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    role_head = RoleProjectionHead(input_dim=encoder.hidden_size).to(DEVICE)
    role_head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    role_head.eval()

    token_vectors = defaultdict(list)

    with torch.no_grad():
        for sentence in sentences:
            token_embs, _ = encoder([sentence])
            role_embs = role_head(token_embs)

            tokens = encoder.tokenizer.tokenize(sentence)

            for tok, emb in zip(tokens, role_embs[0]):
                token_vectors[tok].append(emb.cpu().numpy())

    # Aggregate embeddings per token (mean)
    tokens = []
    vectors = []

    for tok, embs in token_vectors.items():
        tokens.append(tok)
        vectors.append(np.mean(embs, axis=0))

    X = np.vstack(vectors)

    # Reduce dimensionality
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

    for i, tok in enumerate(tokens):
        plt.text(
            X_2d[i, 0],
            X_2d[i, 1],
            tok,
            fontsize=9,
            alpha=0.8,
        )

    plt.title("Role-Based Token Embedding Space (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    output_path = ROOT / "analysis" / "role_space.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"📊 Plot saved to: {output_path}")



# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    visualize()
