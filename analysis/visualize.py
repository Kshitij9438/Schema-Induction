"""
Visualization of learned role embeddings.

This script:
- Loads the trained RoleProjectionHead
- Runs the frozen encoder + role head on the corpus
- Collects token-level role embeddings
- Aggregates embeddings per token (mean)
- Reduces dimensionality using PCA
- Saves a 2D visualization

Purpose:
- Human inspection of the learned role space
- NOT a quantitative evaluation
"""

from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_SENTENCES_PATH = ROOT / "data" / "raw" / "sentences.txt"
CHECKPOINT_PATH = ROOT / "analysis" / "role_head.pt"
OUTPUT_PATH = ROOT / "analysis" / "role_space.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def load_sentences():
    """Load raw training sentences."""

    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------
# Visualization
# --------------------------------------------------


def visualize():
    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    print(f"📄 Loaded {len(sentences)} sentences")

    # ----------------------------------------------
    # Models
    # ----------------------------------------------

    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    role_head = RoleProjectionHead(
        input_dim=encoder.hidden_size
    ).to(DEVICE)

    role_head.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )
    )

    role_head.eval()

    # ----------------------------------------------
    # Collect role embeddings
    # ----------------------------------------------

    token_vectors = defaultdict(list)

    with torch.no_grad():
        for sentence in sentences:

            # Canonical tokenizer + aligned embeddings
            tokens, token_embeddings = encoder.encode_with_tokens(
                sentence
            )

            # Projection head expects batch dimension
            role_embeddings = role_head(
                token_embeddings.unsqueeze(0)
            )[0]

            # Guaranteed alignment:
            # tokens[i] <-> role_embeddings[i]
            for token, embedding in zip(
                tokens,
                role_embeddings,
            ):
                token_vectors[token].append(
                    embedding.cpu().numpy()
                )

    # ----------------------------------------------
    # Aggregate occurrences
    # ----------------------------------------------

    tokens = []
    vectors = []

    for token, embeddings in token_vectors.items():
        tokens.append(token)
        vectors.append(
            np.mean(embeddings, axis=0)
        )

    X = np.vstack(vectors)

    # ----------------------------------------------
    # PCA
    # ----------------------------------------------

    pca = PCA(n_components=2)

    X_2d = pca.fit_transform(X)

    print(
        f"📈 Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    plt.figure(figsize=(12, 10))

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        alpha=0.7,
    )

    for i, token in enumerate(tokens):
        plt.text(
            X_2d[i, 0],
            X_2d[i, 1],
            token,
            fontsize=9,
            alpha=0.8,
        )

    plt.title("Role-Based Token Embedding Space (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"📊 Plot saved to: {OUTPUT_PATH}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    visualize()