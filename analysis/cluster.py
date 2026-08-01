"""
Cluster role embeddings into latent role groups.

This script:
- Loads trained role_head
- Runs encoder + role_head on corpus (on-the-fly)
- Aggregates role embeddings per token
- Clusters tokens in role space (K-Means)
- Saves cluster centroids and token assignments

This is the bridge between:
role discovery → role-based extraction
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

from models.encoder import TokenEncoder
from models.role_head import RoleProjectionHead


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_SENTENCES_PATH = ROOT / "data" / "raw" / "sentences.txt"
CHECKPOINT_PATH = ROOT / "analysis" / "role_head.pt"
OUTPUT_PATH = ROOT / "analysis" / "role_clusters.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
# -----------------------------


def load_sentences():
    with RAW_SENTENCES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -----------------------------
# Clustering logic
# -----------------------------


def cluster_roles(
    num_clusters: int = 4,
    random_state: int = 42,
):
    """
    Cluster tokens by their learned role embeddings.

    Args:
        num_clusters:
            Number of latent role clusters.
    """

    print(f"🖥️ Using device: {DEVICE}")

    sentences = load_sentences()
    print(f"📄 Loaded {len(sentences)} sentences")

    # Load models
    encoder = TokenEncoder().to(DEVICE)
    encoder.eval()

    role_head = RoleProjectionHead(
        input_dim=encoder.hidden_size
    ).to(DEVICE)

    role_head.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    )
    role_head.eval()

    token_vectors = defaultdict(list)

    # -------------------------------------------------
    # Collect aligned role embeddings
    # -------------------------------------------------

    with torch.no_grad():
        for sentence in sentences:

            # Single source of truth
            tokens, token_embs = encoder.encode_with_tokens(sentence)

            # Projection head expects batch dimension
            role_embs = role_head(token_embs.unsqueeze(0))[0]

            for tok, emb in zip(tokens, role_embs):
                token_vectors[tok].append(
                    emb.cpu().numpy()
                )

    # -------------------------------------------------
    # Aggregate embeddings
    # -------------------------------------------------

    tokens = []
    vectors = []

    for tok, embs in token_vectors.items():
        tokens.append(tok)
        vectors.append(np.mean(embs, axis=0))

    X = np.vstack(vectors)

    print(
        f"🔢 Clustering {len(tokens)} tokens into "
        f"{num_clusters} clusters"
    )

    # -------------------------------------------------
    # K-Means
    # -------------------------------------------------

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=10,
    )

    cluster_ids = kmeans.fit_predict(X)

    token_to_cluster = {
        tok: int(cid)
        for tok, cid in zip(tokens, cluster_ids)
    }

    centroids = kmeans.cluster_centers_

    torch.save(
        {
            "token_to_cluster": token_to_cluster,
            "centroids": centroids,
        },
        OUTPUT_PATH,
    )

    print(f"✅ Saved role clusters to: {OUTPUT_PATH}")

    # -------------------------------------------------
    # Display discovered clusters
    # -------------------------------------------------

    clusters = defaultdict(list)

    for tok, cid in token_to_cluster.items():
        clusters[cid].append(tok)

    print("\n📦 Discovered role clusters:")

    for cid in sorted(clusters):
        print(f"  Cluster {cid}: {sorted(clusters[cid])}")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    cluster_roles()