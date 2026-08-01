"""
Quantitative evaluation of learned role embeddings.

This script evaluates the quality of the discovered role space.

Metrics
-------
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Cluster sizes
- Pairwise centroid distances

This is the primary quantitative evaluation used throughout
the project.
"""

from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import torch

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = ROOT / "analysis" / "role_embeddings.pt"
CLUSTERS_PATH = ROOT / "analysis" / "role_clusters.pt"


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def load_embeddings():
    """
    Returns
    -------
    dict[token] -> Tensor(num_occurrences, embedding_dim)
    """
    return torch.load(EMBEDDINGS_PATH, weights_only=False, )


def load_clusters():
    """
    Returns
    -------
    token_to_cluster : dict
    centroids : ndarray
    """
    data = torch.load(CLUSTERS_PATH, weights_only=False)

    return (
        data["token_to_cluster"],
        np.asarray(data["centroids"]),
    )


# --------------------------------------------------
# Build evaluation matrix
# --------------------------------------------------


def build_dataset(
    embeddings,
    token_to_cluster,
):
    """
    Convert stored embeddings into a matrix suitable for sklearn.

    Returns
    -------
    X : ndarray
        Shape (num_tokens, embedding_dim)

    labels : ndarray
        Cluster assignments
    """

    vectors = []
    labels = []

    for token in sorted(token_to_cluster.keys()):

        if token not in embeddings:
            continue

        mean_vector = (
            embeddings[token]
            .mean(dim=0)
            .numpy()
        )

        vectors.append(mean_vector)
        labels.append(token_to_cluster[token])

    X = np.vstack(vectors)

    return X, np.asarray(labels)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------


def evaluate():

    print("=" * 60)
    print("ROLE EMBEDDING EVALUATION")
    print("=" * 60)

    embeddings = load_embeddings()

    token_to_cluster, centroids = load_clusters()

    X, labels = build_dataset(
        embeddings,
        token_to_cluster,
    )

    num_tokens = len(labels)
    num_clusters = len(np.unique(labels))

    print(f"Tokens                : {num_tokens}")
    print(f"Clusters              : {num_clusters}")
    print()

    # ------------------------------------------
    # Standard clustering metrics
    # ------------------------------------------

    sil = silhouette_score(X, labels)

    db = davies_bouldin_score(
        X,
        labels,
    )

    ch = calinski_harabasz_score(
        X,
        labels,
    )

    print("Clustering Metrics")
    print("------------------")
    print(f"Silhouette Score     : {sil:.4f}")
    print(f"Davies-Bouldin Index : {db:.4f}")
    print(f"Calinski-Harabasz    : {ch:.2f}")
    print()

    # ------------------------------------------
    # Cluster sizes
    # ------------------------------------------

    print("Cluster Sizes")
    print("-------------")

    counts = Counter(labels)

    for cid in sorted(counts):
        print(
            f"Cluster {cid:<2} : "
            f"{counts[cid]} tokens"
        )

    print()

    # ------------------------------------------
    # Centroid distances
    # ------------------------------------------

    print("Centroid Distances")
    print("------------------")

    for i, j in combinations(
        range(len(centroids)),
        2,
    ):
        dist = np.linalg.norm(
            centroids[i] - centroids[j]
        )

        print(
            f"C{i} ↔ C{j} : "
            f"{dist:.4f}"
        )

    print()

    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    evaluate()