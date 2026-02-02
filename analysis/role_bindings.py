"""
Role bindings for schema-free extraction.

Purpose:
- Bind discovered cluster IDs to human-readable role names
- This is a POST-HOC interpretation step
- NO learning happens here
- NO feedback into training

This file exists to make extraction usable.
"""

from pathlib import Path
import torch


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_PATH = ROOT / "analysis" / "role_clusters.pt"


# -----------------------------
# Load discovered clusters
# -----------------------------

if not CLUSTERS_PATH.exists():
    raise FileNotFoundError(
        f"❌ role_clusters.pt not found at {CLUSTERS_PATH}\n"
        f"Run analysis/cluster.py first."
    )

cluster_data = torch.load(CLUSTERS_PATH, weights_only=False)


TOKEN_TO_CLUSTER = cluster_data["token_to_cluster"]
CLUSTER_CENTROIDS = cluster_data["centroids"]


# -----------------------------
# Cluster → Role mapping
# -----------------------------
"""
IMPORTANT:

These bindings are NOT learned.
They are NOT used during training.
They are NOT supervision.

They are simply names assigned to clusters
that the system already discovered.

This is equivalent to naming PCA axes.
"""

CLUSTER_ROLE_MAP = {
    1: "amount",
    2: "item",
    3: "person",
    0: "time",   # NEW: temporal role
}



# -----------------------------
# Public API
# -----------------------------

def get_role_for_token(token: str) -> str | None:
    """
    Return the role name for a token, if any.

    Args:
        token: token string (as produced by tokenizer)

    Returns:
        role name (str) or None if token is unassigned
    """
    cluster_id = TOKEN_TO_CLUSTER.get(token)

    if cluster_id is None:
        return None

    return CLUSTER_ROLE_MAP.get(cluster_id)


def get_centroid_for_role(role: str):
    """
    Return the centroid embedding for a given role.

    Args:
        role: role name (e.g. 'amount')

    Returns:
        centroid vector (numpy array)
    """
    for cid, r in CLUSTER_ROLE_MAP.items():
        if r == role:
            return CLUSTER_CENTROIDS[cid]

    raise KeyError(f"Role '{role}' not found in CLUSTER_ROLE_MAP")


def available_roles():
    """
    List all bound roles.
    """
    return list(CLUSTER_ROLE_MAP.values())
