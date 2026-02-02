"""
Contrastive loss for role-based representation learning.

Responsibility:
- Enforce relative distance constraints:
    anchor should be closer to positive than to negative
- Operates purely on geometry (no labels, no classes)

This loss does NOT:
- classify
- predict schema
- infer correctness

It only reshapes embedding space.
"""

import torch
from torch import nn
import torch.nn.functional as F


class TripletContrastiveLoss(nn.Module):
    """
    Margin-based triplet contrastive loss.

    Enforces:
        dist(anchor, positive) + margin < dist(anchor, negative)
    """

    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: minimum separation required between
                    positive and negative distances
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor:
                Tensor of shape (N, D)
            positive:
                Tensor of shape (N, D)
            negative:
                Tensor of shape (N, D)
            weight:
                Optional tensor of shape (N,)
                Used to scale loss contribution per pair

        Returns:
            Scalar loss tensor
        """

        # Cosine distance (since embeddings are L2-normalized)
        pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)

        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)

        if weight is not None:
            losses = losses * weight

        return losses.mean()
