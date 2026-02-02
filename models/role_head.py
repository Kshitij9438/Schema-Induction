"""
Role projection head.

Responsibility:
- Take semantic token embeddings from the encoder
- Project them into a role-specific embedding space
- This is the ONLY trainable component (initially)

Design principles:
- Small
- Transparent
- Geometry-shaping only
- No task assumptions
"""

from torch import nn
import torch


class RoleProjectionHead(nn.Module):
    """
    Simple projection head that maps semantic embeddings
    into a role embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        role_dim: int = 128,
        normalize: bool = True,
    ):
        """
        Args:
            input_dim: dimension of encoder hidden states (e.g. 768)
            role_dim: dimension of role embedding space
            normalize: whether to L2-normalize role embeddings
        """
        super().__init__()

        self.role_dim = role_dim
        self.normalize = normalize

        # Minimal projection (no attention, no depth)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, role_dim),
            nn.ReLU(),
            nn.Linear(role_dim, role_dim),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings:
                Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            role_embeddings:
                Tensor of shape (batch_size, seq_len, role_dim)
        """
        role_embeddings = self.projection(token_embeddings)

        if self.normalize:
            role_embeddings = nn.functional.normalize(
                role_embeddings, p=2, dim=-1
            )

        return role_embeddings
