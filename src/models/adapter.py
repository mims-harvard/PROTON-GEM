"""Molecular adapter for mapping molecular embeddings to PROTON embedding space.

This module provides the MolecularAdapter class, an MLP network that maps
Uni-Mol2 molecular representations into PROTON's embedding space, enabling
reasoning over novel molecules without retraining the PROTON model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MolecularAdapter(nn.Module):
    """MLP adapter to map Uni-Mol2 molecular representations to PROTON embedding space.

    This adapter is trained using existing drugs in NeuroKG to learn a mapping
    from Uni-Mol2's 512-dimensional molecular embeddings to PROTON's 512-dimensional
    learned drug embeddings.

    Args:
        input_dim: Dimension of input Uni-Mol2 embeddings. Default 512.
        hidden_dim: Dimension of hidden layers. Default 512.
        output_dim: Dimension of output PROTON embeddings. Default 512.
        num_layers: Number of hidden layers. Default 2.
        dropout: Dropout probability. Default 0.1.

    Example:
        >>> adapter = MolecularAdapter()
        >>> unimol_emb = torch.randn(32, 512)  # batch of Uni-Mol2 embeddings
        >>> proton_emb = adapter(unimol_emb)   # mapped to PROTON space
        >>> proton_emb.shape
        torch.Size([32, 512])
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.layers(x)
