# meta_learner/network.py -- Meta-Learner Neural Network (Plan 12)
#
# Feedforward network: X_t (29-dim) -> alpha_t in Delta_K (81-dim softmax)
# Architecture: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Softmax

import torch
import torch.nn as nn


class MetaLearnerNetwork(nn.Module):
    """
    Feedforward neural network that maps context X_t to mixing weights alpha_t.

    Input:  X_t = [asset_features (25), regime_onehot (4)] -- dim = 29
    Output: alpha_t in Delta_K (K-simplex) -- dim = K (approx. 81)

    Architecture:
        Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Softmax
    """

    def __init__(
        self,
        input_dim: int,
        n_algorithms: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        """
        Parameters
        ----------
        input_dim    : dimension of X_t (29 for initial setup)
        n_algorithms : K (81 for full algorithm space)
        hidden_dims  : list of hidden layer sizes, default [128, 64]
        dropout      : dropout probability applied after each ReLU
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, n_algorithms)
        self.temperature = 1.0

        # Weight initialisation: Xavier for stability with softmax output
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.input_dim = input_dim
        self.n_algorithms = n_algorithms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns
        -------
        alpha : mixing weights, shape matches input (softmax applied over last dim)
        """
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        return torch.softmax(logits / self.temperature, dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits before softmax (useful for diagnostics)."""
        h = self.feature_extractor(x)
        return self.output_layer(h)
