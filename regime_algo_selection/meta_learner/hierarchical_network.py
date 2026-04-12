# meta_learner/hierarchical_network.py -- Hierarchical Meta-Learner (Plan 13a)
#
# Three-component architecture:
#   TierSelector    : Softmax over 3 tiers, β_t ∈ Δ_3
#   TierSpecialist  : Softmax over K_f algorithms within one tier, γ^(f)_t ∈ Δ_{K_f}
#   HierarchicalMetaLearner : Combines them → α_{t,k} = β_{t,f} · γ^(f)_{t,j}
#
# All use hidden_dims=[64,32], dropout=0.1, Xavier init.
#
# Combined weight formula:
#   α_{t,k} = β_{t,f} · γ^(f)_{t,j}   for algorithm k in tier f at position j
#
# This guarantees α_t ∈ Δ_K because:
#   Σ_k α_{t,k} = Σ_f β_{t,f} · Σ_j γ^(f)_{t,j} = Σ_f β_{t,f} · 1 = 1

import torch
import torch.nn as nn


class TierSelector(nn.Module):
    """
    Level 1: Selects mixing weights over 3 tiers.

    Small network — only 3 output dims. H_max = log(3) ≈ 1.10.
    No phase-transition issues expected at this scale.

    Input  : X_t ∈ R^{input_dim}  (29 dims: 25 asset features + 4 regime one-hot)
    Output : β_t ∈ Δ_3            (softmax over 3 tiers)
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 3)

        # Xavier uniform initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (..., input_dim)

        Returns
        -------
        β_t : Tensor of shape (..., 3)
        """
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)


class TierSpecialist(nn.Module):
    """
    Level 2: Selects mixing weights within one tier.

    Independent network — does not share parameters with the selector or
    other specialists. Largest softmax is K1≈48 (vs. K=81 in Plan 12),
    giving H_max ≈ 3.87 instead of 4.39.

    Input  : X_t ∈ R^{input_dim}
    Output : γ^(f)_t ∈ Δ_{K_f}  (softmax over K_f algorithms in this tier)
    """

    def __init__(
        self,
        input_dim: int,
        n_algorithms: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, n_algorithms)

        # Xavier uniform initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (..., input_dim)

        Returns
        -------
        γ^(f)_t : Tensor of shape (..., n_algorithms)
        """
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)


class HierarchicalMetaLearner(nn.Module):
    """
    Combines TierSelector + 3 TierSpecialists into one hierarchical module.

    Combined weight: α_{t,k} = β_{t,f} · γ^(f)_{t,j}

    Guarantees α_t ∈ Δ_K because:
        Σ_k α_{t,k} = Σ_f β_{t,f} · Σ_j γ^(f)_{t,j} = Σ_f β_{t,f} · 1 = 1

    Parameter counts (with input_dim=29, hidden=[64,32]):
        TierSelector    : 29→64→32→3   ≈ 4,067 params
        Specialist T1   : 29→64→32→48  ≈ 5,504 params
        Specialist T2   : 29→64→32→33  ≈ 5,023 params
        Specialist T3   : 29→64→32→36  ≈ 5,120 params
        Total           : ≈ 19,714 params
    """

    def __init__(
        self,
        input_dim: int,
        tier_sizes: list,
        selector_hidden: list = None,
        specialist_hidden: list = None,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        input_dim        : dimension of input X_t (29 in Plan 13a)
        tier_sizes       : list of K_f, the number of algorithms per tier.
                           E.g. [48, 33, 36] for tiers=[1,2,3].
        selector_hidden  : hidden dims for TierSelector. Default [64,32].
        specialist_hidden: hidden dims for each TierSpecialist. Default [64,32].
        dropout          : dropout rate applied after each hidden ReLU layer.
        """
        super().__init__()
        if selector_hidden is None:
            selector_hidden = [64, 32]
        if specialist_hidden is None:
            specialist_hidden = [64, 32]

        self.tier_sizes = tier_sizes
        self.n_tiers = len(tier_sizes)
        self.total_algorithms = sum(tier_sizes)

        self.tier_selector = TierSelector(input_dim, selector_hidden, dropout)
        self.specialists = nn.ModuleList([
            TierSpecialist(input_dim, k_f, specialist_hidden, dropout)
            for k_f in tier_sizes
        ])

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the full hierarchical model.

        Parameters
        ----------
        x : Tensor of shape (..., input_dim)

        Returns
        -------
        alpha_t : Tensor of shape (..., total_algorithms)
            Combined per-algorithm mixing weights (sums to 1).
        beta_t  : Tensor of shape (..., n_tiers)
            Tier-level weights.
        gammas  : list of Tensors, each of shape (..., K_f)
            Within-tier weights for each specialist.
        """
        beta_t = self.tier_selector(x)               # (..., 3)
        gammas = [spec(x) for spec in self.specialists]  # [(..., K_f), ...]

        # α_{t,k} = β_{t,f} · γ^(f)_{t,j}
        alpha_parts = []
        for f in range(self.n_tiers):
            alpha_parts.append(beta_t[..., f : f + 1] * gammas[f])
        alpha_t = torch.cat(alpha_parts, dim=-1)      # (..., total_K)

        return alpha_t, beta_t, gammas
