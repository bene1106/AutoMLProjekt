# meta_learner/hierarchical_trainer.py -- Two-Phase Hierarchical Trainer (Plan 13a)
#
# Phase A: Train each TierSpecialist independently on its own tier's algorithms.
#          Specialist sees only its tier, optimises as if β_f = 1.
#          Loss: -(reward × 252.0 + λ_spec × H(γ^(f)_t))
#
# Phase B: Freeze all specialists, train only the TierSelector.
#          Selector blends the 3 frozen tier-level composite portfolios.
#          Loss: -(reward × 252.0 + λ_tier × H(β_t))
#
# Both phases use SEQUENTIAL (non-shuffled) time processing because the
# switching cost w_t - w_{t-1} creates temporal dependencies.

import numpy as np
import torch
import torch.nn as nn

from regime_algo_selection.meta_learner.hierarchical_network import HierarchicalMetaLearner
from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset


class HierarchicalTrainer:
    """
    Two-phase trainer for the hierarchical meta-learner.

    Phase A: Train each specialist independently on its own tier's algorithms.
    Phase B: Freeze specialists, train tier selector on blended portfolio.

    Both phases use sequential (non-shuffled) processing through time
    because switching costs create temporal dependencies (same as Plan 12).

    Parameters
    ----------
    model                : HierarchicalMetaLearner instance
    tier_algorithm_indices: list of 3 lists, global indices of algorithms
                            per tier. E.g. [list(range(48)), list(range(48,81)),
                            list(range(81,117))]
    kappa                : portfolio switching cost coefficient (default 0.001)
    kappa_a              : algorithm switching cost (0 = off for first run)
    specialist_lr        : Adam learning rate for Phase A
    selector_lr          : Adam learning rate for Phase B
    specialist_epochs    : number of training epochs per specialist (Phase A)
    selector_epochs      : number of training epochs for tier selector (Phase B)
    lambda_spec          : entropy regularisation weight for specialists
    lambda_tier          : entropy regularisation weight for tier selector
    weight_decay         : L2 regularisation for Adam
    grad_clip            : gradient clipping max norm (applied after epoch 5)
    """

    def __init__(
        self,
        model: HierarchicalMetaLearner,
        tier_algorithm_indices: list,
        kappa: float = 0.001,
        kappa_a: float = 0.0,
        specialist_lr: float = 0.005,
        selector_lr: float = 0.005,
        specialist_epochs: int = 80,
        selector_epochs: int = 50,
        lambda_spec: float = 0.05,
        lambda_tier: float = 0.05,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.tier_indices = tier_algorithm_indices  # list of 3 lists
        self.kappa = kappa
        self.kappa_a = kappa_a
        self.lambda_spec = lambda_spec
        self.lambda_tier = lambda_tier
        self.specialist_epochs = specialist_epochs
        self.selector_epochs = selector_epochs
        self.grad_clip = grad_clip

        # One optimizer per specialist (Phase A)
        self.specialist_optimizers = [
            torch.optim.Adam(
                spec.parameters(), lr=specialist_lr, weight_decay=weight_decay
            )
            for spec in model.specialists
        ]

        # One optimizer for the tier selector (Phase B)
        self.selector_optimizer = torch.optim.Adam(
            model.tier_selector.parameters(),
            lr=selector_lr,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_l1(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Smooth L1 norm: Σ sqrt(x_i² + ε). Differentiable at 0."""
        return torch.sum(torch.sqrt(x ** 2 + eps))

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H(p) = -Σ p_i log(p_i)."""
        return -torch.sum(p * torch.log(p + 1e-10))

    # ------------------------------------------------------------------
    # Phase A: train each specialist independently
    # ------------------------------------------------------------------

    def train_phase_a(
        self, dataset: MetaLearnerDataset, train_indices: np.ndarray
    ) -> list:
        """
        Phase A: Train each TierSpecialist independently.

        For specialist f, only algorithms in tier_indices[f] are used.
        The specialist's composite portfolio is computed as:
            w_f_t = γ^(f)_t @ W_t[tier_f_indices]
        treating β_f = 1 (as if this specialist controls the whole portfolio).

        Returns
        -------
        list of dicts, one per specialist, each with 'epoch_loss' and 'epoch_reward'.
        """
        N = dataset.N
        n_train = len(train_indices)
        histories = []

        for f, spec in enumerate(self.model.specialists):
            t_idx = self.tier_indices[f]   # global algo indices for this tier
            K_f = len(t_idx)
            optimizer = self.specialist_optimizers[f]

            history = {"epoch_loss": [], "epoch_reward": []}
            spec.train()

            for epoch in range(self.specialist_epochs):
                epoch_loss = 0.0
                epoch_reward = 0.0

                # Initialise previous tier portfolio (equal-weight within tier)
                w_prev = torch.ones(N, dtype=torch.float32) / N

                for idx in train_indices:
                    # 1. Input features
                    X_t = torch.tensor(
                        dataset.get_input(idx), dtype=torch.float32
                    )

                    # 2. Within-tier weights γ^(f)_t  (K_f,)
                    gamma_t = spec(X_t)

                    # 3. Algorithm outputs for this tier only (K_f, N) — frozen
                    W_all = dataset.get_algorithm_outputs(idx)  # (K_total, N)
                    W_f = torch.tensor(
                        W_all[t_idx], dtype=torch.float32
                    )  # (K_f, N)

                    # 4. Tier-level composite portfolio (as if β_f = 1)
                    w_t = torch.matmul(gamma_t, W_f)  # (N,)

                    # 5. Next-period returns
                    r_next = torch.tensor(
                        dataset.get_returns(idx), dtype=torch.float32
                    )

                    # 6. Reward + entropy loss
                    portfolio_ret = torch.dot(w_t, r_next)
                    port_cost = self.kappa * self._smooth_l1(w_t - w_prev)
                    entropy = self._entropy(gamma_t)
                    reward = portfolio_ret - port_cost
                    loss = -(reward * 252.0 + self.lambda_spec * entropy)

                    # 7. Backward
                    optimizer.zero_grad()
                    loss.backward()

                    if self.grad_clip is not None and epoch >= 5:
                        nn.utils.clip_grad_norm_(
                            spec.parameters(), max_norm=self.grad_clip
                        )

                    optimizer.step()

                    # 8. Advance state (detach to prevent BPTT)
                    w_prev = w_t.detach()

                    epoch_loss += loss.item()
                    epoch_reward += reward.item()

                history["epoch_loss"].append(epoch_loss / max(n_train, 1))
                history["epoch_reward"].append(epoch_reward / max(n_train, 1))

                if (epoch + 1) % 10 == 0:
                    avg_r = epoch_reward / max(n_train, 1)
                    print(
                        f"    [Specialist {f+1}] Epoch {epoch+1:3d}/{self.specialist_epochs}: "
                        f"avg_reward = {avg_r:+.6f}",
                        flush=True,
                    )

            histories.append(history)

        return histories

    # ------------------------------------------------------------------
    # Phase B: train tier selector with frozen specialists
    # ------------------------------------------------------------------

    def train_phase_b(
        self, dataset: MetaLearnerDataset, train_indices: np.ndarray
    ) -> dict:
        """
        Phase B: Freeze all specialists, train TierSelector.

        Each specialist produces a frozen tier-level portfolio:
            w_f_t = γ^(f)_t @ W_t[tier_f_indices]   (no grad through specialists)

        The selector learns β_t to blend the 3 tier portfolios:
            w_t = Σ_f β_{t,f} · w_f_t

        Returns
        -------
        dict with 'epoch_loss' and 'epoch_reward' lists.
        """
        N = dataset.N
        n_train = len(train_indices)

        # Freeze all specialist parameters
        for spec in self.model.specialists:
            for param in spec.parameters():
                param.requires_grad = False
            spec.eval()

        self.model.tier_selector.train()
        history = {"epoch_loss": [], "epoch_reward": []}

        for epoch in range(self.selector_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0

            # Initialise previous composite portfolio (equal-weight)
            w_prev = torch.ones(N, dtype=torch.float32) / N

            for idx in train_indices:
                # 1. Input features
                X_t = torch.tensor(
                    dataset.get_input(idx), dtype=torch.float32
                )

                # 2. Tier weights β_t  (3,)  — grad flows here
                beta_t = self.model.tier_selector(X_t)

                # 3. Algorithm outputs (full K, N) — frozen constants
                W_all = dataset.get_algorithm_outputs(idx)  # (K_total, N)

                # 4. Tier-level portfolios (frozen specialists → no grad)
                tier_portfolios = []
                with torch.no_grad():
                    for f, spec in enumerate(self.model.specialists):
                        t_idx = self.tier_indices[f]
                        W_f = torch.tensor(
                            W_all[t_idx], dtype=torch.float32
                        )
                        gamma_f = spec(X_t)
                        w_f = torch.matmul(gamma_f, W_f)  # (N,)
                        tier_portfolios.append(w_f)

                # 5. Blended composite portfolio
                #    w_t = Σ_f β_{t,f} · w_f_t
                w_t = sum(
                    beta_t[f] * tier_portfolios[f]
                    for f in range(self.model.n_tiers)
                )

                # 6. Next-period returns
                r_next = torch.tensor(
                    dataset.get_returns(idx), dtype=torch.float32
                )

                # 7. Reward + entropy loss (entropy on β_t)
                portfolio_ret = torch.dot(w_t, r_next)
                port_cost = self.kappa * self._smooth_l1(w_t - w_prev)
                entropy_tier = self._entropy(beta_t)
                reward = portfolio_ret - port_cost
                loss = -(reward * 252.0 + self.lambda_tier * entropy_tier)

                # 8. Backward (only selector params have grad)
                self.selector_optimizer.zero_grad()
                loss.backward()

                if self.grad_clip is not None and epoch >= 5:
                    nn.utils.clip_grad_norm_(
                        self.model.tier_selector.parameters(),
                        max_norm=self.grad_clip,
                    )

                self.selector_optimizer.step()

                # 9. Advance state
                w_prev = w_t.detach()

                epoch_loss += loss.item()
                epoch_reward += reward.item()

            history["epoch_loss"].append(epoch_loss / max(n_train, 1))
            history["epoch_reward"].append(epoch_reward / max(n_train, 1))

            if (epoch + 1) % 5 == 0:
                avg_r = epoch_reward / max(n_train, 1)
                print(
                    f"    [Selector]   Epoch {epoch+1:3d}/{self.selector_epochs}: "
                    f"avg_reward = {avg_r:+.6f}",
                    flush=True,
                )

        # Unfreeze specialists after Phase B (allow re-use / further training)
        for spec in self.model.specialists:
            for param in spec.parameters():
                param.requires_grad = True

        return history

    # ------------------------------------------------------------------
    # Combined: Phase A then Phase B
    # ------------------------------------------------------------------

    def train_fold(
        self, dataset: MetaLearnerDataset, train_indices: np.ndarray
    ) -> dict:
        """
        Full two-phase training for one walk-forward fold.

        Returns
        -------
        dict with keys 'specialist_histories' (list of 3) and 'selector_history'.
        """
        print("--- Phase A: Training Specialists ---", flush=True)
        specialist_histories = self.train_phase_a(dataset, train_indices)
        print("--- Phase B: Training Tier Selector ---", flush=True)
        selector_history = self.train_phase_b(dataset, train_indices)
        return {
            "specialist_histories": specialist_histories,
            "selector_history": selector_history,
        }
