# meta_learner/trainer.py -- Training loop for Meta-Learner (Plan 12)
#
# Trains the MetaLearnerNetwork to maximise cumulative reward via reward
# maximisation (minimise negative reward):
#
#   Loss_t = -(w_t^T r_{t+1} - kappa ||w_t - w_{t-1}||_1 - kappa_a ||alpha_t - alpha_{t-1}||_1)
#
# where w_t = sum_k alpha_{t,k} * w_t^(k)   (composite portfolio)
#
# KEY DESIGN DECISIONS:
#   1. Sequential (not shuffled) — switching cost has temporal dependency
#   2. Detach w_prev / alpha_prev to avoid BPTT through all time steps
#   3. Smooth L1 (Huber-like) for differentiability of ||.||_1
#   4. Gradient clipping for high-dimensional softmax stability

import numpy as np
import torch
import torch.nn as nn

from regime_algo_selection.meta_learner.network import MetaLearnerNetwork
from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset


class MetaLearnerTrainer:
    """
    Trains a MetaLearnerNetwork on one walk-forward fold.

    Training processes days SEQUENTIALLY (chronological order) within each
    epoch so the switching cost term w_t - w_{t-1} is correctly computed.

    Parameters
    ----------
    network       : MetaLearnerNetwork to train (in-place)
    kappa         : portfolio switching cost coefficient (default 0.001)
    kappa_a       : algorithm switching cost coefficient (default 0 = off)
    lr            : Adam learning rate
    weight_decay  : L2 regularisation for Adam
    n_epochs      : number of full passes through the training data
    grad_clip     : max norm for gradient clipping (None = no clipping)
    """

    def __init__(
        self,
        network: MetaLearnerNetwork,
        kappa: float = 0.001,
        kappa_a: float = 0.0,
        lambda_entropy: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 50,
        grad_clip: float = 1.0,
    ):
        self.network = network
        self.kappa = kappa
        self.kappa_a = kappa_a
        self.lambda_entropy = lambda_entropy
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_l1(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Smooth approximation to L1 norm: sum_i sqrt(x_i^2 + eps).

        This is differentiable everywhere (unlike ||x||_1 which has a
        sub-gradient at 0), making it suitable for backpropagation.
        """
        return torch.sum(torch.sqrt(x ** 2 + eps))

    # ------------------------------------------------------------------

    def train_fold(
        self,
        dataset: MetaLearnerDataset,
        train_indices: np.ndarray,
    ) -> dict:
        """
        Train the meta-learner network on one walk-forward fold.

        Parameters
        ----------
        dataset       : MetaLearnerDataset (precomputed outputs + scaler fitted)
        train_indices : integer positions into dataset.dates for training

        Returns
        -------
        dict with keys 'epoch_loss' and 'epoch_reward' (lists, one entry per epoch)
        """
        self.network.train()
        N = dataset.N
        K = dataset.K
        n_train = len(train_indices)

        history = {"epoch_loss": [], "epoch_reward": []}

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0

            # Initialise previous portfolio and mixing weights (equal-weight)
            w_prev = torch.ones(N, dtype=torch.float32) / N
            alpha_prev = torch.ones(K, dtype=torch.float32) / K

            for idx in train_indices:
                # 1. Build input X_t
                X_t = torch.tensor(
                    dataset.get_input(idx), dtype=torch.float32
                )

                # 2. Forward pass: X_t -> alpha_t in Delta_K
                alpha_t = self.network(X_t)  # shape (K,)

                # 3. Algorithm outputs W_t (frozen — no gradient)
                W_t = torch.tensor(
                    dataset.get_algorithm_outputs(idx), dtype=torch.float32
                )  # shape (K, N)

                # 4. Composite portfolio: w_t = alpha_t^T W_t
                w_t = torch.matmul(alpha_t, W_t)  # shape (N,)

                # 5. Realised returns r_{t->t+1}
                r_next = torch.tensor(
                    dataset.get_returns(idx), dtype=torch.float32
                )

                # 6. Reward components
                portfolio_ret = torch.dot(w_t, r_next)
                port_cost = self.kappa * self._smooth_l1(w_t - w_prev)
                algo_cost = self.kappa_a * self._smooth_l1(alpha_t - alpha_prev)

                entropy = -torch.sum(alpha_t * torch.log(alpha_t + 1e-10))
                reward = portfolio_ret - port_cost - algo_cost
                loss = -(reward * 252.0 + self.lambda_entropy * entropy)

                # 7. Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip is not None and epoch >= 5:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=self.grad_clip
                    )

                self.optimizer.step()

                # 8. Advance state (MUST detach to prevent BPTT)
                w_prev = w_t.detach()
                alpha_prev = alpha_t.detach()

                epoch_loss += loss.item()
                epoch_reward += reward.item()

            # Record epoch averages
            history["epoch_loss"].append(epoch_loss / max(n_train, 1))
            history["epoch_reward"].append(epoch_reward / max(n_train, 1))

            if (epoch + 1) % 10 == 0:
                avg_r = epoch_reward / max(n_train, 1)
                print(
                    f"    Epoch {epoch + 1:3d}/{self.n_epochs}: "
                    f"avg_reward = {avg_r:+.6f}",
                    flush=True,
                )

        return history
