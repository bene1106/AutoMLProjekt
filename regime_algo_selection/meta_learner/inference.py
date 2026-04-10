# meta_learner/inference.py -- Meta-Learner Inference (Plan 12)
#
# At decision time: X_t -> alpha_t -> w_t = sum_k alpha_{t,k} * w_t^(k)

import numpy as np
import torch

from regime_algo_selection.meta_learner.network import MetaLearnerNetwork


class MetaLearnerAgent:
    """
    Wraps a trained MetaLearnerNetwork for inference.

    At each time step t:
        1. Receives X_t = [asset_features, regime_onehot]
        2. Feeds X_t through the trained network -> alpha_t in Delta_K
        3. Computes composite portfolio w_t = alpha_t^T W_t

    Parameters
    ----------
    network    : trained MetaLearnerNetwork (will be set to eval mode)
    algorithms : list of K PortfolioAlgorithm instances (for bookkeeping)
    """

    def __init__(self, network: MetaLearnerNetwork, algorithms: list):
        self.network = network
        self.network.eval()
        self.algorithms = algorithms
        self.K = len(algorithms)

    def select(
        self,
        x_t: np.ndarray,
        algorithm_outputs: np.ndarray,
    ) -> tuple:
        """
        Compute the composite portfolio for one time step.

        Parameters
        ----------
        x_t               : input vector of shape (input_dim,) — already scaled
        algorithm_outputs : pre-computed algo weights, shape (K, N)

        Returns
        -------
        w_t     : composite portfolio weights, shape (N,), sums to ≈1
        alpha_t : mixing weights, shape (K,), sums to 1
        """
        with torch.no_grad():
            x_tensor = torch.tensor(x_t, dtype=torch.float32)
            alpha_t = self.network(x_tensor)  # (K,)

            W = torch.tensor(algorithm_outputs, dtype=torch.float32)
            w_t = torch.matmul(alpha_t, W)  # (N,)

        alpha_np = alpha_t.numpy()
        w_np = w_t.numpy()

        # Safety normalisation (softmax guarantees sum≈1, but float noise)
        w_np = np.clip(w_np, 0.0, None)
        total = w_np.sum()
        if total > 1e-12:
            w_np = w_np / total
        else:
            w_np = np.ones(len(w_np)) / len(w_np)

        return w_np, alpha_np

    def top_algorithms(self, alpha_t: np.ndarray, n: int = 5) -> list:
        """
        Return the names of the top-n algorithms by mixing weight.

        Useful for qualitative inspection of the learned policy.
        """
        top_idx = np.argsort(alpha_t)[::-1][:n]
        return [(self.algorithms[i].name, float(alpha_t[i])) for i in top_idx]

    @staticmethod
    def entropy(alpha_t: np.ndarray, eps: float = 1e-10) -> float:
        """Compute Shannon entropy H(alpha_t) = -sum_k alpha_k * log(alpha_k)."""
        return float(-np.sum(alpha_t * np.log(alpha_t + eps)))
