"""The world model behind lawm's adapter interface: the real checkpoint (CUDA), or a stand-in that tests the whole pipeline without a GPU (a latent moves the hand rigidly, nothing else moves)."""
from __future__ import annotations

import numpy as np

from lawm.sim.costs import SEG_HAND


class StandInModel:
    """prior = (zeros, ones) over (n_steps, n_queries, z_dim); decoder(coords, segment)(z) moves the hand by 1 cm per latent unit per step."""

    def __init__(self, n_steps=5, n_queries=2, z_dim=4):
        """Prior of zeros and ones."""
        self.prior = (np.zeros((n_steps, n_queries, z_dim), np.float32), np.ones((n_steps, n_queries, z_dim), np.float32))

    def decoder(self, coords, segment):
        """decode(z) -> cumulative displacements (pop, n_steps, N, 3)."""
        hand = segment == SEG_HAND

        def decode(z):
            """Cumulative rigid hand translations of each candidate."""
            out = np.zeros((len(z), z.shape[1], len(coords), 3), np.float32)
            out[:, :, hand] = np.cumsum(0.01 * z[:, :, 0, :3], axis=1)[:, :, None, :]
            return out
        return decode


def load_model(cfg):
    """lawm's model adapter of the configured checkpoint and prior (needs a CUDA GPU)."""
    import torch

    from lawm.sim import model_adapter
    from lawm_deploy.config import repo_path
    m = cfg["model"]
    checkpoint = repo_path(m["checkpoint"])
    model = model_adapter.load_model(checkpoint, repo_path(m["manifest"]), torch.device("cuda"))
    return model_adapter.ModelAdapter(model, model_adapter.load_prior(repo_path(m["prior"]), checkpoint))
