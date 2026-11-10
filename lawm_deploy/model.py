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


def model_frame(training_up, spin_deg):
    """Rotation (3, 3) from the robot base frame (z up) to the training data's frame: z goes to the training up direction (table towards hands), the base-frame horizontal direction at spin_deg goes to the training x axis (the table's long side)."""
    up = np.asarray(training_up, float) / np.linalg.norm(training_up)
    x = np.array([1.0, 0.0, 0.0]) - up * up[0]
    x /= np.linalg.norm(x)
    t = np.radians(spin_deg)
    h = np.array([np.cos(t), np.sin(t), 0.0])
    return np.stack([x, np.cross(up, x), up], 1) @ np.stack([h, np.cross([0.0, 0.0, 1.0], h), [0.0, 0.0, 1.0]], 1).T


class FramedModel:
    """A model fed in the training data's frame: clouds are rotated by R before the model and its predicted displacements rotated back (the model reads raw coordinates, so its frame matters)."""

    def __init__(self, inner, rotation):
        """Wrap a model adapter with a fixed rotation."""
        self.inner, self.R, self.prior = inner, np.asarray(rotation, np.float32), inner.prior

    def decoder(self, coords, segment):
        """decode(z) for a base-frame cloud, returning base-frame displacements."""
        decode = self.inner.decoder((np.asarray(coords) @ self.R.T).astype(np.float32), segment)
        return lambda z: (decode(z) @ self.R).astype(np.float32)


def load_model(cfg):
    """lawm's model adapter of the configured checkpoint and prior (needs a CUDA GPU), fed in the configured training frame."""
    import torch

    from lawm.sim import model_adapter
    from lawm_deploy.config import repo_path
    m = cfg["model"]
    checkpoint = repo_path(m["checkpoint"])
    model = model_adapter.load_model(checkpoint, repo_path(m["manifest"]), torch.device("cuda"))
    adapter = model_adapter.ModelAdapter(model, model_adapter.load_prior(repo_path(m["prior"]), checkpoint))
    return FramedModel(adapter, model_frame(m["training_up"], m["spin_deg"]))
