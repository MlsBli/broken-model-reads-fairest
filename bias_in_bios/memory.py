"""
memory.py — Stratified episodic memory buffer.

Stores past task samples stratified by sensitive group so minority groups
are represented in the memory gradient — important for fairness tasks.
"""
# NOTE on "stratified": slots are allocated proportionally to group size after
# a floor of 5, and groups with <10 samples never enter memory at all — this
# guarantees representation, not balance.

import numpy as np
import torch

import runctx
from config import MEMORY_SIZE_PER_TASK, MEMORY_STRATIFIED


class EpisodicMemory:
    """
    Buffer draws use a private RNG, not the global NumPy stream.

    With the global stream, which samples a method stores would depend on how
    many draws every *earlier* method in train.py's fixed call order had
    consumed, so a partial run (e.g. `--novel-only`) would store different
    buffers than a full run. Seeding per instance from the run context makes
    each method's buffer depend only on the seed: identical across methods
    within a run, different across seeds.
    """

    def __init__(
        self,
        size_per_task=MEMORY_SIZE_PER_TASK,
        stratified=MEMORY_STRATIFIED,
        device="cpu",
        seed=None,
    ):
        self.size_per_task = size_per_task
        self.stratified = stratified
        self.device = device
        self.rng = np.random.default_rng(
            runctx.effective_seed() if seed is None else seed)

        self.X_buf = []
        self.y_buf = []
        self.s_buf = []
        self.z_buf = []  # stored logits for DER++ replay

        self._X_flat = None
        self._y_flat = None
        self._s_flat = None
        self._z_flat = None

    def _rebuild_flat(self):
        if self.X_buf:
            self._X_flat = torch.cat(self.X_buf)
            self._y_flat = torch.cat(self.y_buf)
            self._s_flat = torch.cat(self.s_buf)
        if self.z_buf:
            self._z_flat = torch.cat(self.z_buf)

    @staticmethod
    def _x_dtype(X):
        return torch.long if X.dtype in (np.int32, np.int64) else torch.float32

    def add_task(self, X, y, s):
        """Sample from a task and add to memory (stratified or random)."""
        if self.stratified:
            idx = self._stratified_sample(s)
        else:
            idx = self.rng.choice(len(X), min(self.size_per_task, len(X)), replace=False)

        self.X_buf.append(torch.tensor(X[idx], dtype=self._x_dtype(X)).to(self.device))
        self.y_buf.append(torch.tensor(y[idx], dtype=torch.long).to(self.device))
        self.s_buf.append(torch.tensor(s[idx], dtype=torch.long).to(self.device))
        self._rebuild_flat()

    def _stratified_sample(self, s):
        """Allocate slots proportionally by group size, with a floor per eligible group."""
        groups, counts = np.unique(s, return_counts=True)
        eligible = [(g, c) for g, c in zip(groups, counts) if c >= 10]

        if not eligible:
            return self.rng.choice(len(s), min(self.size_per_task, len(s)), replace=False)

        total = sum(c for _, c in eligible)
        slots = {}
        remaining = self.size_per_task

        min_slots = 5
        for g, c in eligible:
            slots[g] = min_slots
            remaining -= min_slots

        for g, c in eligible:
            extra = int(remaining * c / total)
            slots[g] += extra

        indices = []
        for g, _ in eligible:
            g_idx = np.where(s == g)[0]
            n = min(slots[g], len(g_idx))
            indices.append(self.rng.choice(g_idx, n, replace=False))

        return np.concatenate(indices)


    def get_all(self):
        """All stored samples as one batch (FSW group statistics over the buffer)."""
        if self._X_flat is None:
            return None
        return self._X_flat, self._y_flat, self._s_flat

    def get_random_batch(self, batch_size):
        """Return a random mini-batch sampled without replacement from memory."""
        if self._X_flat is None:
            return None
        n = self._X_flat.shape[0]
        idx = torch.as_tensor(self.rng.permutation(n)[:min(batch_size, n)], dtype=torch.long)
        return self._X_flat[idx], self._y_flat[idx], self._s_flat[idx]

    def add_task_with_logits(self, X, y, s, logits):
        """Like add_task but also stores model logits for DER++ replay."""
        if self.stratified:
            idx = self._stratified_sample(s)
        else:
            idx = self.rng.choice(len(X), min(self.size_per_task, len(X)), replace=False)

        self.X_buf.append(torch.tensor(X[idx], dtype=self._x_dtype(X)).to(self.device))
        self.y_buf.append(torch.tensor(y[idx], dtype=torch.long).to(self.device))
        self.s_buf.append(torch.tensor(s[idx], dtype=torch.long).to(self.device))
        self.z_buf.append(logits[idx].detach().to(self.device))
        self._rebuild_flat()

    def get_random_batch_with_logits(self, batch_size):
        """Return a random mini-batch including stored logits (for DER++ replay)."""
        if self._X_flat is None or self._z_flat is None:
            return None
        n = self._X_flat.shape[0]
        idx = torch.as_tensor(self.rng.permutation(n)[:min(batch_size, n)], dtype=torch.long)
        return self._X_flat[idx], self._y_flat[idx], self._s_flat[idx], self._z_flat[idx]

    def num_tasks(self):
        return len(self.X_buf)

    def total_size(self):
        return sum(x.shape[0] for x in self.X_buf)

    def group_counts(self):
        """Count samples per group across all memory."""
        if not self.s_buf:
            return {}
        s_all = torch.cat(self.s_buf).cpu().numpy()
        groups, counts = np.unique(s_all, return_counts=True)
        return dict(zip(groups.tolist(), counts.tolist()))
