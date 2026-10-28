"""Shared disparity-surrogate helpers for fairness-aware methods.

FAIR-GEM and Fair-DER++ import this module. Each suite carries an identical
copy (bias_in_bios/safety.py and spurious_shift/safety.py) so that it is a
self-contained package.
"""

from __future__ import annotations

import torch


DISPARITY_MODES = {"unconditional", "conditional"}


def validate_disparity_mode(mode: str) -> str:
    if mode not in DISPARITY_MODES:
        expected = ", ".join(sorted(DISPARITY_MODES))
        raise ValueError(f"disparity_mode must be one of: {expected}")
    return mode


def conditional_disparity(
    losses: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    min_cell_size: int = 2,
) -> tuple[torch.Tensor | None, int]:
    """Average within-label max-minus-min group loss over estimable labels."""
    gaps = []
    unique_groups = groups.unique()
    for label in labels.unique():
        label_mask = labels == label
        cells = [
            losses[label_mask & (groups == group)].mean()
            for group in unique_groups
            if int((label_mask & (groups == group)).sum()) >= min_cell_size
        ]
        if len(cells) >= 2:
            stacked = torch.stack(cells)
            gaps.append(stacked.max() - stacked.min())
    if not gaps:
        return None, 0
    return torch.stack(gaps).mean(), len(gaps)
