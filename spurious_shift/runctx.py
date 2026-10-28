"""
runctx.py — per-seed run context: seed selection, RNG setup, and output paths.

Shared by train.py, run_ablations.py, and aggregate.py. Paths are relative to
the working directory (spurious_shift/results when run through run.sh):

    no --seed        ->  results*.npz, checkpoint.pt      in ./
    --seed N         ->  results*.npz, checkpoint.pt      in ./runs/seedN/

A seed sets *both* sources of run-to-run variation:

  - model init and batch shuffling  (torch / numpy global RNG)
  - the generated dataset itself    (get_spurious_tasks(seed=...))

Varying only the former would understate variance, since Spurious Shift is
synthetic and a fresh draw is part of what a seed should resample.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import config as cfg


def parse_seed(argv) -> int | None:
    """Read `--seed N` from argv. Returns None when absent (single-run mode)."""
    for i, arg in enumerate(argv):
        if arg == "--seed":
            if i + 1 >= len(argv):
                raise SystemExit("--seed requires an integer argument")
            try:
                return int(argv[i + 1])
            except ValueError:
                raise SystemExit(f"--seed expects an integer, got {argv[i + 1]!r}")
        if arg.startswith("--seed="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                raise SystemExit(f"--seed expects an integer, got {arg!r}")
    return None


def output_dir(seed: int | None) -> str:
    """Directory for this run's artifacts, created if needed."""
    if seed is None:
        return "."
    path = os.path.join("runs", f"seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def out_path(name: str, seed: int | None) -> str:
    return os.path.join(output_dir(seed), name)


def apply_seed(seed: int | None) -> int:
    """Seed the global RNGs. Returns the effective seed (cfg.SEED when unseeded)."""
    effective = cfg.SEED if seed is None else seed
    torch.manual_seed(effective)
    np.random.seed(effective)
    return effective


def effective_seed(argv=None) -> int:
    """
    The seed this process is running under, resolved from argv.

    Components that own an RNG — the episodic memory, whose buffer composition
    must not depend on how many draws earlier methods consumed — call this
    rather than reading cfg.SEED directly, which would pin every seed in a
    sweep to the same buffer.
    """
    seed = parse_seed(sys.argv if argv is None else argv)
    return cfg.SEED if seed is None else seed


def data_seed(seed: int | None) -> int:
    """Seed for the dataset generator — cfg.SPURIOUS_SEED unless overridden."""
    return cfg.SPURIOUS_SEED if seed is None else seed


def describe(seed: int | None) -> str:
    if seed is None:
        return f"seed={cfg.SEED} (default, writing to ./)"
    return f"seed={seed} (writing to {output_dir(seed)}/)"
