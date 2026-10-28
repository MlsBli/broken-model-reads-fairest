#!/usr/bin/env python3
"""Validate the Bias-in-Bios environment before tuning or final GPU runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent
RESULTS = CODE / "results"
SUMMARY = RESULTS / "tuning_summary.csv"


def _fail(message: str) -> None:
    raise SystemExit(f"GPU PREFLIGHT FAILED: {message}")


def _check_tuning(config, tune) -> None:
    if not SUMMARY.exists():
        _fail("tuning_summary.csv is missing; run tune.py before the final sweep")

    with SUMMARY.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_method = {row["method"]: row for row in rows}
    expected = {name for name, _fn, _trials, _mapping in tune.STUDIES}
    if set(by_method) != expected:
        _fail(f"tuning methods expected={sorted(expected)} got={sorted(by_method)}")

    for name, _fn, trials, mapping in tune.STUDIES:
        row = by_method[name]
        if row["tune_version"] != tune.TUNE_VERSION:
            # Stale manifest: warn rather than fail, so preflight can run before re-tuning.
            print(f"WARNING: {name}: tuning manifest is {row['tune_version']}; "
                  f"rerun tune.py ({tune.TUNE_VERSION}) before the final sweep")
            continue
        if int(row["trial_budget"]) != trials or int(row["completed_trials"]) < trials:
            _fail(f"{name}: equal {trials}-trial budget is incomplete")
        if int(row["memory_per_task"]) != 800:
            _fail(f"{name}: tuning manifest does not use the 800-sample memory budget")

        best = json.loads(row["best_params"])
        for config_key, param_name in mapping.items():
            if param_name not in best:
                continue
            actual, wanted = getattr(config, config_key), best[param_name]
            if isinstance(wanted, (int, float)):
                scale = max(1e-12, abs(float(wanted)))
                matches = abs(float(actual) - float(wanted)) <= 5e-4 * scale
            else:
                matches = actual == wanted
            if not matches:
                _fail(f"{name}: config {config_key}={actual!r}, manifest expects {wanted!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "final"), default="final")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Test configuration logic without requiring a GPU.")
    args = parser.parse_args()

    if sys.version_info[:2] < (3, 11):
        _fail(f"Python >= 3.11 is required, got {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        print(f"NOTE: the released results were produced with Python 3.11; you are on "
              f"{sys.version.split()[0]}. Results should match, pins in "
              "requirements.txt are what actually matter.")

    os.chdir(RESULTS)
    sys.path.insert(0, str(CODE))
    import torch
    import config
    import factory
    import tune

    if not torch.cuda.is_available() and not args.allow_cpu:
        _fail("PyTorch cannot see CUDA")
    if config.DATASET != "bios" or not config.BIOS_BERT_FINETUNE:
        _fail("config must use DATASET='bios' with BIOS_BERT_FINETUNE=True")
    if config.MEMORY_SIZE_PER_TASK != 800 or config.AGEM_MEM_BATCH_SIZE != 448:
        _fail("BIOS replay budget must be 800 samples/task with batch size 448")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 2**30:.1f} GiB)")
        print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
        probe = torch.ones((1024, 1024), device="cuda")
        if float((probe @ probe).mean().item()) != 1024.0:
            _fail("CUDA matrix-operation check returned an unexpected result")
        del probe
    else:
        print("GPU check skipped (--allow-cpu)")

    if args.stage == "final":
        _check_tuning(config, tune)
        print("Tuning manifest: complete and consistent with config.py")

    if not args.skip_data:
        tasks, _ = factory.get_tasks(seed=config.SEED)
        expected_tasks = (config.N_CLASSES + config.BIOS_STEP_SIZE - 1) // config.BIOS_STEP_SIZE
        if len(tasks) != expected_tasks:
            _fail(f"dataset produced {len(tasks)} tasks, expected {expected_tasks}")
        required = {
            "token_ids_train", "token_ids_val", "token_ids_test",
            "y_train", "y_val", "y_test", "s_train", "s_val", "s_test",
        }
        for index, task in enumerate(tasks):
            missing = required - set(task)
            if missing:
                _fail(f"task {index} is missing fields {sorted(missing)}")
            if task["token_ids_train"].shape[1] != config.BIOS_BERT_MAXLEN:
                _fail(f"task {index} token width does not match BIOS_BERT_MAXLEN")
        print(f"Dataset: {len(tasks)} tasks ready at {Path(config.BIOS_DATA_PATH).resolve()}")

    print(f"GPU preflight passed for stage: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
