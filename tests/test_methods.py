#!/usr/bin/env python3
"""
test_methods.py — functional checks on the Spurious Shift methods.

Runs each method for one epoch on a small slice and asserts the invariants that
keep the nine methods comparable to each other. It is intentionally tiny: a
functional test, not a reproduction run.

The BatchNorm check is the important one. FAIR-GEM and Fair-DER++ compute an
extra forward pass per step for the disparity safety gradient. That pass must
not update BatchNorm's running statistics, or the proposed methods accumulate
more momentum updates per step than the baselines they are measured against and
end up with different inference-time behaviour for reasons unrelated to the
method.

Requires the project dependencies. Usage:
    python tests/test_methods.py
    python tests/test_methods.py --n-train 512 --epochs 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "spurious_shift"
sys.path.insert(0, str(CODE_DIR))

warnings.filterwarnings("ignore")

N_TRAIN = 256
N_TEST = 128
EPOCHS = 1
THREADS = int(os.environ.get("SMOKE_TORCH_THREADS", "1"))

torch = None
nn = None
get_spurious_tasks = None
MLP = None
batchnorm_eval = None
conditional_disparity = None
METHODS = []


def load_deps():
    """Import torch and project modules only after CLI parsing succeeds."""
    global torch, nn, get_spurious_tasks, MLP, batchnorm_eval
    global conditional_disparity, METHODS

    print("test_methods: importing torch and project modules...", flush=True)
    try:
        import torch as _torch  # noqa: PLC0415
        import torch.nn as _nn  # noqa: PLC0415
        # Force torch's lazy submodules in NOW. Optimizer construction lazily
        # imports torch._dynamo; when that first happens inside this test's
        # module-reload loop, the import-lock interaction deadlocks on some
        # setups (observed: macOS, Python 3.13).
        import torch._dynamo  # noqa: F401, PLC0415
        _torch.optim.AdamW([_torch.nn.Parameter(_torch.zeros(1))], lr=1e-3)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency {exc.name!r}. Install requirements first, e.g. "
            "`uv pip install -r requirements.txt`."
        ) from exc

    try:
        from datasets.spurious_shift import get_spurious_tasks as _get_tasks  # noqa: PLC0415
        from baselines_neural import JointRetrain, Finetune, AGEM, DERpp  # noqa: PLC0415
        from baselines_fairness import GroupDRO, FAIRL  # noqa: PLC0415
        from fsw import FSW  # noqa: PLC0415
        from fairgem import FAIRGEM  # noqa: PLC0415
        from fairderpp import FairDERpp  # noqa: PLC0415
        from derpp_penalty import DERppPenalty  # noqa: PLC0415
        from model import MLP as _MLP, batchnorm_eval as _batchnorm_eval  # noqa: PLC0415
        from safety import conditional_disparity as _conditional_disparity  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Project import failed: missing module {exc.name!r}. Run from the "
            "repository root after installing requirements."
        ) from exc

    torch = _torch
    nn = _nn
    get_spurious_tasks = _get_tasks
    MLP = _MLP
    batchnorm_eval = _batchnorm_eval
    conditional_disparity = _conditional_disparity
    METHODS = [
        ("joint_retrain", JointRetrain, False),
        ("finetune",      Finetune,     False),
        ("agem",          AGEM,         True),
        ("der_pp",        DERpp,        True),
        ("group_dro",     GroupDRO,     True),
        ("fairl",         FAIRL,        True),
        ("fsw",           FSW,          True),
        ("fair_gem",      FAIRGEM,      True),
        ("fair_derpp",    FairDERpp,    True),
        ("derpp_penalty", DERppPenalty, True),
    ]

def _bn_modules(model):
    return [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def check_batchnorm_eval_restores(input_dim: int) -> list[str]:
    """
    batchnorm_eval must leave running statistics untouched and restore the
    module's prior mode.

    It must NOT be asserted that BatchNorm ends up in eval mode. The default is
    "freeze", which deliberately keeps train-mode batch-statistic normalisation
    — so that the auxiliary gradient measures disparity on the batch it is
    looking at — and suppresses only the running-statistics update.
    """
    failures = []
    for start_training in (True, False):
        model = MLP(input_dim)
        model.train(start_training)
        bns = _bn_modules(model)
        if start_training:
            model(torch.randn(64, input_dim))          # populate running stats
        before = [(b.running_mean.clone(), b.running_var.clone(),
                   int(b.num_batches_tracked.item())) for b in bns]

        with batchnorm_eval(model):
            if start_training:
                model(torch.randn(64, input_dim))      # an auxiliary pass

        for b, (rm, rv, nb) in zip(bns, before):
            if not torch.allclose(b.running_mean, rm) or not torch.allclose(b.running_var, rv):
                failures.append("batchnorm_eval let an auxiliary pass update "
                                "running statistics")
                break
            if int(b.num_batches_tracked.item()) != nb:
                failures.append("batchnorm_eval let num_batches_tracked advance")
                break

        if any(b.training != start_training for b in bns):
            failures.append(f"batchnorm_eval did not restore training={start_training}")
    return failures


def check_norm_study_modes(input_dim: int) -> list[str]:
    """Normalization-study switches must be distinct and autograd-safe."""
    failures = []
    prior_norm = os.environ.get("CLFAIR_NORM")
    prior_mode = os.environ.get("CLFAIR_BN_MODE")
    try:
        expected = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}
        for kind, cls in expected.items():
            os.environ["CLFAIR_NORM"] = kind
            model = MLP(input_dim)
            norms = [module for module in model.modules() if isinstance(module, cls)]
            if len(norms) != 2:
                failures.append(f"CLFAIR_NORM={kind} created {len(norms)} {cls.__name__} layers")

        os.environ["CLFAIR_NORM"] = "batch"
        os.environ["CLFAIR_BN_MODE"] = "freeze"
        model = MLP(input_dim).train()
        bns = _bn_modules(model)
        means = [bn.running_mean.clone() for bn in bns]
        counts = [int(bn.num_batches_tracked.item()) for bn in bns]
        x = torch.randn(16, input_dim)
        with batchnorm_eval(model):
            output = model(x)
        output.sum().backward()
        if any(not torch.equal(bn.running_mean, before)
               for bn, before in zip(bns, means)):
            failures.append("CLFAIR_BN_MODE=freeze changed running means")
        if any(int(bn.num_batches_tracked.item()) != before
               for bn, before in zip(bns, counts)):
            failures.append("CLFAIR_BN_MODE=freeze changed batch counters")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"normalization study mode failed: {type(exc).__name__}: {exc}")
    finally:
        if prior_norm is None:
            os.environ.pop("CLFAIR_NORM", None)
        else:
            os.environ["CLFAIR_NORM"] = prior_norm
        if prior_mode is None:
            os.environ.pop("CLFAIR_BN_MODE", None)
        else:
            os.environ["CLFAIR_BN_MODE"] = prior_mode
    return failures


def check_bn_mode_switch(input_dim: int) -> list[str]:
    """
    CLFAIR_BN_MODE is the only BatchNorm knob, and every value must be honoured.

    Two things are asserted, so that a misspelled or unsupported setting
    errors instead of silently selecting a different protocol:
      * each recognised mode changes what it claims to change,
      * an unrecognised mode raises instead of falling through to a default.
    """
    failures = []
    tracked = ("CLFAIR_BN_MODE",)
    prior = {k: os.environ.get(k) for k in tracked}

    def _run_aux():
        """Run one auxiliary pass; return whether running stats moved."""
        model = MLP(input_dim).train()
        model(torch.randn(64, input_dim))
        bn = _bn_modules(model)[0]
        before = bn.running_mean.clone()
        with batchnorm_eval(model):
            model(torch.randn(64, input_dim))
        return not torch.allclose(before, bn.running_mean)

    try:
        for env, expect_update in (({"CLFAIR_BN_MODE": "legacy"}, True),
                                   ({"CLFAIR_BN_MODE": "freeze"}, False),
                                   ({"CLFAIR_BN_MODE": "eval"}, False),
                                   ({}, False)):
            for k in tracked:
                os.environ.pop(k, None)
            os.environ.update(env)
            if _run_aux() != expect_update:
                failures.append(f"{env or '(default)'}: running stats updated="
                                f"{not expect_update}, expected {expect_update}")

        # An unrecognised mode must raise, not quietly pick a default. These are
        # different methods, so a typo would otherwise produce plausible numbers
        # under a protocol nobody chose.
        for k in tracked:
            os.environ.pop(k, None)
        os.environ["CLFAIR_BN_MODE"] = "frozen"   # plausible typo for "freeze"
        try:
            _run_aux()
            failures.append("CLFAIR_BN_MODE=frozen was accepted; expected SystemExit")
        except SystemExit:
            pass
    finally:
        for k, v in prior.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return failures


def check_methods_train(tasks, input_dim: int, n_train: int, n_test: int,
                        epochs: int) -> list[str]:
    """
    Train each method over TWO tasks, not one.

    The second task is the point. Every replay path — A-GEM's memory gradient,
    DER++'s distillation, FaIRL's replay term — is guarded by `if mem is not
    None`, so on task 0 the buffer is empty and none of that code executes. A
    one-task test would leave the replay branches uncovered.
    """
    def slice_task(t):
        return (t["X_train"][:n_train], t["y_train"][:n_train],
                t["s_train"][:n_train])

    X_test = tasks[0]["X_test"][:n_test]

    failures = []
    for name, factory, takes_s in METHODS:
        t0 = time.time()
        print(f"      {name}: train/predict smoke (2 tasks)...", flush=True)
        try:
            obj = factory(input_dim)
            for task in tasks[:2]:
                X, y, s = slice_task(task)
                if takes_s:
                    obj.train_task(X, y, s, epochs=epochs)
                else:
                    obj.train_task(X, y, epochs=epochs)
                if hasattr(obj, "remember"):
                    obj.remember(X, y, s)
            preds = obj.predict(X_test)
            if preds.shape != (n_test,):
                failures.append(f"{name}: predict returned {preds.shape}, "
                                f"expected ({n_test},)")
            else:
                print(f"      {name}: ok in {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 — report, don't mask
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def check_batchnorm_update_parity(task, input_dim: int, n_train: int) -> list[str]:
    """
    A-GEM and FAIR-GEM must update BatchNorm the same number of times per epoch.

    FAIR-GEM adds a safety-gradient forward on top of A-GEM's task and memory
    passes. Only the task pass may update running statistics.
    """
    X, y, s = (task["X_train"][:n_train], task["y_train"][:n_train],
               task["s_train"][:n_train])

    def updates(factory):
        obj = factory(input_dim)
        bn = _bn_modules(obj.model)[0]
        obj.remember(X, y, s)  # so the memory pass actually fires
        before = int(bn.num_batches_tracked.item())
        obj.train_task(X, y, s, epochs=1)
        return int(bn.num_batches_tracked.item()) - before

    # Look the factories up from METHODS rather than closing over names that
    # only exist inside load_deps().
    by_name = {name: factory for name, factory, _ in METHODS}
    counts = {name: updates(by_name[name])
              for name in ("agem", "fair_gem", "der_pp", "fair_derpp",
                           "derpp_penalty")}

    failures = []
    if counts["agem"] != counts["fair_gem"]:
        failures.append(
            f"BatchNorm updates differ: agem={counts['agem']} "
            f"fair_gem={counts['fair_gem']} — the safety pass is updating "
            f"running statistics")
    if counts["der_pp"] != counts["fair_derpp"]:
        failures.append(
            f"BatchNorm updates differ: der_pp={counts['der_pp']} "
            f"fair_derpp={counts['fair_derpp']} — the safety pass is updating "
            f"running statistics")
    if counts["der_pp"] != counts["derpp_penalty"]:
        failures.append(
            f"BatchNorm updates differ: der_pp={counts['der_pp']} "
            f"derpp_penalty={counts['derpp_penalty']}")
    print(f"      BatchNorm updates/epoch: " +
          "  ".join(f"{k}={v}" for k, v in counts.items()))
    return failures


def check_safety_controls(input_dim: int) -> list[str]:
    """Surrogate and projection controls must be explicit and fail closed."""
    failures = []
    losses = torch.tensor(
        [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 6.0, 6.0, 10.0],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    groups = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0])
    gap, n_labels = conditional_disparity(losses, labels, groups)
    if gap is None or n_labels != 2 or not torch.isclose(gap, torch.tensor(3.0)):
        failures.append("conditional disparity did not average estimable label cells")
    else:
        gap.backward()
        if not torch.isfinite(losses.grad).all() or losses.grad[-1] != 0:
            failures.append("conditional disparity gradient included an inestimable cell")

    by_name = {name: factory for name, factory, _ in METHODS}
    for name in ("fair_gem", "fair_derpp"):
        factory = by_name[name]
        try:
            obj = factory(input_dim, disparity_mode="conditional")
            if obj.disparity_mode != "conditional":
                failures.append(f"{name}: conditional mode was not retained")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: conditional mode failed: {exc}")
        try:
            factory(input_dim, disparity_mode="typo")
            failures.append(f"{name}: invalid disparity mode was accepted")
        except ValueError:
            pass

    try:
        by_name["fair_gem"](input_dim, proj_iters=0)
        failures.append("fair_gem: proj_iters=0 was accepted")
    except ValueError:
        pass
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick functional checks on Spurious Shift methods.",
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help="Torch intra-op threads for the smoke run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    load_deps()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(
        f"test_methods: n_train={args.n_train} n_test={args.n_test} "
        f"epochs={args.epochs} torch_threads={args.threads}\n",
        flush=True,
    )
    tasks, _ = get_spurious_tasks(seed=42)
    task = tasks[0]
    input_dim = task["X_train"].shape[1]

    ok = True
    for label, failures in (
        ("batchnorm_eval freezes stats and restores mode",
         check_batchnorm_eval_restores(input_dim)),
        ("normalization-study modes", check_norm_study_modes(input_dim)),
        ("CLFAIR_BN_MODE switches behaviour and rejects bad values",
         check_bn_mode_switch(input_dim)),
        ("all methods train and predict (2 tasks, exercises replay)",
         check_methods_train(tasks, input_dim, args.n_train, args.n_test, args.epochs)),
        ("BatchNorm update parity",
         check_batchnorm_update_parity(task, input_dim, args.n_train)),
        ("explicit safety controls", check_safety_controls(input_dim)),
    ):
        if failures:
            ok = False
            print(f"FAIL  {label}")
            for f in failures:
                print(f"      {f}")
        else:
            print(f"ok    {label}")

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
