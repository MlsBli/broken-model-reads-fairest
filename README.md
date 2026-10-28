# The Broken Model Reads Fairest

Code and results for *The Broken Model Reads Fairest: Head Collapse Conceals
Demographic Disparity in Continually Fine-Tuned Encoders* (Miles Bliey and Keira
Chatwin, CL4FMAgents Workshop at NeurIPS 2026). The paper is
[`paper.pdf`](paper.pdf).

In class-incremental learning, a classifier head that collapses onto the most
recent classes drives prediction-level fairness metrics toward zero, so a
standard fairness table ranks the broken models as the fairest. Refitting a
linear head on the frozen encoder restores accuracy and the disparity together.
This repository contains the methods, the two experiment suites, every released
result (ten seeds), and the scripts that produce the paper's tables and figures.

## Repository layout

```
bias_in_bios/       Bias in Bios suite: BERT-base encoder, 25 professions in 5 tasks (GPU)
  run.sh            train, evaluate every readout, aggregate and plot
  run_campaign.sh   the full ten-seed campaign plus the frozen-encoder control
  results/          per-seed arrays (runs/seed<N>/), aggregates, summaries, figures
spurious_shift/     Spurious Shift suite: synthetic six-task sequence (CPU)
  run.sh            train, evaluate, run ablations and sweeps, aggregate and plot
  results/
analysis/           post-hoc analyses over both suites; outputs in analysis/results/
tests/              functional tests of the methods
paper.pdf
```

Both suites implement the same nine methods: joint retraining, fine-tuning,
A-GEM, DER++, GroupDRO, a FaIRL-inspired adversarial baseline, FSW, and our two
disparity-constrained methods, **FAIR-GEM** (`fairgem.py`) and **Fair-DER++**
(`fairderpp.py`), which share the group-loss disparity surrogate in `safety.py`.
Hyperparameters are fixed in each suite's `config.py`, and the validation search
that selected them is recorded in `results/tuning_summary.csv`.

## Setup

Python 3.11 with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

If the default PyTorch build does not match your CUDA driver, install the
matching wheel as the comment in `requirements.txt` describes.

**Data.** Spurious Shift is generated procedurally; nothing to download. Bias in
Bios is read from `BIOS_DATA_PATH` (default `bias_in_bios/data/bios`) and is
downloaded from the Hugging Face dataset `LabHC/bias_in_bios` on first use:

```bash
export BIOS_DATA_PATH=/path/to/bios
```

## Reproducing the paper's tables and figures

Every number in the paper can be recomputed from the tracked results without a
GPU. The main tables are the summary files:

| Paper | File |
| --- | --- |
| Table 1 (Bias in Bios) | `bias_in_bios/results/results_test_summary.csv` (deployed head), `results_retune_summary.csv` (head refit on retained memory) |
| Table 2 (Spurious Shift) | `spurious_shift/results/results_summary.csv` |
| Refit on a shared held-out pool | `bias_in_bios/results/results_probe_shared_summary.csv` |
| Repair-budget sweep | `bias_in_bios/results/ttbudget_summary.csv` |
| Frozen-encoder control | `bias_in_bios/results/frozen_encoder/` |

The figures and post-hoc analyses are regenerated from the repository root:

```bash
python analysis/plot_paper_figures.py        # Figures 1 and 2
python analysis/plot_control_figure.py       # Figure 3
python analysis/posthoc_studies.py           # permutation tests with Holm correction, readout concordance
python analysis/surrogate_alignment.py --results-dir spurious_shift/results
python analysis/surrogate_alignment.py --results-dir bias_in_bios/results
python analysis/surrogate_alignment.py --results-dir bias_in_bios/results \
    --methods joint_retrain,derpp,fairl,fsw,fair_derpp --suffix _nondegenerate
python analysis/diagnostics.py               # projection statistics and representation probes
python analysis/sensitivity_summary.py       # projection-cycle, buffer-size and plain-SGD controls
python analysis/runtime_environment.py       # runtime overhead across environments
```

The Bias in Bios surrogate analysis runs twice: over every method, and over the
five methods whose deployed heads do not collapse.

The appendix figures for each suite are produced by its plotting scripts, run
from the suite's `results/` directory:

```bash
(cd bias_in_bios/results && python ../plot_results.py --test && python ../plot_ncm.py)
(cd spurious_shift/results && python ../plot_results.py && python ../plot_ablations.py \
    && python ../plot_norm_study.py)
```

Four analyses retrain or re-encode data and therefore take longer:
`analysis/norm_study.py` (normalization study), `analysis/fsw_reference.py` (FSW
at its published hyperparameters), and `analysis/measure_truncation.py` and
`analysis/probe_token_budget.py` (the cost of the 32-token input budget).

## Running the experiments

**Spurious Shift** runs on CPU. All ten seeds, with ablations and the
fairness-strength sweep:

```bash
bash spurious_shift/run.sh
```

A single seed without the ablations and sweep takes about 15 minutes on a
laptop CPU:

```bash
SEEDS="0" SKIP_ABLATIONS=1 SKIP_TRADEOFF=1 bash spurious_shift/run.sh
```

**Bias in Bios** needs one CUDA GPU with at least 24 GB of memory and at least
80 GB of disk (each method saves its own BERT-base, about 4 GB per seed). Ten
seeds take about 6–7 hours. Check the environment first, then run the campaign:

```bash
python bias_in_bios/preflight.py --stage final
BIOS_DATA_PATH=/path/to/bios bash bias_in_bios/run_campaign.sh
```

`run_campaign.sh` runs seed 0 first and stops if it did not produce every
expected artifact, then runs the remaining seeds, aggregates, and repeats the
sequence with the encoder frozen. To run a chosen set of seeds directly:

```bash
SEEDS="0 1 2" bash bias_in_bios/run.sh
```

Training resumes from per-seed checkpoints unless `FORCE_FRESH=1`. A resumed
run is not bitwise identical to an uninterrupted one, so avoid pre-emptible
machines when reproducing the released numbers. Hyperparameter search
(`python ../tune.py --fresh` from a suite's `results/` directory) uses
validation data only and is needed only to start a new search.

## Tests

```bash
python tests/test_methods.py
```

## Notes

The suites store their fairness metrics under different keys in
`results_agg.npz`. In Spurious Shift, `eo` holds the accuracy-parity gap and
`eo_true` the equalized-odds gap; in Bias in Bios, `eo` holds the equalized-odds
gap and `acc_parity` the accuracy-parity gap. The summary CSV columns
(`AccParity`, `EO`) are unambiguous in both. The definitions also differ by
design (binary max-gaps on Spurious Shift, RMS one-vs-rest gaps over 25 classes
on Bias in Bios), so values are not comparable across suites.

## Citation

```bibtex
@inproceedings{bliey2026broken,
  title     = {The Broken Model Reads Fairest: Head Collapse Conceals Demographic
               Disparity in Continually Fine-Tuned Encoders},
  author    = {Bliey, Miles and Chatwin, Keira},
  booktitle = {Continual Learning in the Era of Foundation Models and Embodied
               Agents Workshop at NeurIPS},
  year      = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
