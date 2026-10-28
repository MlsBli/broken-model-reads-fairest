"""
bios_tasks.py — Bias-in-Bios dataset loader for continual learning.

Source: De-Arteaga et al. (2019) "Bias in Bios: A Case Study of Semantic
Representation Bias in a High-Stakes Setting."

Dataset: Biography text with 28 raw profession categories. The released
protocol uses the 25 most frequent professions (multi-class classification).
Sensitive attribute: binary gender (0=male, 1=female).

Data source priority (first found wins):
  1. Cached .npy files in data_path/ — loaded instantly on subsequent runs.
  2. train.pkl / test.pkl in data_path/ — provided manually.
  3. HuggingFace Hub: LabHC/bias_in_bios — auto-downloaded, no manual steps.

Task structure (step_size=5 → 5 tasks):
  Task 0: professions 0-4   (5 most frequent by training count)
  Task 1: professions 5-9   ...   Task 4: professions 20-24

Output per task dict:
  token_ids_train/val/test : int64 [n, 32]  — for BERT fine-tuning at train time
  X_train/val/test         : float32 [n, 768] — frozen BERT embeds for sklearn
  y_train/val/test         : int64
  s_train/val/test         : int64  (gender: 0=male, 1=female)
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Raw data loading
def _load_pkl(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _unpack_pkl(raw):
    """
    Unpack raw pkl records into (texts, labels, genders).

    Supports two formats:
      - Dict  (original De-Arteaga et al. format): keys 'hard_text', 'p', 'g'
      - Tuple (4-tuple): (_, text, profession, gender)

    If the format is unrecognised a descriptive ValueError is raised.
    """
    texts, ys, gs = [], [], []
    for item in raw:
        if isinstance(item, dict):
            txt = item["hard_text"]
            y   = item["p"]
            g   = item["g"]
        elif hasattr(item, "__len__") and len(item) == 4:
            _, txt, y, g = item
        else:
            raise ValueError(
                f"_unpack_pkl: unexpected record type {type(item)}. "
                "Expected a dict with keys 'hard_text'/'p'/'g' (De-Arteaga et al.) "
                "or a 4-tuple (_, text, profession, gender)."
            )
        texts.append(txt)
        ys.append(int(y))
        gs.append(int(g))
    return texts, np.array(ys, dtype=np.int64), np.array(gs, dtype=np.int64)


def _load_from_huggingface(data_path: str):
    """Download LabHC/bias_in_bios via a subprocess (avoids a PicklingError in notebook environments).

    Saves parquet files to data_path/ and skips download on subsequent calls.
    """
    import subprocess, sys, tempfile

    train_pq = os.path.join(data_path, "_hf_train.parquet")
    test_pq  = os.path.join(data_path, "_hf_test.parquet")

    if not (os.path.exists(train_pq) and os.path.exists(test_pq)):
        print("  Downloading LabHC/bias_in_bios from HuggingFace Hub (subprocess)...")

        _DOWNLOAD_SCRIPT = r"""
import sys, os
import huggingface_hub, pandas as pd

data_path = sys.argv[1]
os.makedirs(data_path, exist_ok=True)

api = huggingface_hub.HfApi()
all_files = list(api.list_repo_files("LabHC/bias_in_bios", repo_type="dataset"))
parquet = [f for f in all_files if f.endswith(".parquet")]

def pick(keyword):
    hits = sorted(f for f in parquet if f"/{keyword}" in f)
    return hits or sorted(f for f in parquet if keyword in f)

train_files = pick("train")
_test_files_exact = pick("test")
_val_files  = pick("validation") or pick("dev")
test_files  = _test_files_exact or _val_files
if not _test_files_exact and _val_files:
    print(
        "WARNING: no 'test' split found in LabHC/bias_in_bios — falling back to "
        "'validation' split as test set.  If this validation split was used during "
        "hyperparameter tuning, test metrics will be contaminated.",
        file=sys.stderr,
        flush=True,
    )
if not train_files or not test_files:
    print(f"ERROR: could not find train/test parquet in {parquet}", file=sys.stderr)
    sys.exit(1)

def load_split(files):
    dfs = []
    for fname in files:
        local = huggingface_hub.hf_hub_download(
            "LabHC/bias_in_bios", fname, repo_type="dataset"
        )
        dfs.append(pd.read_parquet(local))
    return pd.concat(dfs, ignore_index=True)

train_df = load_split(train_files)
test_df  = load_split(test_files)
train_df.to_parquet(os.path.join(data_path, "_hf_train.parquet"), index=False)
test_df.to_parquet( os.path.join(data_path, "_hf_test.parquet"),  index=False)
print(f"Downloaded: {len(train_df):,} train  {len(test_df):,} test", flush=True)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(_DOWNLOAD_SCRIPT)
            script_path = f.name

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path, data_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                print(" ", line, end="", flush=True)
            proc.wait()
        finally:
            os.unlink(script_path)

        if proc.returncode != 0:
            raise RuntimeError(
                "HuggingFace download subprocess failed.\n"
                "Place train.pkl and test.pkl manually in data_path as a fallback."
            )

    import pandas as pd
    train_df = pd.read_parquet(train_pq)
    test_df  = pd.read_parquet(test_pq)

    tr_texts = train_df["hard_text"].tolist()
    tr_y     = train_df["profession"].to_numpy(dtype=np.int64)
    tr_g     = train_df["gender"].to_numpy(dtype=np.int64)
    te_texts = test_df["hard_text"].tolist()
    te_y     = test_df["profession"].to_numpy(dtype=np.int64)
    te_g     = test_df["gender"].to_numpy(dtype=np.int64)
    print(f"  Loaded: {len(tr_texts):,} train  {len(te_texts):,} test")
    return tr_texts, tr_y, tr_g, te_texts, te_y, te_g


def _load_raw_data(data_path):
    """
    Load raw text + labels from the best available source.

    Priority: pkl files (manual) → HuggingFace Hub (automatic).
    """
    train_pkl = os.path.join(data_path, "train.pkl")
    test_pkl  = os.path.join(data_path, "test.pkl")

    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print("  Loading from pkl files...")
        train_raw = _load_pkl(train_pkl)
        test_raw  = _load_pkl(test_pkl)
        tr_texts, tr_y, tr_g = _unpack_pkl(train_raw)
        te_texts, te_y, te_g = _unpack_pkl(test_raw)
        print(f"  Loaded: {len(tr_texts):,} train  {len(te_texts):,} test")
        return tr_texts, tr_y, tr_g, te_texts, te_y, te_g

    print("  pkl files not found - downloading from HuggingFace Hub...")
    return _load_from_huggingface(data_path)


# Tokenisation and embedding
def _tokenize(texts, cache_path, max_length=32):
    """
    Tokenize with BertTokenizerFast and cache to disk.

    Matches FaIRL bios_loader.py:
      tokenizer.encode(txt, max_length=32, padding="max_length",
                       truncation=True, add_special_tokens=True)
    Returns int64 array [n, max_length].
    """
    if os.path.exists(cache_path):
        print(f"    Loading cached token IDs from {os.path.basename(cache_path)}")
        return np.load(cache_path)

    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print(f"    Tokenizing {len(texts):,} texts (max_len={max_length}) ...")
    ids = np.array(
        [tokenizer.encode(txt, max_length=max_length, padding="max_length",
                          truncation=True, add_special_tokens=True)
         for txt in tqdm(texts)],
        dtype=np.int64,
    )
    np.save(cache_path, ids)
    print(f"    Cached to {cache_path}")
    return ids


def _compute_bert_pooled(token_ids, cache_path, batch_size=64):
    """Compute frozen BERT pooled-output embeddings and cache to disk.

    Used only by sklearn baselines; neural methods use token IDs directly.
    """
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(token_ids):
            print(f"    Loading cached embeddings from {os.path.basename(cache_path)}")
            return cached
        print(f"    Cached embeddings ({len(cached):,} rows) do not match "
              f"{len(token_ids):,} requested rows - recomputing.")

    import torch
    from transformers import BertModel

    print(f"    Computing frozen BERT embeddings for {len(token_ids):,} samples ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    all_embs = []
    ids_t = torch.tensor(token_ids, dtype=torch.long)
    for i in tqdm(range(0, len(ids_t), batch_size)):
        batch = ids_t[i: i + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
        all_embs.append(out.pooler_output.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.save(cache_path, embeddings)
    print(f"    Cached to {cache_path}")
    return embeddings


# Main entry point
_BERT_POOLED_DIM = 768   # bert-base pooled-output width (frozen-embedding dim)


def get_bios_tasks(
    data_path: str,
    step_size: int = 5,
    val_frac: float = 0.15,
    seed: int = 42,
    bert_batch_size: int = 64,
    bert_max_length: int = 32,
    compute_frozen_embeds: bool = True,
) -> tuple[list[dict], None]:
    """
    Load the Bias-in-Bios dataset and return CL tasks.

    Data is sourced from pkl files if present, otherwise auto-downloaded from
    HuggingFace Hub (LabHC/bias_in_bios). Tokenisation and frozen BERT
    embeddings are cached to data_path/ on first run.

    Returns
    -------
    tasks : list[dict]  (5 tasks for default step_size=5)
        token_ids_train/val/test : int64 [n, 32]   — BERT fine-tuning input
        X_train/val/test         : float32 [n, 768] — frozen embeds for sklearn
        y_train/val/test, s_train/val/test : int64
        state                    : str
    scaler : None
    """
    os.makedirs(data_path, exist_ok=True)

    # Cache paths
    train_ids_cache = os.path.join(data_path, "train_token_ids.npy")
    test_ids_cache  = os.path.join(data_path, "test_token_ids.npy")
    train_y_cache   = os.path.join(data_path, "train_labels.npy")
    test_y_cache    = os.path.join(data_path, "test_labels.npy")
    train_g_cache   = os.path.join(data_path, "train_gender.npy")
    test_g_cache    = os.path.join(data_path, "test_gender.npy")
    train_emb_cache = os.path.join(data_path, "train_bert_pooled.npy")
    test_emb_cache  = os.path.join(data_path, "test_bert_pooled.npy")

    all_token_caches = [train_ids_cache, test_ids_cache,
                        train_y_cache, test_y_cache,
                        train_g_cache, test_g_cache]

    if all(os.path.exists(p) for p in all_token_caches):
        # All caches present — load without touching raw data at all
        print("Loading Bios from cache (no raw data needed)...")
        train_ids = np.load(train_ids_cache)
        test_ids  = np.load(test_ids_cache)
        # The cache filename does not encode the tokenizer settings, so a
        # changed BIOS_BERT_MAXLEN would silently reuse stale tokens.
        if train_ids.shape[1] != bert_max_length:
            raise SystemExit(
                f"Token cache {train_ids_cache} has width {train_ids.shape[1]} "
                f"but bert_max_length={bert_max_length}. Delete the cached "
                "*_token_ids.npy files to re-tokenise.")
        train_y   = np.load(train_y_cache)
        test_y    = np.load(test_y_cache)
        train_g   = np.load(train_g_cache)
        test_g    = np.load(test_g_cache)
    else:
        # Load raw texts from pkl or HuggingFace
        print("Loading Bios raw data...")
        tr_texts, train_y_raw, train_g, te_texts, test_y_raw, test_g = \
            _load_raw_data(data_path)

        # Re-order labels by training-set frequency (matches FaIRL)
        unique_labels, counts = np.unique(train_y_raw, return_counts=True)
        order = unique_labels[np.argsort(counts)[::-1]]
        o2id  = {orig: new for new, orig in enumerate(order)}
        train_y = np.array([o2id[y] for y in train_y_raw], dtype=np.int64)

        # Guard: test split must not contain profession labels absent from training.
        unknown_labels = set(test_y_raw.tolist()) - set(o2id.keys())
        if unknown_labels:
            raise ValueError(
                f"Test split contains profession label(s) not present in training: "
                f"{unknown_labels}.  This usually means the HuggingFace dataset split "
                "has changed — re-download or filter the test set to known labels."
            )
        test_y  = np.array([o2id[y] for y in test_y_raw],  dtype=np.int64)

        # Tokenise
        print("Tokenising...")
        train_ids = _tokenize(tr_texts, train_ids_cache, max_length=bert_max_length)
        test_ids  = _tokenize(te_texts, test_ids_cache,  max_length=bert_max_length)

        # Cache labels and gender alongside token IDs
        np.save(train_y_cache, train_y)
        np.save(test_y_cache,  test_y)
        np.save(train_g_cache, train_g)
        np.save(test_g_cache,  test_g)

    # Keep only the N_CLASSES most frequent professions (labels are already
    # frequency-ordered, 0 = most frequent). With all 28, a sixth block would
    # hold just the 3 rarest professions — ~1.8k train rows, ~14 optimizer steps
    # at batch 256 x 2 epochs — which no method can learn. 25 professions give
    # 5 blocks of 5, each with a learnable budget.
    from config import N_CLASSES
    if int(train_y.max()) + 1 > N_CLASSES:
        keep_tr, keep_te = train_y < N_CLASSES, test_y < N_CLASSES
        train_ids, train_y, train_g = train_ids[keep_tr], train_y[keep_tr], train_g[keep_tr]
        test_ids,  test_y,  test_g  = test_ids[keep_te],  test_y[keep_te],  test_g[keep_te]
        print(f"Filtered to the {N_CLASSES} most frequent professions "
              f"({keep_tr.sum():,} train / {keep_te.sum():,} test rows kept).")

    n_classes = int(train_y.max()) + 1
    assert n_classes == N_CLASSES, (
        f"Dataset has {n_classes} professions but config.N_CLASSES={N_CLASSES}. "
        "Update N_CLASSES in config.py to match the dataset."
    )

    # Frozen BERT embeddings (sklearn baselines only)
    # The BERT fine-tuning pipeline (train.py/tune.py) reads token_ids and only
    # needs the [n, 768] shape of these arrays. When compute_frozen_embeds=False
    # the (expensive, full-dataset) BERT forward pass is skipped and zero
    # placeholders of the correct shape are substituted, so every `.shape[1]` /
    # `len()` caller still works.
    if compute_frozen_embeds:
        print("Getting frozen BERT embeddings for sklearn baselines...")
        train_X = _compute_bert_pooled(train_ids, train_emb_cache, batch_size=bert_batch_size)
        test_X  = _compute_bert_pooled(test_ids,  test_emb_cache,  batch_size=bert_batch_size)
    else:
        print("Skipping frozen BERT embeddings (BERT fine-tune mode - unused by neural methods).")
        train_X = np.zeros((len(train_ids), _BERT_POOLED_DIM), dtype=np.float32)
        test_X  = np.zeros((len(test_ids),  _BERT_POOLED_DIM), dtype=np.float32)

    # Build CL tasks
    n_tasks = (n_classes + step_size - 1) // step_size
    tasks   = []

    print(f"\nBuilding {n_tasks} tasks ({step_size} professions/task, {n_classes} total):")
    for step in range(n_tasks):
        lo = step * step_size
        hi = min(n_classes, (step + 1) * step_size)
        task_labels = list(range(lo, hi))
        task_name   = f"task_{step}__professions={lo}-{hi-1}"

        tr_mask = np.isin(train_y, task_labels)
        ids_all = train_ids[tr_mask]
        X_all   = train_X[tr_mask]
        y_all   = train_y[tr_mask]
        s_all   = train_g[tr_mask]

        split_idx = np.arange(len(y_all))
        # Stratify on (profession × gender) so rare group combos are proportionally represented.
        strat_labels = np.array([f"{y}_{s}" for y, s in zip(y_all, s_all)])
        tr_idx, val_idx = train_test_split(
            split_idx, test_size=val_frac, random_state=seed, stratify=strat_labels,
        )

        te_mask = np.isin(test_y, task_labels)
        ids_te  = test_ids[te_mask]
        X_te    = test_X[te_mask]
        y_te    = test_y[te_mask]
        s_te    = test_g[te_mask]

        male_frac = (s_all[tr_idx] == 0).mean()
        print(f"  Task {step} (professions {lo}-{hi-1}): "
              f"{len(tr_idx):,} train  {len(val_idx):,} val  {len(y_te):,} test | "
              f"M/F {male_frac:.2f}/{1-male_frac:.2f}")

        tasks.append({
            "state":            task_name,
            "token_ids_train":  ids_all[tr_idx].astype(np.int64),
            "token_ids_val":    ids_all[val_idx].astype(np.int64),
            "token_ids_test":   ids_te.astype(np.int64),
            "X_train":          X_all[tr_idx].astype(np.float32),
            "X_val":            X_all[val_idx].astype(np.float32),
            "X_test":           X_te.astype(np.float32),
            "y_train":          y_all[tr_idx].astype(np.int64),
            "y_val":            y_all[val_idx].astype(np.int64),
            "y_test":           y_te.astype(np.int64),
            "s_train":          s_all[tr_idx].astype(np.int64),
            "s_val":            s_all[val_idx].astype(np.int64),
            "s_test":           s_te.astype(np.int64),
        })

    return tasks, None
