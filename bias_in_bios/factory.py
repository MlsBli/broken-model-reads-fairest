"""
factory.py — Dataset dispatch for train.py.

Reads config.DATASET and returns the appropriate task list.
Supported: "spurious_shift", "bios".
"""

import config as cfg


def get_tasks(config=cfg, need_frozen_embeds=None, seed=None):
    """Load tasks for the configured dataset.

    Returns (tasks, scaler). scaler is a fitted StandardScaler for tabular
    data, or None for BIOS.

    need_frozen_embeds (bios only): if False, skips the frozen BERT embedding
    pass (used only by sklearn baselines). Defaults to not BIOS_BERT_FINETUNE.

    seed: overrides config.SEED for data construction, so a --seed sweep varies
    the train/validation split rather than reusing one split across every seed.
    """
    if config.DATASET == "spurious_shift":
        from datasets.spurious_shift import get_spurious_tasks
        tasks, scaler = get_spurious_tasks(
            seed=config.SPURIOUS_SEED if seed is None else seed)
        return tasks, scaler
    elif config.DATASET == "bios":
        from datasets.bios_tasks import get_bios_tasks
        if need_frozen_embeds is None:
            need_frozen_embeds = not getattr(config, "BIOS_BERT_FINETUNE", True)
        tasks, scaler = get_bios_tasks(
            data_path=config.BIOS_DATA_PATH,
            step_size=config.BIOS_STEP_SIZE,
            seed=config.SEED if seed is None else seed,
            bert_batch_size=config.BIOS_BERT_BATCH,
            bert_max_length=config.BIOS_BERT_MAXLEN,
            compute_frozen_embeds=need_frozen_embeds,
        )
        return tasks, scaler
    else:
        raise ValueError(
            f"Unknown DATASET={config.DATASET!r}. "
            "Expected 'spurious_shift' or 'bios'."
        )


def is_bert_task(task):
    """True iff the task uses BERT token IDs (BIOS fine-tuning mode)."""
    return "token_ids_train" in task and getattr(cfg, "BIOS_BERT_FINETUNE", True)
