# shared utilities: seeding, metrics, checkpointing, logging
import os
import json
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def set_seed(seed):
    """deterministic seed for all rng sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(targets, preds):
    """compute accuracy, macro f1, precision, recall from flat lists."""
    acc  = accuracy_score(targets, preds)
    f1   = f1_score(targets, preds, average="macro", zero_division=0)
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    rec  = recall_score(targets, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}


def save_checkpoint(state, path):
    """save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    """load checkpoint and return epoch + best metric."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


def save_epoch_log(log_path, records):
    """write per-epoch metrics list to json."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(records, f, indent=2)