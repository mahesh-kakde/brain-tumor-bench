# training loop — shared across all four models
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils import set_seed, compute_metrics, save_checkpoint, save_epoch_log


def run_epoch(model, loader, criterion, optimizer, device, training):
    """run one train or eval epoch. returns (loss, metrics dict)."""
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False, desc="train" if training else "val"):
            imgs, labels = imgs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


def train(model_name, model, train_loader, val_loader,
          ckpt_path, log_path,
          seed=42, max_epochs=30, patience=7, device=None):
    """
    full training loop with:
      - adamw optimizer (lr=1e-4, weight_decay=0.01)
      - cosine annealing lr scheduler (t_max=30)
      - cross-entropy loss with label_smoothing=0.05
      - early stopping on val macro f1 (patience=7)
      - saves best checkpoint by val macro f1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1        = 0.0
    best_epoch     = 0
    patience_count = 0
    epoch_records  = []

    print(f"\n[{model_name}] seed={seed}  device={device}")
    print(f"  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")
    print("-" * 60)

    for epoch in range(1, max_epochs + 1):
        # train
        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        # validate
        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        # step scheduler every epoch regardless of early stopping
        scheduler.step()

        val_f1 = val_m["macro_f1"]

        # log
        record = {
            "epoch":          epoch,
            "train_loss":     round(train_loss, 6),
            "train_f1":       round(train_m["macro_f1"], 6),
            "train_accuracy": round(train_m["accuracy"], 6),
            "val_loss":       round(val_loss, 6),
            "val_f1":         round(val_f1, 6),
            "val_accuracy":   round(val_m["accuracy"], 6),
            "lr":             round(scheduler.get_last_lr()[0], 8),
        }
        epoch_records.append(record)

        print(
            f"  epoch {epoch:02d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_m['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # save best checkpoint
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            patience_count = 0
            save_checkpoint({
                "model_name":    model_name,
                "seed":          seed,
                "epoch":         epoch,
                "best_f1":       best_f1,
                "model_state":   model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"    *** new best — saved checkpoint (f1={best_f1:.4f}) ***")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  early stopping at epoch {epoch} (patience={patience})")
                print(f"  best epoch: {best_epoch}  best val f1: {best_f1:.4f}")
                break

    save_epoch_log(log_path, epoch_records)
    print(f"\n  training done. log saved to {log_path}")
    return best_f1, best_epoch, epoch_records