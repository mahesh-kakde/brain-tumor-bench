# evaluation script — kaggle test set + figshare sensitivity check
# computes: accuracy, macro f1, precision, recall, auroc, ece, confusion matrix, per-class f1
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
from PIL import Image
from torchvision import transforms, models

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── dataset paths ────────────────────────────────────────────────────────────
PROJECT    = os.path.dirname(os.path.dirname(__file__))
KAGGLE_TEST_DIR  = os.path.join(PROJECT, 'data', 'kaggle', 'Testing')
FIGSHARE_DIR     = os.path.join(PROJECT, 'data', 'figshare', 'processed')
RUNS_DIR         = os.path.join(PROJECT, 'results', 'kaggle_runs')
OUT_DIR          = os.path.join(PROJECT, 'results', 'test_eval')

KAGGLE_CLASSES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
FIGSHARE_CLASSES = ['glioma', 'meningioma', 'pituitary']   # no "notumor" class

# imagenet normalisation — same as training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── datasets ─────────────────────────────────────────────────────────────────
class ImageFolderFlat:
    """loads images from class subfolders. returns (img_tensor, label_int)."""
    def __init__(self, root, class_names, transform):
        self.transform = transform
        self.class_names = class_names
        self.cls_to_idx  = {c: i for i, c in enumerate(class_names)}
        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, self.cls_to_idx[cls]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


# ─── model builders (identical to training) ───────────────────────────────────
def build_resnet50(num_classes=4):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(2048, num_classes))
    return m

def build_efficientnet_b0(num_classes=4):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))
    return m

def build_vit_b16(num_classes=4):
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(768, num_classes))
    return m

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        bb = models.resnet50(weights=None)
        self.cnn = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4
        )
        self.proj      = nn.Linear(2048, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, 512) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(512, num_classes))

    def forward(self, x):
        b = x.size(0)
        feat   = self.cnn(x).flatten(2).transpose(1, 2)
        tokens = self.proj(feat) + self.pos_embed
        cls    = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.head(self.transformer(tokens)[:, 0])

BUILDERS = {
    'resnet50':        build_resnet50,
    'efficientnet_b0': build_efficientnet_b0,
    'vit_b16':         build_vit_b16,
    'hybrid':          HybridCNNViT,
}


# ─── metrics ──────────────────────────────────────────────────────────────────
def expected_calibration_error(probs, labels, n_bins=15):
    """computes ece: average gap between confidence and accuracy per bin."""
    confs  = probs.max(axis=1)
    preds  = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece / len(labels))


def evaluate_model(model, loader, class_names, device):
    """run inference and return all metrics."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_probs   = np.array(all_probs)
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    n_classes = len(class_names)

    # auroc — one-vs-rest, macro averaged
    try:
        auroc = roc_auc_score(
            all_targets, all_probs,
            multi_class='ovr', average='macro'
        )
    except Exception:
        auroc = None

    ece = expected_calibration_error(all_probs, all_targets)

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()

    return {
        'accuracy':        float(accuracy_score(all_targets, all_preds)),
        'macro_f1':        float(f1_score(all_targets, all_preds, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(all_targets, all_preds, average='macro', zero_division=0)),
        'macro_recall':    float(recall_score(all_targets, all_preds, average='macro', zero_division=0)),
        'auroc':           float(auroc) if auroc is not None else None,
        'ece':             float(ece),
        'per_class_f1':    {class_names[i]: round(per_class_f1[i], 4) for i in range(n_classes)},
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
        'n_samples':        int(len(all_targets)),
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_list = ['resnet50', 'efficientnet_b0', 'vit_b16', 'hybrid']
    seeds   = [42, 1337, 2026]
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    for model_name in models_list:
        print(f'\n{"="*60}')
        print(f'model: {model_name}')
        print(f'{"="*60}')
        all_results[model_name] = {}

        for seed in seeds:
            ckpt_path = os.path.join(RUNS_DIR, model_name, f'seed_{seed}', 'best.pt')
            print(f'\n  seed={seed}  loading checkpoint...')

            # load checkpoint
            ckpt  = torch.load(ckpt_path, map_location='cpu')
            model = BUILDERS[model_name]()
            model.load_state_dict(ckpt['model_state'])
            model = model.to(device)
            print(f'  checkpoint loaded (trained epoch {ckpt["epoch"]}, val_f1={ckpt["best_f1"]:.4f})')

            seed_results = {}

            # ── kaggle test set ────────────────────────────────────────────
            print(f'  evaluating on kaggle test set...')
            test_ds     = ImageFolderFlat(KAGGLE_TEST_DIR, KAGGLE_CLASSES, VAL_TF)
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
            kaggle_m    = evaluate_model(model, test_loader, KAGGLE_CLASSES, device)
            seed_results['kaggle_test'] = kaggle_m
            print(f'    acc={kaggle_m["accuracy"]:.4f}  f1={kaggle_m["macro_f1"]:.4f}  '
                  f'auroc={kaggle_m["auroc"]:.4f}  ece={kaggle_m["ece"]:.4f}')

            # ── figshare sensitivity check ─────────────────────────────────
            # model outputs 4 logits; for figshare (3 classes: glioma, meningioma, pituitary)
            # we use only the logits at indices matching those class positions in kaggle classes
            # kaggle indices: glioma=0, meningioma=1, notumor=2, pituitary=3
            # figshare classes: glioma=0, meningioma=1, pituitary=2
            # we select kaggle logit columns [0, 1, 3] and re-normalise
            print(f'  evaluating on figshare (sensitivity check)...')
            fig_ds     = ImageFolderFlat(FIGSHARE_DIR, FIGSHARE_CLASSES, VAL_TF)
            fig_loader = DataLoader(fig_ds, batch_size=16, shuffle=False, num_workers=0)

            # wrap model to project 4-class output → 3-class for figshare
            class FigshareWrapper(nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    # columns to keep: glioma=0, meningioma=1, pituitary=3
                    self.keep = [0, 1, 3]
                def forward(self, x):
                    return self.base(x)[:, self.keep]

            fig_model = FigshareWrapper(model).to(device)
            fig_m     = evaluate_model(fig_model, fig_loader, FIGSHARE_CLASSES, device)
            seed_results['figshare_sensitivity'] = fig_m
            print(f'    acc={fig_m["accuracy"]:.4f}  f1={fig_m["macro_f1"]:.4f}  '
                  f'auroc={fig_m["auroc"]:.4f}  ece={fig_m["ece"]:.4f}')

            all_results[model_name][f'seed_{seed}'] = seed_results

            # save per-run results
            run_out = os.path.join(OUT_DIR, model_name, f'seed_{seed}')
            os.makedirs(run_out, exist_ok=True)
            with open(os.path.join(run_out, 'eval.json'), 'w') as f:
                json.dump(seed_results, f, indent=2)

    # ── aggregate across seeds ─────────────────────────────────────────────────
    print(f'\n\n{"="*60}')
    print('FINAL RESULTS — KAGGLE TEST SET (mean ± std across 3 seeds)')
    print(f'{"="*60}')

    summary = {}
    for model_name in models_list:
        metrics_list = {k: [] for k in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'auroc', 'ece']}
        for seed in seeds:
            m = all_results[model_name][f'seed_{seed}']['kaggle_test']
            for k in metrics_list:
                if m[k] is not None:
                    metrics_list[k].append(m[k])

        agg = {}
        for k, vals in metrics_list.items():
            if vals:
                agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

        summary[model_name] = agg
        print(f'\n  {model_name}:')
        for k, v in agg.items():
            print(f'    {k:20s}: {v["mean"]:.4f} ± {v["std"]:.4f}')

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump({'per_run': all_results, 'aggregated': summary}, f, indent=2)

    print(f'\n\nresults saved to {OUT_DIR}')


if __name__ == '__main__':
    main()