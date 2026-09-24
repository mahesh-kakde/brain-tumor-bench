# chunk 19 — run full xai analysis
# for each model's best seed: 10 images/class (40 total)
# generates overlays, del-auc, stab — saves to results/xai/
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.explainability import (
    get_saliency_map, overlay_heatmap, deletion_auc, perturbation_stability
)

# ─── config ───────────────────────────────────────────────────────────────────
KAGGLE_TEST  = os.path.join(PROJECT, 'data', 'kaggle', 'Testing')
RUNS_DIR     = os.path.join(PROJECT, 'results', 'kaggle_runs')
XAI_DIR      = os.path.join(PROJECT, 'results', 'xai')
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_PER_CLASS  = 10    # images per class per model
DEL_STEPS    = 10    # del-auc steps (blueprint spec)
STAB_TRIALS  = 5     # stab trials (σ=0.02)

# best checkpoint per model (by test macro f1)
BEST_SEEDS = {
    'resnet50':        1337,
    'efficientnet_b0': 2026,
    'vit_b16':         1337,
    'hybrid':          42,
}

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device('cpu')


# ─── model builders ───────────────────────────────────────────────────────────
def load_model(model_name, seed):
    ckpt_path = os.path.join(RUNS_DIR, model_name, f'seed_{seed}', 'best.pt')
    if model_name == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 4))
    elif model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 4))
    elif model_name == 'vit_b16':
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 4))
    elif model_name == 'hybrid':
        from src.models import HybridCNNViT
        m = HybridCNNViT()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m


# ─── image loader ─────────────────────────────────────────────────────────────
def sample_test_images(n_per_class=10):
    """returns list of (img_path, class_name, class_idx) sampled from test set."""
    cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(KAGGLE_TEST, cls)
        files   = sorted([
            f for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f))
        ])
        # take first n_per_class (deterministic, no randomness needed)
        for fname in files[:n_per_class]:
            samples.append((os.path.join(cls_dir, fname), cls, cls_to_idx[cls]))
    return samples


# ─── main loop ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(XAI_DIR, exist_ok=True)
    all_model_metrics = {}
    samples = sample_test_images(N_PER_CLASS)
    print(f'images per model: {len(samples)} ({N_PER_CLASS}/class)')
    print(f'del-auc steps: {DEL_STEPS}   stab trials: {STAB_TRIALS}')

    for model_name, seed in BEST_SEEDS.items():
        t_start = time.time()
        print(f'\n{"="*60}')
        print(f'model: {model_name}  seed={seed}')
        print(f'{"="*60}')

        # output dirs
        overlay_base = os.path.join(XAI_DIR, model_name, 'overlays')
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(overlay_base, cls), exist_ok=True)
        metrics_path = os.path.join(XAI_DIR, model_name, 'per_image_metrics.json')

        model = load_model(model_name, seed)
        per_image = []

        for i, (img_path, cls_name, cls_idx) in enumerate(samples):
            t0 = time.time()

            # load image
            img_pil    = Image.open(img_path).convert('RGB')
            img_tensor = VAL_TF(img_pil).unsqueeze(0)   # (1,3,224,224)

            # predicted class (may differ from true class)
            with torch.no_grad():
                logits     = model(img_tensor)
                pred_class = logits.argmax(1).item()

            # saliency map
            hmap = get_saliency_map(model, model_name, img_tensor, class_idx=pred_class)

            # save overlay
            overlay  = overlay_heatmap(img_tensor, hmap)
            fname    = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(overlay_base, cls_name, f'{fname}_overlay.png')
            Image.fromarray(overlay).save(out_path)

            # del-auc
            dauc = deletion_auc(model, img_tensor, hmap, pred_class,
                                steps=DEL_STEPS, device=device)

            # perturbation stability
            stab = perturbation_stability(model, model_name, img_tensor,
                                          n_trials=STAB_TRIALS, sigma=0.02, device=device)

            elapsed = time.time() - t0
            per_image.append({
                'image':     os.path.basename(img_path),
                'class':     cls_name,
                'class_idx': cls_idx,
                'pred':      pred_class,
                'correct':   int(pred_class == cls_idx),
                'del_auc':   round(dauc, 4),
                'stab':      round(stab, 4) if not (stab != stab) else None,  # nan→None
                'elapsed_s': round(elapsed, 1),
            })
            print(f'  [{i+1:02d}/{len(samples)}] {cls_name}/{fname}  '
                  f'del_auc={dauc:.4f}  stab={stab:.4f}  ({elapsed:.1f}s)')

        # save per-image log
        with open(metrics_path, 'w') as f:
            json.dump(per_image, f, indent=2)

        # aggregate across images (skip None stab values)
        dauc_vals = [r['del_auc'] for r in per_image]
        stab_vals = [r['stab'] for r in per_image if r['stab'] is not None]

        agg = {
            'model':      model_name,
            'seed':       seed,
            'n_images':   len(per_image),
            'del_auc': {
                'mean': round(float(np.mean(dauc_vals)), 4),
                'std':  round(float(np.std(dauc_vals)),  4),
            },
            'stab': {
                'mean': round(float(np.mean(stab_vals)), 4) if stab_vals else None,
                'std':  round(float(np.std(stab_vals)),  4) if stab_vals else None,
            },
            'total_time_min': round((time.time() - t_start) / 60, 1),
        }
        all_model_metrics[model_name] = agg

        print(f'\n  del_auc: {agg["del_auc"]["mean"]:.4f} +/- {agg["del_auc"]["std"]:.4f}')
        if stab_vals:
            print(f'  stab:    {agg["stab"]["mean"]:.4f} +/- {agg["stab"]["std"]:.4f}')
        print(f'  time:    {agg["total_time_min"]:.1f} min')

    # save summary
    summary_path = os.path.join(XAI_DIR, 'xai_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_model_metrics, f, indent=2)

    print(f'\n\n{"="*60}')
    print('XAI ANALYSIS COMPLETE')
    print(f'{"="*60}')
    for name, m in all_model_metrics.items():
        da = m['del_auc']
        st = m['stab']
        print(f'  {name:20s}  del_auc={da["mean"]:.4f}+/-{da["std"]:.4f}  '
              f'stab={st["mean"]:.4f}+/-{st["std"]:.4f}' if st['mean'] else
              f'  {name:20s}  del_auc={da["mean"]:.4f}+/-{da["std"]:.4f}  stab=N/A')

    print(f'\nresults saved to {XAI_DIR}')


if __name__ == '__main__':
    main()