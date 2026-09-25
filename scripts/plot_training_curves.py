# chunk 20 part b — training curve plots
# plots loss and val_f1 vs epoch for all 12 runs
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT, 'results', 'kaggle_runs')
PLOT_DIR = os.path.join(PROJECT, 'results', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

MODELS     = ['resnet50', 'efficientnet_b0', 'vit_b16', 'hybrid']
MODEL_LABELS = {
    'resnet50':        'ResNet-50',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b16':         'ViT-B/16',
    'hybrid':          'Hybrid CNN-ViT',
}
SEEDS      = [42, 1337, 2026]
SEED_COLORS = {42: '#58a6ff', 1337: '#3fb950', 2026: '#f78166'}

def load_log(model_name, seed):
    path = os.path.join(RUNS_DIR, model_name, f'seed_{seed}', 'log.json')
    with open(path) as f:
        return json.load(f)

# ─── plot 1: val_f1 curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')
axes = axes.flatten()

for ax_idx, model_name in enumerate(MODELS):
    ax = axes[ax_idx]
    ax.set_facecolor('#161b22')
    for seed in SEEDS:
        records = load_log(model_name, seed)
        epochs  = [r['epoch'] for r in records]
        f1s     = [r['val_f1'] for r in records]
        ax.plot(epochs, f1s, color=SEED_COLORS[seed], linewidth=1.8,
                label=f'seed {seed}', marker='o', markersize=3, alpha=0.9)
        # mark best
        best_idx = int(np.argmax(f1s))
        ax.scatter([epochs[best_idx]], [f1s[best_idx]],
                   color=SEED_COLORS[seed], s=80, zorder=5, edgecolors='white', linewidths=0.8)

    ax.set_title(MODEL_LABELS[model_name], color='#c9d1d9', fontsize=12, fontweight='bold')
    ax.set_xlabel('epoch', color='#8b949e', fontsize=10)
    ax.set_ylabel('val macro f1', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color('#30363d')
    ax.set_ylim(0.92, 1.01)
    ax.legend(fontsize=9, facecolor='#0d1117', edgecolor='#30363d',
              labelcolor='#c9d1d9', loc='lower right')
    ax.grid(True, color='#30363d', linewidth=0.5, alpha=0.5)

fig.suptitle('Validation Macro F1 vs Epoch — All 12 Training Runs',
             color='#58a6ff', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'training_curves_val_f1.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'saved: {out}')

# ─── plot 2: training loss curves ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')
axes = axes.flatten()

for ax_idx, model_name in enumerate(MODELS):
    ax = axes[ax_idx]
    ax.set_facecolor('#161b22')
    for seed in SEEDS:
        records   = load_log(model_name, seed)
        epochs    = [r['epoch'] for r in records]
        tr_loss   = [r['train_loss'] for r in records]
        val_loss  = [r['val_loss']   for r in records]
        ax.plot(epochs, tr_loss,  color=SEED_COLORS[seed], linewidth=1.5,
                label=f'seed {seed} train', linestyle='-',  alpha=0.9)
        ax.plot(epochs, val_loss, color=SEED_COLORS[seed], linewidth=1.5,
                label=f'seed {seed} val',   linestyle='--', alpha=0.6)

    ax.set_title(MODEL_LABELS[model_name], color='#c9d1d9', fontsize=12, fontweight='bold')
    ax.set_xlabel('epoch', color='#8b949e', fontsize=10)
    ax.set_ylabel('loss', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color('#30363d')
    ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#30363d',
              labelcolor='#c9d1d9', loc='upper right', ncol=2)
    ax.grid(True, color='#30363d', linewidth=0.5, alpha=0.5)

fig.suptitle('Train / Val Loss vs Epoch — All 12 Training Runs',
             color='#58a6ff', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'training_curves_loss.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'saved: {out}')

# ─── plot 3: final test metrics bar chart ─────────────────────────────────────
test_f1_mean  = [0.9522, 0.9554, 0.9476, 0.9470]
test_f1_std   = [0.0026, 0.0023, 0.0038, 0.0028]
test_auroc    = [0.9856, 0.9826, 0.9777, 0.9800]
labels        = ['ResNet-50', 'EfficientNet-B0', 'ViT-B/16', 'Hybrid']
x             = np.arange(len(labels))
width         = 0.35
bar_colors    = ['#58a6ff', '#3fb950', '#f78166', '#bc8cff']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

for ax in (ax1, ax2):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color('#30363d')
    ax.grid(True, color='#30363d', linewidth=0.5, alpha=0.5, axis='y')

bars = ax1.bar(x, test_f1_mean, width=0.6, color=bar_colors, alpha=0.85,
               yerr=test_f1_std, capsize=4, error_kw={'color': '#8b949e', 'linewidth': 1.5})
ax1.set_title('Test Set Macro F1 (mean ± std, 3 seeds)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax1.set_ylabel('macro f1', color='#8b949e')
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha='right', color='#c9d1d9', fontsize=9)
ax1.set_ylim(0.93, 0.97)
for bar, val in zip(bars, test_f1_mean):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}',
             ha='center', va='bottom', color='#c9d1d9', fontsize=8)

bars2 = ax2.bar(x, test_auroc, width=0.6, color=bar_colors, alpha=0.85)
ax2.set_title('Test Set AUROC (mean, 3 seeds)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax2.set_ylabel('auroc', color='#8b949e')
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha='right', color='#c9d1d9', fontsize=9)
ax2.set_ylim(0.96, 1.00)
for bar, val in zip(bars2, test_auroc):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}',
             ha='center', va='bottom', color='#c9d1d9', fontsize=8)

fig.suptitle('Final Test Set Performance — All Models',
             color='#58a6ff', fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'final_test_metrics.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'saved: {out}')

print('\nall plots saved to results/plots/')