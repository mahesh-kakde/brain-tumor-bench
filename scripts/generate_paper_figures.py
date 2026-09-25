# generate publication-ready figures for the brain tumor bench paper
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

PROJECT = r'c:\Users\mahes\Desktop\STUFF\brain-tumor-bench'
sys.path.insert(0, PROJECT)
FIG_DIR = os.path.join(PROJECT, 'results', 'paper_figures')
os.makedirs(FIG_DIR, exist_ok=True)

PLT_RC = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
}
matplotlib.rcParams.update(PLT_RC)

MODELS      = ['resnet50', 'efficientnet_b0', 'vit_b16', 'hybrid']
MODEL_LABEL = {
    'resnet50':        'ResNet-50',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b16':         'ViT-B/16',
    'hybrid':          'Hybrid CNN-ViT',
}
SEEDS       = [42, 1337, 2026]
COLORS      = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
SEED_COLORS = {42: '#4878d0', 1337: '#6acc65', 2026: '#d65f5f'}
CLASS_KEYS  = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
RUNS_DIR    = os.path.join(PROJECT, 'results', 'kaggle_runs')
TEST_EVAL   = os.path.join(PROJECT, 'results', 'test_eval')

def save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p + '.pdf', bbox_inches='tight')
    fig.savefig(p + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'saved: {name}')

def fig_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.flatten()
    for ax_i, model in enumerate(MODELS):
        ax = axes[ax_i]
        for si, seed in enumerate(SEEDS):
            log_path = os.path.join(RUNS_DIR, model, f'seed_{seed}', 'log.json')
            with open(log_path) as f:
                recs = json.load(f)
            epochs = [r['epoch'] for r in recs]
            f1s    = [r['val_f1'] for r in recs]
            ax.plot(epochs, f1s, color=SEED_COLORS[seed], lw=1.6,
                    label=f'seed {seed}', marker='o', markersize=3, alpha=0.9)
            best = int(np.argmax(f1s))
            ax.scatter([epochs[best]], [f1s[best]], color=SEED_COLORS[seed],
                       s=60, zorder=5, edgecolors='k', linewidths=0.8)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val. Macro F1')
        ax.set_ylim(0.93, 1.005)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    fig.suptitle('Validation Macro F\u2081 Learning Curves \u2014 All 12 Training Runs',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig1_learning_curves')

def fig_main_results():
    metrics = {
        'Accuracy':      ([0.9531, 0.9562, 0.9487, 0.9483], [0.0027, 0.0022, 0.0037, 0.0028]),
        'Macro F1':      ([0.9522, 0.9554, 0.9476, 0.9470], [0.0026, 0.0023, 0.0038, 0.0028]),
        'AUROC':         ([0.9856, 0.9826, 0.9777, 0.9800], [0.0024, 0.0026, 0.0033, 0.0030]),
        'ECE (down)':    ([0.0313, 0.0278, 0.0128, 0.0189], [0.0127, 0.0058, 0.0013, 0.0062]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    x = np.arange(len(MODELS))
    xlabels = [MODEL_LABEL[m] for m in MODELS]
    for ax, (metric, (vals, stds)) in zip(axes, metrics.items()):
        bars = ax.bar(x, vals, yerr=stds, capsize=4, color=COLORS, alpha=0.82,
                      error_kw={'linewidth': 1.4, 'color': '#333'}, width=0.6,
                      edgecolor='white', linewidth=0.8)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Score')
        if 'ECE' in metric:
            best_i = int(np.argmin(vals))
        else:
            best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor('#333')
        bars[best_i].set_linewidth(1.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(stds)*1.6,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)
        margin = max(stds) * 6
        ax.set_ylim(min(vals) - margin, max(vals) + margin * 2.5)
    fig.suptitle('Test Set Performance \u2014 Mean \u00b1 Std across 3 Seeds (1,600 images)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig2_main_results')

def fig_confusion_matrices():
    best_seeds = {'resnet50': 1337, 'efficientnet_b0': 2026, 'vit_b16': 1337, 'hybrid': 42}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    for ax_i, model in enumerate(MODELS):
        seed = best_seeds[model]
        eval_path = os.path.join(TEST_EVAL, model, f'seed_{seed}', 'eval.json')
        with open(eval_path) as f:
            data = json.load(f)
        cm = np.array(data['kaggle_test']['confusion_matrix'])
        n_classes = cm.shape[0]
        class_names = CLASS_NAMES[:n_classes]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax = axes[ax_i]
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{MODEL_LABEL[model]}\n(seed {seed}, F1={data["kaggle_test"]["macro_f1"]:.4f})')
        for r in range(n_classes):
            for c in range(n_classes):
                color = 'white' if cm_norm[r, c] > 0.5 else 'black'
                ax.text(c, r, f'{cm_norm[r,c]:.2f}\n({cm[r,c]})',
                        ha='center', va='center', fontsize=8, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Normalised Confusion Matrices \u2014 Best Seed per Model (Kaggle Test Set)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig3_confusion_matrices')

def fig_efficiency_pareto():
    acc    = [0.9531, 0.9562, 0.9487, 0.9483]
    flops  = [4.109,  0.400,  16.867, 4.791]
    params = [23.52,  4.01,   85.80,  37.19]
    infer  = [44.81,  17.9,   132.84, 58.14]
    labels = [MODEL_LABEL[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sizes = [p * 8 for p in params]
    sc = ax.scatter(flops, acc, s=sizes, c=infer, cmap='RdYlGn_r',
                    alpha=0.85, edgecolors='#333', linewidths=1.2, zorder=5)
    cb = plt.colorbar(sc, ax=ax, label='CPU Inference Time (ms/image)', pad=0.02)
    cb.ax.tick_params(labelsize=9)
    offsets = {
        'ResNet-50':       (0.25,  0.0005),
        'EfficientNet-B0': (0.08,  0.0005),
        'ViT-B/16':        (0.25, -0.0015),
        'Hybrid CNN-ViT':  (0.25,  0.0005),
    }
    for i, (label, x, y) in enumerate(zip(labels, flops, acc)):
        dx, dy = offsets.get(label, (0.1, 0.0005))
        ax.annotate(f'{label}\n({params[i]:.1f}M params)',
                    xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=8.5, ha='left',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))
    ax.set_xlabel('FLOPs (G)')
    ax.set_ylabel('Test Set Accuracy')
    ax.set_title('Efficiency\u2013Accuracy Trade-off\n(bubble area proportional to parameter count; colour = inference time)',
                 fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.set_xlim(-1, 20)
    ax.set_ylim(0.944, 0.960)
    ax.annotate('Pareto\nOptimum', xy=(0.400, 0.9562),
                xytext=(2.5, 0.9555),
                fontsize=8.5, color='#d62728', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
    fig.tight_layout()
    save(fig, 'fig4_efficiency_pareto')

def fig_xai_metrics():
    del_auc_mean = [0.4201, 0.3489, 0.3913, 0.4970]
    del_auc_std  = [0.2669, 0.2547, 0.2671, 0.3573]
    stab_mean    = [0.9847, 0.9057, 0.9998, 0.8209]
    stab_std     = [0.0171, 0.1671, 0.0002, 0.2178]
    labels       = [MODEL_LABEL[m] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bars1 = ax1.bar(x, del_auc_mean, yerr=del_auc_std, capsize=5,
                    color=COLORS, alpha=0.82, width=0.6,
                    error_kw={'linewidth': 1.4, 'color': '#333'},
                    edgecolor='white', linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=18, ha='right')
    ax1.set_ylabel('Deletion AUC (lower is better)')
    ax1.set_title('Deletion Faithfulness (Del-AUC)')
    ax1.set_ylim(0, 0.90)
    best1 = int(np.argmin(del_auc_mean))
    bars1[best1].set_edgecolor('#333'); bars1[best1].set_linewidth(1.8)
    for bar, v in zip(bars1, del_auc_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=8)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='Random baseline')
    ax1.legend(fontsize=8)
    bars2 = ax2.bar(x, stab_mean, yerr=stab_std, capsize=5,
                    color=COLORS, alpha=0.82, width=0.6,
                    error_kw={'linewidth': 1.4, 'color': '#333'},
                    edgecolor='white', linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=18, ha='right')
    ax2.set_ylabel('Spearman Correlation (higher is better)')
    ax2.set_title('Perturbation Stability (Stab)')
    ax2.set_ylim(0.5, 1.10)
    best2 = int(np.argmax(stab_mean))
    bars2[best2].set_edgecolor('#333'); bars2[best2].set_linewidth(1.8)
    for bar, v in zip(bars2, stab_mean):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=8)
    fig.suptitle('Explainability Metrics \u2014 XAI Quality Comparison', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig5_xai_metrics')

def fig_perclass_f1():
    best_seeds = {'resnet50': 1337, 'efficientnet_b0': 2026, 'vit_b16': 1337, 'hybrid': 42}
    perclass = np.zeros((4, 4))
    for mi, model in enumerate(MODELS):
        seed = best_seeds[model]
        eval_path = os.path.join(TEST_EVAL, model, f'seed_{seed}', 'eval.json')
        with open(eval_path) as f:
            data = json.load(f)
        pc = data['kaggle_test']['per_class_f1']  # dict keyed by class name
        for ci, key in enumerate(CLASS_KEYS):
            perclass[mi, ci] = pc[key]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(perclass, cmap='YlOrRd', vmin=0.88, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_xlabel('Class')
    ax.set_ylabel('Model')
    ax.set_title('Per-Class F1 Score \u2014 Best Seed per Model (Kaggle Test Set)', fontsize=11)
    for r in range(4):
        for c in range(4):
            color = 'white' if perclass[r, c] < 0.93 else 'black'
            ax.text(c, r, f'{perclass[r,c]:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, label='F1 Score', fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, 'fig6_perclass_f1')

def fig_calibration():
    ece_mean = [0.0313, 0.0278, 0.0128, 0.0189]
    ece_std  = [0.0127, 0.0058, 0.0013, 0.0062]
    acc_mean = [0.9531, 0.9562, 0.9487, 0.9483]
    labels   = [MODEL_LABEL[m] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = ax1.bar(x, ece_mean, yerr=ece_std, capsize=5, color=COLORS,
                   alpha=0.82, width=0.6, error_kw={'linewidth': 1.4, 'color': '#333'},
                   edgecolor='white', linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=18, ha='right')
    ax1.set_ylabel('Expected Calibration Error (lower is better)')
    ax1.set_title('Calibration Error (ECE)')
    ax1.set_ylim(0, 0.060)
    best_ece = int(np.argmin(ece_mean))
    bars[best_ece].set_edgecolor('#333'); bars[best_ece].set_linewidth(1.8)
    for bar, v in zip(bars, ece_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=8)
    sc = ax2.scatter(ece_mean, acc_mean, s=[p/85.80*800 + 100 for p in [23.52, 4.01, 85.80, 37.19]],
                     c=COLORS, alpha=0.85, edgecolors='#333', linewidths=1.2, zorder=5)
    for i, (label, ex, ay) in enumerate(zip(labels, ece_mean, acc_mean)):
        ax2.annotate(label, (ex, ay), textcoords='offset points', xytext=(6, 3), fontsize=8.5)
    ax2.set_xlabel('Expected Calibration Error (ECE) lower is better')
    ax2.set_ylabel('Test Accuracy higher is better')
    ax2.set_title('Accuracy-Calibration Trade-off\n(bubble size proportional to parameter count)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax2.set_xlim(-0.003, 0.045)
    ax2.set_ylim(0.944, 0.960)
    ax2.annotate('Ideal\ncorner', xy=(0, 1.0), xytext=(0.003, 0.9585),
                 fontsize=8, color='green', fontstyle='italic',
                 arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
    fig.suptitle('Confidence Calibration Analysis', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig7_calibration')

def fig_significance():
    pvals = np.array([
        [1.0,    0.0296, 0.2074, 0.2481],
        [0.0296, 1.0,    0.1180, 0.1286],
        [0.2074, 0.1180, 1.0,    0.8118],
        [0.2481, 0.1286, 0.8118, 1.0   ],
    ])
    labels = [MODEL_LABEL[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    masked = np.ma.masked_where(np.eye(4, dtype=bool), pvals)
    im = ax.imshow(masked, cmap='RdYlGn', vmin=0, vmax=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    ax.set_title('Paired t-test p-values for Macro F1\n(n=3 seeds; diagonal = N/A)', fontsize=11)
    for r in range(4):
        for c in range(4):
            if r == c:
                ax.text(c, r, '-', ha='center', va='center', fontsize=12, color='gray')
            else:
                p = pvals[r, c]
                txt = f'p={p:.3f}'
                sig = ' *' if p < 0.05 else ''
                color = 'white' if p < 0.15 else 'black'
                ax.text(c, r, txt + sig, ha='center', va='center',
                        fontsize=9, fontweight='bold' if p < 0.05 else 'normal', color=color)
    plt.colorbar(im, ax=ax, label='p-value (lower = more significant)', fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, 'fig8_significance')

def fig_figshare():
    fs_acc = {
        'resnet50':        [0.9125, 0.9086, 0.9393],
        'efficientnet_b0': [0.9683, 0.9667, 0.9664],
        'vit_b16':         [0.9445, 0.9452, 0.9050],
        'hybrid':          [0.8965, 0.9370, 0.9151],
    }
    kaggle_acc = [0.9531, 0.9562, 0.9487, 0.9483]
    fs_means = [np.mean(v) for v in fs_acc.values()]
    fs_stds  = [np.std(v)  for v in fs_acc.values()]
    labels = [MODEL_LABEL[m] for m in MODELS]
    x = np.arange(len(MODELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w/2, kaggle_acc, w, color=COLORS, alpha=0.85,
                label='Kaggle Test (primary)', edgecolor='white')
    b2 = ax.bar(x + w/2, fs_means, w, yerr=fs_stds, capsize=4,
                color=COLORS, alpha=0.45, hatch='//',
                label='Figshare (sensitivity check, 3-class)', edgecolor='white',
                error_kw={'linewidth': 1.2, 'color': '#555'})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Dataset Comparison: Kaggle vs Figshare Sensitivity Check\n'
                 '(Figshare: 3-class; overlap audit found 1,762 near-duplicate pairs - not independent)',
                 fontsize=10)
    ax.set_ylim(0.86, 0.98)
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.axhline(0.90, color='gray', linestyle=':', alpha=0.5)
    fig.tight_layout()
    save(fig, 'fig9_figshare_sensitivity')

if __name__ == '__main__':
    print('generating publication figures...')
    fig_learning_curves()
    fig_main_results()
    fig_confusion_matrices()
    fig_efficiency_pareto()
    fig_xai_metrics()
    fig_perclass_f1()
    fig_calibration()
    fig_significance()
    fig_figshare()
    print(f'all figures saved to {FIG_DIR}')