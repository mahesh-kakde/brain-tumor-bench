# generate grad-cam composite figure for the paper
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE = r'c:\Users\mahes\Desktop\STUFF\brain-tumor-bench'
XAI  = os.path.join(BASE, 'results', 'xai')
OUT  = os.path.join(BASE, 'results', 'paper_figures')
os.makedirs(OUT, exist_ok=True)

MODELS = ['resnet50', 'efficientnet_b0', 'vit_b16', 'hybrid']
LABELS = ['ResNet-50\n(Grad-CAM)', 'EfficientNet-B0\n(Grad-CAM)', 'ViT-B/16\n(Attn. Rollout)', 'Hybrid CNN-ViT\n(Grad-CAM)']

# two rows: glioma and meningioma
ROWS = [
    {'class': 'glioma',     'file': 'Te-gl_1_overlay.png',    'label': 'Glioma'},
    {'class': 'meningioma', 'file': 'Te-aug-me_1_overlay.png','label': 'Meningioma'},
]

matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 9})

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))

for ri, row in enumerate(ROWS):
    for mi, (model, label) in enumerate(zip(MODELS, LABELS)):
        ax = axes[ri, mi]
        path = os.path.join(XAI, model, 'overlays', row['class'], row['file'])
        print(f'checking: {path} -> exists={os.path.exists(path)}')
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_facecolor('#cccccc')
        ax.axis('off')
        if ri == 0:
            ax.set_title(label, fontsize=9, fontweight='bold', pad=4)
        if mi == 0:
            ax.set_ylabel(row['label'], fontsize=10, fontweight='bold', labelpad=6)

fig.text(0.5, 1.01, 'Saliency Maps for Representative MRI Samples',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=fig.transFigure)
fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)

for ext in ['pdf', 'png']:
    p = os.path.join(OUT, f'fig_gradcam_composite.{ext}')
    fig.savefig(p, bbox_inches='tight', dpi=300)
    print(f'saved: {p}')
plt.close(fig)
print('done')