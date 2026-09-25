# chunk 20 part a — efficiency metrics and statistical significance
# computes: params, flops (fvcore), inference time, paired t-tests
import os, sys, json, time, platform, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel, wilcoxon
from torchvision import models

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

RUNS_DIR = os.path.join(PROJECT, 'results', 'kaggle_runs')
TEST_DIR = os.path.join(PROJECT, 'data', 'kaggle', 'Testing')
OUT_DIR  = os.path.join(PROJECT, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SEEDS       = [42, 1337, 2026]

# ─── model builders (no pretrained weights needed) ────────────────────────────
def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 4))
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 4))
    return m

def build_vit_b16():
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 4))
    return m

def build_hybrid():
    from src.models import HybridCNNViT
    return HybridCNNViT()

BUILDERS = {
    'resnet50':        build_resnet50,
    'efficientnet_b0': build_efficientnet_b0,
    'vit_b16':         build_vit_b16,
    'hybrid':          build_hybrid,
}

# ─── 1. param count ───────────────────────────────────────────────────────────
print('='*60)
print('1. PARAMETER COUNTS')
print('='*60)

param_results = {}
for name, builder in BUILDERS.items():
    m = builder()
    n_params = sum(p.numel() for p in m.parameters())
    n_train  = sum(p.numel() for p in m.parameters() if p.requires_grad)
    param_results[name] = {
        'total_params':     n_params,
        'trainable_params': n_train,
        'total_M':          round(n_params / 1e6, 2),
        'trainable_M':      round(n_train  / 1e6, 2),
    }
    print(f'  {name:20s}: {n_params/1e6:.2f}M total  {n_train/1e6:.2f}M trainable')

# ─── 2. flops ─────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('2. FLOPS (fvcore, input 1×3×224×224)')
print('='*60)

flop_results = {}
try:
    from fvcore.nn import FlopCountAnalysis
    dummy = torch.zeros(1, 3, 224, 224)
    for name, builder in BUILDERS.items():
        m = builder().eval()
        try:
            flops = FlopCountAnalysis(m, dummy)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            gflops = flops.total() / 1e9
            flop_results[name] = {'gflops': round(gflops, 3)}
            print(f'  {name:20s}: {gflops:.3f} GFLOPs')
        except Exception as e:
            flop_results[name] = {'gflops': None, 'error': str(e)}
            print(f'  {name:20s}: error — {e}')
except ImportError:
    print('  fvcore not available — skipping flops')
    for name in BUILDERS:
        flop_results[name] = {'gflops': None}

# ─── 3. inference time ────────────────────────────────────────────────────────
print('\n' + '='*60)
print('3. INFERENCE TIME (batch_size=1, 100 images, CPU)')
print('='*60)

from PIL import Image
from torchvision import transforms

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# collect 100 test images (25 per class)
test_imgs = []
for cls in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls)
    files   = sorted([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
    for f in files[:25]:
        test_imgs.append(os.path.join(cls_dir, f))

# preload tensors to exclude i/o from timing
print(f'  preloading {len(test_imgs)} images...')
tensors = [VAL_TF(Image.open(p).convert('RGB')).unsqueeze(0) for p in test_imgs]

time_results = {}
BEST_SEEDS = {'resnet50': 1337, 'efficientnet_b0': 2026, 'vit_b16': 1337, 'hybrid': 42}

for name in BUILDERS:
    seed = BEST_SEEDS[name]
    ckpt_path = os.path.join(RUNS_DIR, name, f'seed_{seed}', 'best.pt')
    m = BUILDERS[name]()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    m.load_state_dict(ckpt['model_state'])
    m.eval()

    # warmup
    with torch.no_grad():
        for t in tensors[:5]:
            _ = m(t)

    # timed run
    times = []
    with torch.no_grad():
        for t in tensors:
            t0 = time.perf_counter()
            _ = m(t)
            times.append(time.perf_counter() - t0)

    mean_ms = float(np.mean(times)) * 1000
    std_ms  = float(np.std(times)) * 1000
    time_results[name] = {
        'mean_ms': round(mean_ms, 2),
        'std_ms':  round(std_ms, 2),
        'n_images': len(tensors),
        'device':  'cpu',
    }
    print(f'  {name:20s}: {mean_ms:.1f} ± {std_ms:.1f} ms/image')

# ─── 4. statistical significance ─────────────────────────────────────────────
print('\n' + '='*60)
print('4. STATISTICAL SIGNIFICANCE (paired t-test on macro f1, n=3 seeds)')
print('='*60)

# load per-seed test macro f1 from eval results
f1_per_model = {}
for name in BUILDERS:
    vals = []
    for seed in SEEDS:
        eval_path = os.path.join(PROJECT, 'results', 'test_eval', name, f'seed_{seed}', 'eval.json')
        with open(eval_path) as f_:
            data = json.load(f_)
        vals.append(data['kaggle_test']['macro_f1'])
    f1_per_model[name] = vals
    print(f'  {name:20s}: {vals}  mean={np.mean(vals):.4f}')

model_names  = list(BUILDERS.keys())
sig_results  = {}
print()
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        a, b   = model_names[i], model_names[j]
        fa, fb = f1_per_model[a], f1_per_model[b]
        diff   = float(np.mean(fa)) - float(np.mean(fb))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, p_t = ttest_rel(fa, fb)
        pair_key = f'{a}_vs_{b}'
        sig_results[pair_key] = {
            'model_a':    a,
            'model_b':    b,
            'f1_a':       fa,
            'f1_b':       fb,
            'mean_diff':  round(diff, 4),
            'p_value_ttest': round(float(p_t), 4),
            'significant_05': bool(p_t < 0.05),
        }
        sig = '* p<0.05' if p_t < 0.05 else 'n.s.'
        print(f'  {a} vs {b}: diff={diff:+.4f}  p={p_t:.4f}  {sig}')

print('\n  note: n=3 seeds → low statistical power. results are indicative.')

# ─── 5. save all results ──────────────────────────────────────────────────────
efficiency = {
    'platform': {
        'python':    platform.python_version(),
        'pytorch':   torch.__version__,
        'device':    'cpu (no gpu available locally)',
        'os':        platform.system(),
    },
    'params':    param_results,
    'flops':     flop_results,
    'inference': time_results,
}

with open(os.path.join(OUT_DIR, 'efficiency.json'), 'w') as f:
    json.dump(efficiency, f, indent=2)

with open(os.path.join(OUT_DIR, 'significance.json'), 'w') as f:
    json.dump(sig_results, f, indent=2)

print(f'\nresults saved to {OUT_DIR}/efficiency.json and significance.json')