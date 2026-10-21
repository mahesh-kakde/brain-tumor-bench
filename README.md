# brain-tumor-bench

a reproducible benchmark comparing four deep learning architectures for brain tumor mri classification under identical training conditions.

## what is this?

this project performs a fair, head-to-head comparison of:
- **resnet-50** — classic cnn baseline (23.52M params, 4.11 GFLOPs)
- **efficientnet-b0** — lightweight cnn (4.01M params, 0.40 GFLOPs)
- **vit-b/16** — vision transformer (85.80M params, 16.87 GFLOPs)
- **hybrid cnn-vit** — resnet-50 backbone + 4-layer transformer head (37.19M params, 4.79 GFLOPs)

all four models are trained under the **exact same protocol** (same optimizer, lr, scheduler, augmentation, early stopping) so performance differences are attributable to architectural inductive biases, not tuning advantages.

## task

four-class classification of brain mri slices:

| class | description |
|---|---|
| glioma | tumor originating from glial cells |
| meningioma | tumor on brain/spinal cord membranes |
| no tumor | healthy brain mri |
| pituitary | tumor in the pituitary gland |

## datasets

| dataset | images | classes | role |
|---|---|---|---|
| kaggle brain tumor mri | 7,200 (5,600 train + 1,600 test) | 4 | primary benchmark |
| figshare brain tumor | 3,064 | 3 | sensitivity check only |

> ⚠️ figshare results are sensitivity checks only — ~1,762 perceptual near-duplicate pairs exist with the kaggle training set.

## results (kaggle held-out test set, mean ± std across 3 seeds)

| model | accuracy | macro f1 | auroc | ece | params | flops | cpu inference |
|---|---|---|---|---|---|---|---|
| resnet-50 | 0.9531 ± 0.0027 | 0.9522 ± 0.0026 | 0.9856 ± 0.0024 | 0.031 | 23.52M | 4.11G | 44.8ms |
| **efficientnet-b0** | **0.9562 ± 0.0022** | **0.9554 ± 0.0023** | 0.9826 ± 0.0026 | 0.028 | **4.01M** | **0.40G** | **17.9ms** |
| vit-b/16 | 0.9487 ± 0.0037 | 0.9476 ± 0.0038 | 0.9777 ± 0.0033 | **0.013** | 85.80M | 16.87G | 132.8ms |
| hybrid cnn-vit | 0.9483 ± 0.0028 | 0.9470 ± 0.0028 | 0.9800 ± 0.0030 | 0.019 | 37.19M | 4.79G | 58.1ms |

**efficientnet-b0 is the pareto optimum** — highest accuracy with the fewest parameters, lowest flops, and fastest inference.

see [RESULTS.md](RESULTS.md) for full tables including per-run metrics, figshare sensitivity check, explainability analysis, and statistical significance tests.

## key design principles

- identical training protocol across all architectures
- three random seeds (42, 1337, 2026) for mean ± std reporting
- evaluation beyond accuracy: calibration (ece), explainability (del-auc, stab), efficiency (params, flops, inference time)
- statistical significance testing between all 6 model pairs
- full provenance audit between datasets

## training

all training runs are executed on **kaggle notebooks** (gpu). local machine is used for code writing, data prep, and analysis only.

### hyperparameters

| param | value |
|---|---|
| optimizer | adamw |
| lr | 1e-4 |
| weight decay | 0.01 |
| scheduler | cosine annealing (t_max=30) |
| label smoothing | 0.05 |
| max epochs | 30 |
| early stopping | patience=7, criterion=val macro f1 |
| batch size | 32 |
| input size | 224×224 |

## repo structure

```
brain-tumor-bench/
├── README.md          # this file
├── RESULTS.md         # all results and insights
├── requirements.txt   # python dependencies
├── data/
│   ├── kaggle/        # primary dataset (train + test) — gitignored
│   ├── figshare/      # external dataset (sensitivity check)
│   └── splits/        # train/val split json (seed=42)
├── src/               # model code, training, evaluation, xai
│   ├── models.py      # all 4 architectures + build_model() factory
│   ├── data.py        # dataset classes + dataloaders
│   ├── train.py       # training loop (adamw, cosine lr, early stopping)
│   ├── utils.py       # seeding, metrics, checkpointing, logging
│   └── explainability.py  # grad-cam, attention rollout, del-auc, stab
├── notebooks/         # kaggle training notebook (train_all.ipynb)
├── results/           # all output artifacts
│   ├── kaggle_runs/   # training logs (checkpoints gitignored — too large)
│   ├── test_eval/     # evaluation results per model/seed
│   ├── xai/           # heatmap overlays + xai metrics
│   ├── plots/         # training curves + final metrics bar chart
│   ├── efficiency.json
│   └── significance.json
└── scripts/           # utility scripts
    ├── verify_data.py
    ├── prepare_figshare.py
    ├── split_data.py
    ├── audit_overlap.py
    ├── evaluate.py
    ├── run_xai.py
    ├── efficiency_and_significance.py
    └── plot_training_curves.py
```

## docs

- [PROJECT.html](PROJECT.html) — project tools, datasets, and architecture details

## license

this project is for academic research purposes.

## author


[mahesh kakde](https://github.com/mahesh-kakde)
