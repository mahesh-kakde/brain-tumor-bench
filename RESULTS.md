# results

all experimental results, metrics, and insights from the brain-tumor-bench project.

> this file is updated as results are obtained. check the date stamps for the latest entries.

---

## table of contents

- [dataset verification](#dataset-verification)
- [data split counts](#data-split-counts)
- [provenance audit](#provenance-audit)
- [training results](#training-results)
- [evaluation results — internal test set](#evaluation-results--internal-test-set)
- [evaluation results — external dataset (figshare)](#evaluation-results--external-dataset-figshare)
- [explainability metrics](#explainability-metrics)
- [efficiency metrics](#efficiency-metrics)
- [statistical significance](#statistical-significance)
- [insights and observations](#insights-and-observations)

---

## dataset verification

**date:** 2026-09-10
**verified by:** `scripts/verify_data.py` ✅

### kaggle brain tumor mri

| split | glioma | meningioma | no tumor | pituitary | total |
|---|---|---|---|---|---|
| training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| testing | 400 | 400 | 400 | 400 | 1,600 |
| **total** | **1,800** | **1,800** | **1,800** | **1,800** | **7,200** |

### figshare brain tumor mri

| class | images |
|---|---|
| glioma | 1,426 |
| meningioma | 708 |
| pituitary | 930 |
| **total** | **3,064** |

- source format: matlab .mat files (v7.3 / HDF5)
- patients: 233
- note: no "no tumor" class in this dataset
- **conversion:** .mat → png complete ✅ (via `scripts/prepare_figshare.py`)
- output: `data/figshare/processed/{glioma, meningioma, pituitary}/`

---

## data split counts

**date:** 2026-09-10
**created by:** `scripts/split_data.py` (seed=42, val_ratio=0.15, stratified) ✅
**split file:** `data/splits/train_val_split.json`

| subset | glioma | meningioma | no tumor | pituitary | total |
|---|---|---|---|---|---|
| train (85%) | 1,190 | 1,190 | 1,190 | 1,190 | 4,760 |
| val (15%) | 210 | 210 | 210 | 210 | 840 |
| test (held-out) | 400 | 400 | 400 | 400 | 1,600 |

---

## provenance audit

**date:** 2026-09-11
**script:** `scripts/audit_overlap.py` ✅
**log:** `results/audit_log.json`

| check | result |
|---|---|
| exact sha-256 matches | **0** (as expected) |
| perceptual near-duplicates (phash, hamming ≤ 10) | **1,762 pairs** |

**note on near-duplicate count:** blueprint estimated ~706 pairs. our audit found 1,762 at hamming distance ≤ 10. the discrepancy likely reflects a stricter threshold in the original audit. key conclusion is unchanged — significant overlap exists. figshare results must be labelled as **sensitivity check only**, never as independent generalisation evidence.

---

## training results

**date:** 2026-09-13
**source:** kaggle GPU (T4) — all 12 runs complete ✅
**logs:** `results/kaggle_runs/{model}/seed_{seed}/log.json`
**checkpoints:** `results/kaggle_runs/{model}/seed_{seed}/best.pt`

### per-run training summary

| model | seed | best epoch | epochs run | best val macro f1 | best val acc |
|---|---|---|---|---|---|
| resnet-50 | 42 | 5 | 12 | 0.9941 | 0.9940 |
| resnet-50 | 1337 | 25 | 30 | 0.9988 | 0.9988 |
| resnet-50 | 2026 | 16 | 23 | 0.9976 | 0.9976 |
| efficientnet-b0 | 42 | 16 | 23 | 0.9988 | 0.9988 |
| efficientnet-b0 | 1337 | 16 | 23 | 0.9988 | 0.9988 |
| efficientnet-b0 | 2026 | 13 | 20 | 0.9988 | 0.9988 |
| vit-b/16 | 42 | 26 | 30 | 0.9905 | 0.9905 |
| vit-b/16 | 1337 | 26 | 30 | 0.9929 | 0.9929 |
| vit-b/16 | 2026 | 15 | 22 | 0.9941 | 0.9940 |
| hybrid cnn-vit | 42 | 23 | 30 | 0.9952 | 0.9952 |
| hybrid cnn-vit | 1337 | 16 | 23 | 0.9941 | 0.9940 |
| hybrid cnn-vit | 2026 | 4 | 11 | 0.9905 | 0.9905 |

### mean val f1 across seeds (val set — not final test set)

| model | mean val macro f1 | mean val acc |
|---|---|---|
| efficientnet-b0 | **0.9988** | **0.9988** |
| resnet-50 | 0.9968 | 0.9968 |
| hybrid cnn-vit | 0.9933 | 0.9933 |
| vit-b/16 | 0.9925 | 0.9925 |

> ⚠️ these are **validation set** metrics only. final rankings will use the held-out test set (1,600 images) — see evaluation results section below.

---

## evaluation results — internal test set

**date:** 2026-09-13
**source:** local CPU inference on kaggle held-out test set (1,600 images, 400/class)
**detail logs:** `results/test_eval/{model}/seed_{seed}/eval.json`
**summary:** `results/test_eval/summary.json`

### aggregated results (mean ± std across 3 seeds)

| model | accuracy | macro f1 | macro precision | macro recall | auroc | ece |
|---|---|---|---|---|---|---|
| resnet-50 | 0.9531 ± 0.0027 | 0.9522 ± 0.0026 | 0.9571 ± 0.0021 | 0.9531 ± 0.0027 | 0.9856 ± 0.0024 | 0.0313 ± 0.0127 |
| **efficientnet-b0** | **0.9562 ± 0.0022** | **0.9554 ± 0.0023** | **0.9596 ± 0.0021** | **0.9562 ± 0.0022** | 0.9826 ± 0.0026 | 0.0278 ± 0.0058 |
| vit-b/16 | 0.9487 ± 0.0037 | 0.9476 ± 0.0038 | 0.9517 ± 0.0040 | 0.9488 ± 0.0037 | 0.9777 ± 0.0033 | **0.0128 ± 0.0013** |
| hybrid cnn-vit | 0.9483 ± 0.0028 | 0.9470 ± 0.0028 | 0.9522 ± 0.0030 | 0.9483 ± 0.0028 | 0.9800 ± 0.0030 | 0.0189 ± 0.0062 |

> **bold** = best in column. ece = expected calibration error (lower is better).

### per-run test results

| model | seed | accuracy | macro f1 | auroc | ece |
|---|---|---|---|---|---|
| resnet-50 | 42 | 0.9494 | 0.9486 | 0.9823 | 0.0140 |
| resnet-50 | 1337 | 0.9556 | 0.9546 | 0.9880 | 0.0440 |
| resnet-50 | 2026 | 0.9544 | 0.9535 | 0.9867 | 0.0360 |
| efficientnet-b0 | 42 | 0.9531 | 0.9521 | 0.9843 | 0.0326 |
| efficientnet-b0 | 1337 | 0.9575 | 0.9567 | 0.9847 | 0.0313 |
| efficientnet-b0 | 2026 | 0.9581 | 0.9573 | 0.9789 | 0.0196 |
| vit-b/16 | 42 | 0.9475 | 0.9463 | 0.9767 | 0.0110 |
| vit-b/16 | 1337 | 0.9537 | 0.9527 | 0.9822 | 0.0135 |
| vit-b/16 | 2026 | 0.9450 | 0.9438 | 0.9742 | 0.0139 |
| hybrid cnn-vit | 42 | 0.9506 | 0.9493 | 0.9836 | 0.0122 |
| hybrid cnn-vit | 1337 | 0.9500 | 0.9487 | 0.9762 | 0.0174 |
| hybrid cnn-vit | 2026 | 0.9444 | 0.9431 | 0.9801 | 0.0271 |

### key observations

- all four models exceed **94.4% accuracy and 94.3% macro f1** on the held-out test set.
- **efficientnet-b0** leads on accuracy (95.62%) and macro f1 (95.54%) — remarkable given it has the fewest parameters (4M).
- **resnet-50** leads on auroc (0.9856), suggesting the best class-separation confidence.
- **vit-b/16** has the lowest ece (0.0128) — best calibrated; confidence closely tracks accuracy.
- val/test gap is ~4–5% across all models — expected (val is random in-distribution split, test is fixed held-out).
- low variance across seeds (std < 0.004) — training protocol is stable and reproducible.

---

## evaluation results — external dataset (figshare)

> ⚠️ **sensitivity check only** — not independent generalisation evidence due to ~1,762 perceptual near-duplicate pairs with kaggle training data.
> model trained on 4 classes; figshare evaluated on 3 matching classes (glioma, meningioma, pituitary) by masking the "notumor" logit.

### per-run figshare results (3-class: glioma / meningioma / pituitary)

| model | seed | accuracy | macro f1 | auroc | ece |
|---|---|---|---|---|---|
| resnet-50 | 42 | 0.9125 | 0.9016 | 0.9895 | 0.0334 |
| resnet-50 | 1337 | 0.9086 | 0.9008 | 0.9890 | 0.0252 |
| resnet-50 | 2026 | 0.9393 | 0.9304 | 0.9898 | 0.0238 |
| efficientnet-b0 | 42 | 0.9683 | 0.9662 | 0.9979 | 0.0283 |
| efficientnet-b0 | 1337 | 0.9667 | 0.9636 | 0.9969 | 0.0310 |
| efficientnet-b0 | 2026 | 0.9664 | 0.9629 | 0.9976 | 0.0292 |
| vit-b/16 | 42 | 0.9445 | 0.9381 | 0.9892 | 0.0193 |
| vit-b/16 | 1337 | 0.9452 | 0.9396 | 0.9917 | 0.0122 |
| vit-b/16 | 2026 | 0.9050 | 0.8962 | 0.9819 | 0.0342 |
| hybrid cnn-vit | 42 | 0.8965 | 0.8838 | 0.9784 | 0.0587 |
| hybrid cnn-vit | 1337 | 0.9370 | 0.9331 | 0.9913 | 0.0145 |
| hybrid cnn-vit | 2026 | 0.9151 | 0.9078 | 0.9868 | 0.0181 |

### figshare sensitivity observations

- **efficientnet-b0** leads (~96.6% accuracy, ~96.4% f1) — consistent across all 3 seeds.
- results must not be cited as independent generalisation due to dataset overlap.

---

## explainability metrics

**date:** 2026-09-13
**source:** `scripts/run_xai.py` — 10 images/class (40 total) per model, best seed each
**detail logs:** `results/xai/{model}/per_image_metrics.json`
**summary:** `results/xai/xai_summary.json`
**overlays:** `results/xai/{model}/overlays/{class}/`

| model | xai method | del-auc ↓ (mean ± std) | stab ↑ (mean ± std) |
|---|---|---|---|
| resnet-50 | grad-cam (`layer4[-1].conv3`) | 0.4201 ± 0.2669 | 0.9847 ± 0.0171 |
| **efficientnet-b0** | grad-cam (`features[-1]`) | **0.3489 ± 0.2547** | 0.9057 ± 0.1671 |
| vit-b/16 | attention rollout (12 layers) | 0.3913 ± 0.2671 | **0.9998 ± 0.0002** |
| hybrid cnn-vit | grad-cam (cnn `layer4`) | 0.4970 ± 0.3573 | 0.8209 ± 0.2178 |

> del-auc: deletion faithfulness — lower is better (saliency correctly identifies most important pixels).
> stab: perturbation stability (σ=0.02, 5 trials, spearman corr) — higher is better.

### xai key observations

- **efficientnet-b0** has the lowest del-auc (0.3489) — its Grad-CAM maps are the most faithful; removing top-salient pixels causes the sharpest confidence drop.
- **vit-b/16** has near-perfect stability (0.9998 ± 0.0002) — attention rollout maps are extremely consistent under noise, confirming that the attention mechanism is genuinely attending to stable image regions.
- **resnet-50** is the most stable CNN (stab=0.9847) with low variance — reliable Grad-CAM.
- **hybrid cnn-vit** has the highest del-auc (0.4970) and most variable stab (0.2178) — the Grad-CAM on the CNN branch of a hybrid model may not fully reflect the transformer's actual decision process.
- high stab does not imply low del-auc: ViT has very stable maps (0.9998) but moderate del-auc (0.3913), suggesting attention rollout highlights consistent regions but they are not always the most discriminative ones.

---

## efficiency metrics

**date:** 2026-09-13
**source:** `scripts/efficiency_and_significance.py` — cpu (python 3.14, pytorch 2.14.0+cpu)
**detail:** `results/efficiency.json`

| model | params (M) | flops (G) | inference time (ms/img, cpu) |
|---|---|---|---|
| resnet-50 | 23.52 | 4.109 | 44.8 ± 6.1 |
| **efficientnet-b0** | **4.01** | **0.400** | **17.9 ± 2.4** |
| vit-b/16 | 85.80 | 16.867 | 132.8 ± 4.4 |
| hybrid cnn-vit | 37.19 | 4.791 | 58.1 ± 3.6 |

> inference time: batch_size=1, 100 images, cpu only. gpu times would be substantially faster (especially for vit).
> **efficientnet-b0** is the pareto optimum — highest accuracy, fewest params, lowest flops, fastest inference.

---

## statistical significance

**date:** 2026-09-13
**source:** paired t-test on 3 test-set macro f1 values per model pair.
**detail:** `results/significance.json`

### paired t-test p-values for test macro f1 (n=3 seeds)

| model pair | f1 a (mean) | f1 b (mean) | diff | p-value | significant? |
|---|---|---|---|---|---|
| resnet-50 vs efficientnet-b0 | 0.9522 | 0.9554 | −0.0031 | **0.0296** | **✓ p<0.05** |
| resnet-50 vs vit-b/16 | 0.9522 | 0.9476 | +0.0046 | 0.2074 | n.s. |
| resnet-50 vs hybrid | 0.9522 | 0.9470 | +0.0052 | 0.2481 | n.s. |
| efficientnet-b0 vs vit-b/16 | 0.9554 | 0.9476 | +0.0078 | 0.1180 | n.s. |
| efficientnet-b0 vs hybrid | 0.9554 | 0.9470 | +0.0083 | 0.1286 | n.s. |
| vit-b/16 vs hybrid | 0.9476 | 0.9470 | +0.0006 | 0.8118 | n.s. |

> note: n=3 seeds gives low statistical power (df=2). only the efficientnet-b0 vs resnet-50 difference (p=0.030) is significant. all other pairs are not distinguishable with n=3 — differences are small (< 1% absolute) and consistent in direction but not statistically confirmed.

---

## insights and observations

> running log of insights discovered during the project.

### 2026-09-10
- **project initialized.** datasets verified. kaggle has 7,200 images (4 classes, balanced). figshare has 3,064 images (3 classes, imbalanced).
- **figshare has no "no tumor" class** — evaluation on figshare will use 3 classes only.
- **provenance concern noted** — kaggle lists figshare as a source. overlap audit is required before any cross-dataset claims.