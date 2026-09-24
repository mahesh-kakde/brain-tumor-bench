# xai methods: grad-cam, attention rollout, deletion faithfulness, perturbation stability
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2


# ─── grad-cam ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    computes grad-cam saliency maps for a given model and target layer.
    works for any layer that outputs a spatial feature map (h, w).
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._features    = None
        self._grads       = None
        self._fwd_hook    = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook    = target_layer.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, img_tensor, class_idx=None):
        """
        img_tensor: (1, 3, 224, 224)
        returns:    (224, 224) numpy heatmap in [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)

        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # global average pool gradients over spatial dims
        weights = self._grads.mean(dim=(2, 3), keepdim=True)        # (1, c, 1, 1)
        cam     = (weights * self._features).sum(dim=1, keepdim=True) # (1, 1, h, w)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_layer(model, model_name):
    """return the target layer for grad-cam given model name."""
    if model_name == 'resnet50':
        return model.layer4[-1].conv3      # last bottleneck conv in layer4
    elif model_name == 'efficientnet_b0':
        # hook the whole last conv block output (7×7×1280 spatial map)
        return model.features[-1]
    elif model_name == 'hybrid':
        return model.cnn[-1][-1].conv3     # resnet layer4 last bottleneck
    else:
        raise ValueError(f'grad-cam not applicable for {model_name}')


# ─── attention rollout ────────────────────────────────────────────────────────

class AttentionRollout:
    """
    attention rollout for vit-style models.
    torchvision ViT calls self_attention with need_weights=False internally,
    so we monkeypatch the MHA forward to force need_weights=True and capture weights.
    """

    def __init__(self, model, model_name):
        self.model        = model
        self.model_name   = model_name
        self._attentions  = []
        self._attn_mods   = []
        self._orig_fwds   = []
        self._patch()

    def _patch(self):
        """replace each MHA forward with a wrapper that forces need_weights=True."""
        if self.model_name == 'vit_b16':
            attn_mods = [layer.self_attention for layer in self.model.encoder.layers]
        elif self.model_name == 'hybrid':
            attn_mods = [layer.self_attn for layer in self.model.transformer.layers]
        else:
            return

        self._attn_mods = attn_mods
        for mod in attn_mods:
            orig = mod.forward
            self._orig_fwds.append(orig)

            def make_patched(original_forward):
                def patched(query, key, value, **kwargs):
                    # override whatever the caller passed
                    kwargs['need_weights']        = True
                    kwargs['average_attn_weights'] = False
                    out, weights = original_forward(query, key, value, **kwargs)
                    if weights is not None:
                        self._attentions.append(weights.detach().cpu())
                    return out, weights
                return patched

            mod.forward = make_patched(orig)

    def __call__(self, img_tensor):
        """
        img_tensor: (1, 3, 224, 224)
        returns:    (224, 224) numpy rollout heatmap in [0, 1]
        """
        self.model.eval()
        self._attentions = []

        with torch.no_grad():
            _ = self.model(img_tensor)

        if not self._attentions:
            size = 14 if self.model_name == 'vit_b16' else 7
            return np.ones((size, size), dtype=np.float32) / (size * size)

        # rollout: multiply attention maps layer-by-layer
        # each entry shape: (1, heads, seq, seq) — average heads first
        rollout = None
        for attn in self._attentions:
            a = attn.mean(dim=1).squeeze(0).numpy()  # (seq, seq)
            # residual connection: 0.5·A + 0.5·I
            a = 0.5 * a + 0.5 * np.eye(a.shape[0])
            a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
            rollout = a if rollout is None else rollout @ a

        if rollout is None:
            size = 14 if self.model_name == 'vit_b16' else 7
            return np.ones((size, size), dtype=np.float32) / (size * size)

        # [cls] → patch attention: row 0, skip cls token col (col 0)
        # vit_b16: seq=197 (1 cls + 196 patches = 14×14)
        # hybrid:  seq=50  (1 cls + 49 patches  = 7×7)
        cls_attn = rollout[0, 1:]
        grid_size = 14 if self.model_name == 'vit_b16' else 7
        heatmap   = cls_attn.reshape(grid_size, grid_size)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.astype(np.float32)

    def remove(self):
        """restore original MHA forwards."""
        for mod, orig in zip(self._attn_mods, self._orig_fwds):
            mod.forward = orig


def rollout_to_224(heatmap_small):
    """upsample a (14,14) or (7,7) rollout map to (224,224)."""
    return cv2.resize(heatmap_small, (224, 224), interpolation=cv2.INTER_LINEAR)


# ─── unified saliency map interface ───────────────────────────────────────────

def get_saliency_map(model, model_name, img_tensor, class_idx=None):
    """
    returns a (224, 224) numpy heatmap in [0, 1].
    grad-cam for cnn-based models, attention rollout for vit/hybrid.
    """
    if model_name in ('resnet50', 'efficientnet_b0'):
        layer = get_gradcam_layer(model, model_name)
        cam   = GradCAM(model, layer)
        hmap  = cam(img_tensor, class_idx)
        cam.remove()
        return hmap

    elif model_name == 'vit_b16':
        rollout = AttentionRollout(model, model_name)
        hmap    = rollout_to_224(rollout(img_tensor))
        rollout.remove()
        return hmap

    elif model_name == 'hybrid':
        # grad-cam on cnn branch (primary spatial saliency)
        layer = get_gradcam_layer(model, model_name)
        cam   = GradCAM(model, layer)
        hmap  = cam(img_tensor, class_idx)
        cam.remove()
        return hmap

    else:
        raise ValueError(f'unknown model: {model_name}')


# ─── heatmap overlay ──────────────────────────────────────────────────────────

def overlay_heatmap(img_tensor, heatmap, alpha=0.4):
    """
    blends a saliency heatmap onto the original image.
    img_tensor: (1, 3, 224, 224) imagenet-normalised
    heatmap:    (224, 224) in [0, 1]
    returns:    (224, 224, 3) uint8 rgb image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)
    img  = (img * 255).astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return (alpha * heatmap_rgb + (1 - alpha) * img).astype(np.uint8)


# ─── deletion faithfulness (del-auc) ─────────────────────────────────────────

def deletion_auc(model, img_tensor, heatmap, class_idx, steps=10, device='cpu'):
    """
    progressively masks most-salient pixels (high → low) and measures
    how fast the correct-class probability drops.
    lower del-auc = more faithful saliency map.
    returns: float del-auc in [0, 1]
    """
    model.eval()
    flat_indices = np.argsort(heatmap.flatten())[::-1]
    n_pixels     = len(flat_indices)
    step_size    = max(1, n_pixels // steps)

    probs    = []
    img_base = img_tensor.squeeze(0).clone()

    for step in range(steps + 1):
        n_masked = step * step_size
        masked   = img_base.clone()
        if n_masked > 0:
            rows, cols = np.unravel_index(flat_indices[:n_masked], (224, 224))
            masked[:, rows, cols] = 0.0
        with torch.no_grad():
            logit = model(masked.unsqueeze(0).to(device))
            prob  = F.softmax(logit, dim=1)[0, class_idx].item()
        probs.append(prob)

    x   = np.linspace(0, 1, steps + 1)
    auc = float(np.trapezoid(probs, x))
    return auc


# ─── perturbation stability (stab) ────────────────────────────────────────────

def perturbation_stability(model, model_name, img_tensor, n_trials=10, sigma=0.02, device='cpu'):
    """
    adds gaussian noise (sigma=0.02) n_trials times, generates a saliency map
    each time, and measures mean spearman rank correlation with the original map.
    higher stab = more consistent saliency.
    returns: float mean spearman correlation in [-1, 1]
    """
    import warnings
    from scipy.stats import spearmanr

    model.eval()
    base_hmap = get_saliency_map(model, model_name, img_tensor.to(device))
    base_flat = base_hmap.flatten()

    # skip if base map is constant (undefined correlation)
    if base_flat.std() < 1e-6:
        return float('nan')

    correlations = []
    for _ in range(n_trials):
        noise      = torch.randn_like(img_tensor) * sigma
        noisy_img  = (img_tensor + noise).to(device)
        noisy_hmap = get_saliency_map(model, model_name, noisy_img)
        noisy_flat = noisy_hmap.flatten()
        if noisy_flat.std() < 1e-6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corr, _ = spearmanr(base_flat, noisy_flat)
        if not np.isnan(corr):
            correlations.append(float(corr))

    return float(np.mean(correlations)) if correlations else float('nan')