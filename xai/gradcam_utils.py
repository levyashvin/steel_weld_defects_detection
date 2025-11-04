"""
Grad-CAM utilities for LF-YOLO and Ultralytics models.

- Finds a suitable target conv layer (prefer model.reduce[0] or backbone.s10.fuse.conv).
- Computes Grad-CAM with improved score selection for object detectors.
- Provides YOLO-style preprocessing and visualization helpers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    """Return the last nn.Conv2d module inside `module` (best-effort)."""
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def pick_target_layer(model: nn.Module, min_channels: int = 64) -> nn.Module:
    """Pick a robust conv layer for Grad-CAM.

    Preference order:
      1) model.reduce[0] (Conv2d 6144->256 in this repo)
      2) model.backbone.s10.fuse.conv (richest spatial features)
      3) last Conv2d found in the model (fallback)
    """
    # 1) reduce conv
    try:
        layer = getattr(model, "reduce")[0]
        # Some Sequential blocks wrap conv at .conv
        if hasattr(layer, "conv") and isinstance(layer.conv, nn.Conv2d):
            layer = layer.conv
        if isinstance(layer, nn.Conv2d) and getattr(layer, "out_channels", 0) >= min_channels:
            return layer
    except Exception:
        pass

    # 2) backbone rich conv
    try:
        layer = model.backbone.s10.fuse.conv  # CBL(...).conv
        if isinstance(layer, nn.Conv2d) and getattr(layer, "out_channels", 0) >= min_channels:
            return layer
    except Exception:
        pass

    # 3) generic fallback
    # 3) search for the last Conv2d meeting minimum channels (and prefer k>=3)
    candidates: list[nn.Conv2d] = []
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            candidates.append(m)

    # prefer higher channel convs and spatial kernels
    for m in reversed(candidates):
        k = m.kernel_size if hasattr(m, "kernel_size") else (1, 1)
        if getattr(m, "out_channels", 0) >= min_channels and max(k) >= 3:
            return m

    # fallback: any conv with enough channels
    for m in reversed(candidates):
        if getattr(m, "out_channels", 0) >= min_channels:
            return m

    # last resort: any last conv
    layer = candidates[-1] if candidates else None
    if layer is None:
        raise RuntimeError("No Conv2d layer found to use for Grad-CAM.")
    return layer


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, score_mode: str = "auto", use_relu: bool = True):
        self.model = model
        self.target_layer = target_layer
        self.score_mode = score_mode  # 'auto' | 'logit' | 'sigmoid' | 'absmax'
        self.use_relu = use_relu
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        def fwd_hook(_: nn.Module, __, out: torch.Tensor):
            self.activations = out.detach()

        def bwd_hook(_: nn.Module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        # full_backward_hook available on newer PyTorch; fallback to backward_hook
        try:
            target_layer.register_full_backward_hook(bwd_hook)
        except Exception:
            target_layer.register_backward_hook(bwd_hook)

    def _choose_score(self, out: Union[torch.Tensor, Sequence]) -> torch.Tensor:
        """Choose a scalar score for backprop suitable for detectors.

        - If a tensor with last dim >= 5 exists, use objectness sigmoid().max().
        - Else if tensor, use sigmoid().max() as generic positive score.
        - Else fallback to abs().max().
        """

        # Tensor case
        if isinstance(out, torch.Tensor):
            if out.ndim >= 2 and out.shape[-1] >= 5:
                if self.score_mode == "logit":
                    return out[..., 4].max()
                elif self.score_mode == "sigmoid":
                    return out[..., 4].sigmoid().max()
                elif self.score_mode == "absmax":
                    return out[..., 4].abs().max()
                else:  # auto
                    return out[..., 4].max()
            # generic tensor
            if self.score_mode == "sigmoid":
                return out.sigmoid().max() if out.is_floating_point() else out.abs().max()
            elif self.score_mode == "absmax":
                return out.abs().max()
            else:  # auto/logit
                return out.max() if out.is_floating_point() else out.abs().max()

        # List/Tuple case
        if isinstance(out, (list, tuple)):
            # Prefer a YOLO-like detection tensor
            for o in out:
                if isinstance(o, torch.Tensor) and o.ndim >= 2 and o.shape[-1] >= 5:
                    if self.score_mode == "logit":
                        return o[..., 4].max()
                    elif self.score_mode == "sigmoid":
                        return o[..., 4].sigmoid().max()
                    elif self.score_mode == "absmax":
                        return o[..., 4].abs().max()
                    else:
                        return o[..., 4].max()
            # Fallback: any tensor
            for o in out:
                if isinstance(o, torch.Tensor):
                    if self.score_mode == "sigmoid":
                        return o.sigmoid().max() if o.is_floating_point() else o.abs().max()
                    elif self.score_mode == "absmax":
                        return o.abs().max()
                    else:  # auto/logit
                        return o.max() if o.is_floating_point() else o.abs().max()

        # Final fallback (should rarely happen)
        raise RuntimeError("Could not determine a scalar score to backprop from model outputs.")

    @torch.enable_grad()
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        score_tensor: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM for the given input.

        Args:
            input_tensor: Float tensor [1, C, H, W]
            score_tensor: Optional scalar tensor to backprop. If None, a score
                          is chosen automatically via _choose_score.
        Returns:
            CAM as a numpy array of shape [H, W], normalized 0..1
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor must be shape [1, C, H, W]")

        self.model.zero_grad(set_to_none=True)
        out = self.model(input_tensor)

        if score_tensor is None:
            score_tensor = self._choose_score(out)
        score_tensor.backward(retain_graph=True)

        acts = self.activations  # [1, C, h, w]
        grads = self.gradients   # [1, C, h, w]
        if acts is None or grads is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # Global-average-pool gradients -> weights per channel
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        if self.use_relu:
            cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _letterbox_params(orig_w: int, orig_h: int, size: int) -> Tuple[int, int, int, int, float]:
    """Compute letterbox resize and padding for square `size`.

    Returns: (new_w, new_h, pad_left, pad_top, scale)
    """
    scale = min(size / float(orig_w), size / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_w = size - new_w
    pad_h = size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return new_w, new_h, pad_left, pad_top, scale


def preprocess_image(
    path: Union[str, os.PathLike],
    img_size: int = 640,
    device: Union[str, torch.device] = "cpu",
    letterbox: bool = True,
) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Preprocess image to [1, 3, img_size, img_size] in [0,1] range.

    Returns:
        tensor, original PIL, meta dict with mapping info for unletterbox overlay.
    """
    img = Image.open(path).convert("RGB")
    orig = img.copy()
    W, H = img.size
    S = img_size

    if letterbox:
        new_w, new_h, pad_left, pad_top, scale = _letterbox_params(W, H, S)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (S, S), (0, 0, 0))
        canvas.paste(img, (pad_left, pad_top))
        img = canvas
        meta = dict(size=S, new_w=new_w, new_h=new_h, pad_left=pad_left, pad_top=pad_top, scale=scale, orig_w=W, orig_h=H)
    else:
        img = img.resize((S, S), Image.BILINEAR)
        meta = dict(size=S, new_w=S, new_h=S, pad_left=0, pad_top=0, scale=min(S / W, S / H), orig_w=W, orig_h=H)

    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, orig, meta


def _unletterbox_cam(cam_square: np.ndarray, meta: dict) -> np.ndarray:
    """Remove letterbox padding and resize CAM back to original image size."""
    S = meta["size"]
    new_w = meta["new_w"]
    new_h = meta["new_h"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    # Ensure shape
    if cam_square.shape[0] != S or cam_square.shape[1] != S:
        cam_square = np.array(Image.fromarray((cam_square * 255).astype("uint8")).resize((S, S), Image.BILINEAR)) / 255.0

    # Crop the letterboxed region
    cam_cropped = cam_square[pad_top: pad_top + new_h, pad_left: pad_left + new_w]
    # Resize back to original image size
    cam_orig = np.array(Image.fromarray((cam_cropped * 255).astype("uint8")).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
    return cam_orig


def overlay_and_show(
    orig_pil: Image.Image,
    cam_square: np.ndarray,
    meta: Optional[dict] = None,
    boxes: Optional[List[Sequence[float]]] = None,
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    alpha: float = 0.4,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Display original, heatmap, and overlay; optionally save overlay.

    Expects `cam_square` in 0..1. If `meta` provided (from letterbox preprocessing),
    unletterbox CAM before overlaying on the original image.
    """
    import matplotlib.pyplot as plt

    img = np.array(orig_pil).astype(np.float32) / 255.0

    if meta is not None:
        cam_resized = _unletterbox_cam(cam_square, meta)
    else:
        W, H = orig_pil.size
        cam_resized = np.array(Image.fromarray((cam_square * 255).astype("uint8")).resize((W, H), Image.BILINEAR)) / 255.0

    cmap = plt.colormaps["jet"]
    heatmap_rgb = cmap(cam_resized)[:, :, :3]
    overlay = img * (1 - alpha) + heatmap_rgb * alpha

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(cam_resized, cmap="jet")
    axs[1].set_title("Grad-CAM (normalized)")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")
    for ax in axs:
        ax.axis("off")

    # Draw boxes if provided
    if boxes:
        for ax in axs:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="lime", linewidth=2, fill=False)
                ax.add_patch(rect)
                if labels is not None and scores is not None and i < len(labels) and i < len(scores):
                    caption = f"{labels[i]} {scores[i]:.2f}"
                    ax.text(x1, max(0, y1 - 6), caption, fontsize=9, color="black",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    if save:
        try:
            Image.fromarray((overlay * 255).astype("uint8")).save(save)
        except Exception:
            # Fallback to matplotlib savefig of the overlay panel if direct save fails
            fig.savefig(save)
    if show:
        plt.show()
    else:
        plt.close(fig)
