"""
Run Grad-CAM on a YOLO/LF-YOLO model checkpoint and visualize heatmaps.

Usage:
  python gradcam_run.py --model model.pt --image image.jpg \
      [--imgsz 640] [--conf 0.25] [--device cuda] [--layer auto|reduce|backbone]

This script:
  - Loads your checkpoint via Ultralytics YOLO wrapper when possible (to get boxes easily)
  - Picks a robust conv layer for CAM (or as specified)
  - Computes Grad-CAM with better objectness score selection
  - Overlays CAM and detection boxes on the original image
  - Saves overlay as gradcam_overlay.png (by default)
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from PIL import Image

# Ensure custom class definitions are importable if needed for torch.load fallbacks
try:
    from lf_yolo import LFYOLO_WeldDefect  # noqa: F401
except Exception:
    pass

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

from xai.gradcam_utils import (
    GradCAM,
    overlay_and_show,
    pick_target_layer,
    preprocess_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM for YOLO/LF-YOLO checkpoints")
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (square)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing boxes")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device: cuda or cpu")
    p.add_argument("--layer", default="auto", choices=["auto", "reduce", "backbone"], help="Target layer selection")
    p.add_argument("--save", default="gradcam_overlay.png", help="Path to save overlay image")
    p.add_argument("--no-show", action="store_true", help="Do not display interactive plot")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Try to load with Ultralytics to get a YOLO wrapper for boxes
    yolo = None
    model = None

    if YOLO is not None:
        try:
            yolo = YOLO(args.model)
            model = yolo.model
            print("Loaded via Ultralytics YOLO wrapper.")
        except Exception as e:
            print(f"Could not load with YOLO wrapper: {e}")

    if model is None:
        # Fallback: pure torch load (boxes will be unavailable unless you implement decoding)
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        print("Loaded model via torch.load fallback.")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    # Choose target conv layer
    if args.layer == "reduce":
        try:
            target_layer = model.reduce[0]
            print(f"Using target layer: reduce[0] -> {target_layer}")
        except Exception as e:
            print(f"Failed to access reduce[0]: {e}. Falling back to auto pick.")
            target_layer = pick_target_layer(model)
    elif args.layer == "backbone":
        try:
            target_layer = model.backbone.s10.fuse.conv
            print(f"Using target layer: backbone.s10.fuse.conv -> {target_layer}")
        except Exception as e:
            print(f"Failed to access backbone.s10.fuse.conv: {e}. Falling back to auto pick.")
            target_layer = pick_target_layer(model)
    else:
        target_layer = pick_target_layer(model)
        print(f"Using target layer (auto): {target_layer}")

    # Preprocess
    input_tensor, orig_pil, meta = preprocess_image(args.image, img_size=args.imgsz, device=device)
    input_tensor.requires_grad_(True)

    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    print("Grad-CAM computed.")

    # Boxes via Ultralytics wrapper if available
    boxes = []
    labels = []
    scores = []
    if yolo is not None:
        with torch.no_grad():
            # Map device to Ultralytics device param
            dev_param: Optional[object]
            if device.type == "cuda":
                dev_param = 0  # first CUDA device
            else:
                dev_param = "cpu"
            res = yolo(args.image, conf=args.conf, verbose=False, device=dev_param)[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()
            # names dict can be on either results or model
            names = getattr(res, "names", None) or getattr(yolo.model, "names", None) or {}
            cls_list = res.boxes.cls.cpu().numpy().tolist()
            labels = [names.get(int(c), str(int(c))) for c in cls_list]
            print(f"Detections: {len(boxes)} box(es)")
        else:
            print("No boxes parsed from Ultralytics result (or none above conf threshold).")
    else:
        print("Ultralytics YOLO wrapper unavailable; skipping boxes overlay.")

    # Visualize/Save
    overlay_and_show(
        orig_pil,
        cam_square=cam,
        meta=meta,
        boxes=boxes,
        labels=labels,
        scores=scores,
        alpha=0.45,
        show=not args.no_show,
        save=args.save,
    )
    print(f"Saved overlay to: {os.path.abspath(args.save)}")


if __name__ == "__main__":
    main()

