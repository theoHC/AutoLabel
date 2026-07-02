#!/usr/bin/env python3
"""
sam2_auto_segment.py

Segment densely-packed, near-identical objects (bottles/caps on a tray) into
individual instances using SAM2's automatic mask generator (no text prompt),
then filter the resulting masks by size/shape to drop background and
merged/"whole tray" blobs, and export YOLO-format labels + a visualization.

This deliberately skips Grounding DINO. For a tight, repeated grid of
visually-identical objects, text-grounded detection tends to either miss
everything (semantic mismatch) or collapse overlapping boxes via NMS into a
single box covering the whole tray. SAM2's automatic mask generator instead
seeds a dense grid of points across the whole image and asks "is there a
distinct object here", which finds every blob without needing to name it.
We then use area/shape filtering (informed by looking at your tray) to keep
only bottle-cap-sized masks and discard the background/tray/table masks.

-----------------------------------------------------------------------------
SETUP (run this on a machine with GPU + internet access to Meta's CDN;
downloading the checkpoint will NOT work from a network-restricted sandbox)
-----------------------------------------------------------------------------

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install sam2 opencv-python numpy pillow

    # Download a checkpoint (pick one; "small" is a good speed/quality tradeoff
    # for small dense objects since it has finer feature resolution than "tiny"):
    mkdir -p checkpoints
    curl -L -o checkpoints/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

    # The matching model config ships with the sam2 pip package, referenced
    # below by name: "sam2.1_hiera_s.yaml"

-----------------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------------

    python sam2_auto_segment.py \
        --input_dir /path/to/tray_images \
        --output_dir /path/to/yolo_dataset \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --model_cfg sam2.1_hiera_s.yaml \
        --class_id 0

Outputs:
    output_dir/images/train/*.jpg, images/val/*.jpg, (images/test/*.jpg)
    output_dir/labels/train/*.txt, labels/val/*.txt, (labels/test/*.txt)
    output_dir/viz/*.jpg                      (overlay for sanity-checking, all images together)
    output_dir/data.yaml                      (Ultralytics dataset config, ready to train against)

Images are deterministically shuffled (--split_seed) and split by
--val_ratio / --test_ratio (default 15% val, 0% test — pass --test_ratio
to add one). Shuffling matters because sequentially-captured frames of the
same tray are highly correlated; a naive first-N/last-N split would leak
near-duplicate frames between train and val.

Default label format is YOLO-OBB (Ultralytics):
    <class_id> x1 y1 x2 y2 x3 y3 x4 y4          (all coords normalized 0-1)
where (x1,y1)..(x4,y4) are the 4 corners of the minimum-area rotated
rectangle fit to each mask, ordered clockwise starting from top-left of the
rectangle. This is the correct choice here since your caps sit in a grid
that's often slightly rotated relative to the image axes, and OBB fits their
true footprint much tighter than an axis-aligned box would.

Pass --bbox_only for plain axis-aligned YOLO detection labels instead
(<class_id> cx cy w h), or --seg for polygon segmentation labels instead.

-----------------------------------------------------------------------------
TUNING NOTES (read this before you run it on your real folder)
-----------------------------------------------------------------------------

1. points_per_side: how densely SAM2 seeds candidate points. Your bottles
   look like they're ~35-40px apart in the uploaded photo. Default SAM2
   automatic mask generator uses points_per_side=32 across the WHOLE image,
   which is far too sparse for a tray that only occupies part of the frame.
   Crop to the tray first (--crop_box, or run --interactive_crop once to
   click it) so the point grid concentrates on the objects you care about,
   and raise points_per_side to 64-96.

2. area filtering: this is the key knob. After generating all masks, we
   compute the area of every mask, take the MEDIAN area of masks in a
   plausible size band, and keep only masks within
   [min_area_ratio, max_area_ratio] of that median. This throws out the
   one giant "whole tray" mask and any tiny noise specks automatically,
   without you having to hardcode pixel counts per photo/resolution.

3. stability_score_thresh / pred_iou_thresh: raised above SAM2 defaults
   here (0.92/0.88) because with objects this close together, low-confidence
   masks are usually two bottles fused together, not partial real objects.

4. This script is NOT expected to be perfect out of the box. Run it on 3-5
   images first, check output_dir/viz/, and adjust --min_area_ratio /
   --max_area_ratio / --points_per_side before batch processing everything.
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, help="Folder of tray photos (.jpg/.png)")
    p.add_argument("--output_dir", required=True, help="Where to write images/labels/viz")
    p.add_argument("--checkpoint", required=True, help="Path to SAM2 .pt checkpoint")
    p.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml", help="SAM2 model config name")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--class_id", type=int, default=0, help="YOLO class id to assign to every bottle")
    p.add_argument("--points_per_side", type=int, default=64,
                   help="Density of SAM2's automatic point grid. Raise for smaller/denser objects.")
    p.add_argument("--pred_iou_thresh", type=float, default=0.88)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)
    p.add_argument("--min_mask_region_area", type=int, default=50,
                   help="Drop tiny noise masks below this pixel area before size filtering.")
    p.add_argument("--min_area_ratio", type=float, default=0.4,
                   help="Keep masks with area >= min_area_ratio * median_object_area")
    p.add_argument("--max_area_ratio", type=float, default=2.5,
                   help="Keep masks with area <= max_area_ratio * median_object_area")
    p.add_argument("--max_solidity_reject", type=float, default=0.5,
                   help="Reject masks whose (area / convex_hull_area) is below this — "
                        "catches weird fused blobs of 2+ touching bottles.")
    p.add_argument("--crop_box", type=int, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Optional pixel box to crop to the tray before segmenting, "
                        "e.g. --crop_box 300 60 720 400. Strongly recommended.")
    p.add_argument("--bbox_only", action="store_true",
                   help="Write plain axis-aligned YOLO detection labels instead of OBB.")
    p.add_argument("--seg", action="store_true",
                   help="Write YOLO polygon segmentation labels instead of OBB.")
    p.add_argument("--poly_epsilon_ratio", type=float, default=0.004,
                   help="cv2.approxPolyDP epsilon as a fraction of contour perimeter, "
                        "to simplify mask polygons for the label file.")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Fraction of images assigned to the val split.")
    p.add_argument("--test_ratio", type=float, default=0.0,
                   help="Fraction of images assigned to the test split (0 = no test split).")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for shuffling images before splitting, for reproducibility.")
    p.add_argument("--class_name", default="bottle",
                   help="Human-readable class name written into data.yaml.")
    return p.parse_args()


def build_mask_generator(args):
    """Lazily import torch/sam2 so --help works without them installed."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        # crop_n_layers=0 keeps this to a single full-image pass; raise to 1
        # if bottles near the crop edges get missed, at the cost of speed.
        crop_n_layers=0,
        box_nms_thresh=0.7,
    )
    return mask_generator


def filter_masks(masks, args):
    """
    masks: list of dicts from SAM2AutomaticMaskGenerator, each with
           'segmentation' (bool HxW array), 'area', etc.
    Returns the subset that look like individual bottles.
    """
    if not masks:
        return []

    areas = np.array([m["area"] for m in masks], dtype=np.float64)

    # Robust "typical object size" estimate: median of the middle 60% of
    # areas, which is fairly insensitive to a few huge (tray) or tiny
    # (noise) outlier masks.
    sorted_areas = np.sort(areas)
    lo = int(len(sorted_areas) * 0.2)
    hi = int(len(sorted_areas) * 0.8) or 1
    trimmed = sorted_areas[lo:hi] if hi > lo else sorted_areas
    median_area = float(np.median(trimmed))

    kept = []
    for m in masks:
        area = m["area"]
        if not (args.min_area_ratio * median_area <= area <= args.max_area_ratio * median_area):
            continue

        seg = m["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = (cv2.contourArea(contour) / hull_area) if hull_area > 0 else 0
        if solidity < args.max_solidity_reject:
            # low solidity = irregular/concave blob, likely 2+ fused bottles
            # or a partial edge artifact rather than one clean round object
            continue

        m["_contour"] = contour
        kept.append(m)

    # De-duplicate near-identical overlapping masks (keep the larger one)
    kept.sort(key=lambda m: m["area"], reverse=True)
    final = []
    for m in kept:
        seg = m["segmentation"]
        is_dup = False
        for f in final:
            inter = np.logical_and(seg, f["segmentation"]).sum()
            union = np.logical_or(seg, f["segmentation"]).sum()
            iou = inter / union if union > 0 else 0
            if iou > 0.7:
                is_dup = True
                break
        if not is_dup:
            final.append(m)

    return final


def _order_corners_clockwise(pts):
    """
    Order 4 points clockwise starting from the top-left-most corner, so
    labels are consistent across objects/images (Ultralytics doesn't
    strictly require a fixed starting corner, but consistent ordering
    avoids edge-case ambiguity for near-square boxes and makes the labels
    easier to sanity-check visually).
    """
    pts = pts[np.argsort(pts[:, 1])]  # sort by y
    top_two = pts[:2][np.argsort(pts[:2, 0])]
    bottom_two = pts[2:][np.argsort(pts[2:, 0])[::-1]]
    ordered = np.vstack([top_two, bottom_two])  # TL, TR, BR, BL
    return ordered


def mask_to_yolo_obb(mask_dict, img_w, img_h, class_id):
    """
    Fit a minimum-area rotated rectangle to the mask's contour and emit it
    as a YOLO-OBB label: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized).
    """
    contour = mask_dict["_contour"]
    rect = cv2.minAreaRect(contour)          # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)                # 4x2 float array, pixel coords
    box = _order_corners_clockwise(box)

    box = box.astype(np.float64)
    box[:, 0] = np.clip(box[:, 0] / img_w, 0.0, 1.0)
    box[:, 1] = np.clip(box[:, 1] / img_h, 0.0, 1.0)

    flat = " ".join(f"{v:.6f}" for v in box.flatten())
    return f"{class_id} {flat}"


def mask_to_yolo_polygon(mask_dict, img_w, img_h, class_id, epsilon_ratio):
    contour = mask_dict["_contour"]
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    if len(approx) < 3:
        approx = contour  # fallback to full contour if polygon collapsed
    coords = approx.reshape(-1, 2).astype(np.float64)
    coords[:, 0] /= img_w
    coords[:, 1] /= img_h
    coords = np.clip(coords, 0.0, 1.0)
    flat = " ".join(f"{v:.6f}" for v in coords.flatten())
    return f"{class_id} {flat}"


def mask_to_yolo_bbox(mask_dict, img_w, img_h, class_id):
    x, y, w, h = mask_dict["bbox"]  # SAM2 gives XYWH in pixels
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def assign_splits(image_files, val_ratio, test_ratio, seed):
    """
    Deterministically shuffle and assign each filename to 'train', 'val', or
    'test'. Shuffling (rather than a straight slice) matters here because
    your images were likely captured in sequence (similar tray, similar
    lighting, similar frame-to-frame), so a naive first-N/last-N split could
    put near-duplicate frames only in train or only in val and give you a
    misleadingly optimistic (or pessimistic) val score.
    """
    files = list(image_files)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    n = len(files)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_val = min(n_val, n)
    n_test = min(n_test, n - n_val)

    val_set = set(files[:n_val])
    test_set = set(files[n_val:n_val + n_test])
    split_map = {}
    for f in files:
        if f in val_set:
            split_map[f] = "val"
        elif f in test_set:
            split_map[f] = "test"
        else:
            split_map[f] = "train"
    return split_map


def write_data_yaml(output_dir, class_name, has_test):
    lines = [
        f"path: {os.path.abspath(output_dir)}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.append("")
    lines.append("names:")
    lines.append(f"  0: {class_name}")
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write("\n".join(lines))


def draw_viz(img, masks, show_obb=True):
    overlay = img.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        color = rng.integers(0, 255, size=3).tolist()
        seg = m["segmentation"]
        overlay[seg] = (overlay[seg] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        if show_obb and "_contour" in m:
            rect = cv2.minAreaRect(m["_contour"])
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(overlay, [box], 0, color, 2)
        else:
            x, y, w, h = [int(v) for v in m["bbox"]]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.putText(overlay, f"{len(masks)} objects", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return overlay


def main():
    args = parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
    if not image_files:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    split_map = assign_splits(image_files, args.val_ratio, args.test_ratio, args.split_seed)
    has_test = args.test_ratio > 0
    splits_present = sorted(set(split_map.values()))
    for split in splits_present:
        os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "viz"), exist_ok=True)

    counts = {s: 0 for s in splits_present}

    print("Loading SAM2 (this can take a minute)...")
    mask_generator = build_mask_generator(args)

    for fname in image_files:
        in_path = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"  [skip] could not read {fname}")
            continue

        offset_x, offset_y = 0, 0
        full_h, full_w = img_bgr.shape[:2]
        proc_img = img_bgr
        if args.crop_box:
            x1, y1, x2, y2 = args.crop_box
            proc_img = img_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        raw_masks = mask_generator.generate(img_rgb)
        kept = filter_masks(raw_masks, args)

        # Shift mask coordinates back to full-image space if we cropped
        if args.crop_box:
            for m in kept:
                full_seg = np.zeros((full_h, full_w), dtype=bool)
                ph, pw = m["segmentation"].shape
                full_seg[offset_y:offset_y + ph, offset_x:offset_x + pw] = m["segmentation"]
                m["segmentation"] = full_seg
                bx, by, bw, bh = m["bbox"]
                m["bbox"] = [bx + offset_x, by + offset_y, bw, bh]
                m["_contour"] = m["_contour"] + np.array([offset_x, offset_y])

        print(f"{fname}: {len(raw_masks)} raw masks -> {len(kept)} kept after filtering")

        label_lines = []
        for m in kept:
            if args.bbox_only:
                label_lines.append(mask_to_yolo_bbox(m, full_w, full_h, args.class_id))
            elif args.seg:
                label_lines.append(mask_to_yolo_polygon(m, full_w, full_h, args.class_id, args.poly_epsilon_ratio))
            else:
                label_lines.append(mask_to_yolo_obb(m, full_w, full_h, args.class_id))

        stem = os.path.splitext(fname)[0]
        split = split_map[fname]
        counts[split] += 1
        with open(os.path.join(args.output_dir, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(label_lines))

        shutil.copy2(in_path, os.path.join(args.output_dir, "images", split, fname))

        viz = draw_viz(img_bgr, kept)
        cv2.imwrite(os.path.join(args.output_dir, "viz", fname), viz)

    write_data_yaml(args.output_dir, args.class_name, has_test)

    print("\nSplit summary:", ", ".join(f"{s}={n}" for s, n in counts.items()))
    print(f"Wrote {os.path.join(args.output_dir, 'data.yaml')}")
    print("\nDone. Check the viz/ folder first on a handful of images before")
    print("trusting the full batch — tune --min_area_ratio / --max_area_ratio /")
    print("--points_per_side / --crop_box based on what you see there.")


if __name__ == "__main__":
    main()