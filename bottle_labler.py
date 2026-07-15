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
    # below as a path relative to the sam2 package's config root:
    # "configs/sam2.1/sam2.1_hiera_s.yaml"

-----------------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------------

    python sam2_auto_segment.py \
        --input_dir /path/to/tray_images \
        --output_dir /path/to/yolo_dataset \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --model_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
        --class_id 0

Outputs (per input image `foo.jpg`):
    output_dir/images/foo.jpg                 (copied)
    output_dir/labels/foo.txt                 (YOLO OBB labels)
    output_dir/viz/foo.jpg                    (overlay for sanity-checking)

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
   Crop to the tray first (--crop_box, or --roi_json for a fixed-camera
   setup) so the point grid concentrates on the objects you care about,
   and raise points_per_side to 64-96.

   For a FIXED camera + a tray that's always roughly the same regular grid
   (this is your setup), skip the blind uniform point grid entirely: run
   --calibrate_grid once to click 3 points (a cap, its right neighbor, its
   bottom neighbor) on a reference photo, then pass --grid_predict
   --grid_calib_json <that file> to prompt SAM2 exactly once per expected
   slot instead of hoping a uniform grid happens to land on every cap. A
   slot with no bottle (partial/empty tray) simply scores below
   pred_iou_thresh and is skipped -- no need to know which slots are filled.

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
    p.add_argument("--input_dir", help="Folder of tray photos (.jpg/.png)")
    p.add_argument("--output_dir", help="Where to write images/labels/viz")
    p.add_argument("--checkpoint", help="Path to SAM2 .pt checkpoint")
    p.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml",
                   help="SAM2 model config name, as a path relative to the sam2 package's config "
                        "root (Hydra searches from inside the installed sam2 package).")
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
    p.add_argument("--roi_json", default=None,
                   help="Path to a JSON file with a fixed ROI as {'x1':.., 'y1':.., 'x2':.., 'y2':..}. "
                        "For a fixed camera this is the recommended way to set the crop; "
                        "takes precedence over --crop_box if both are given.")
    p.add_argument("--grid_predict", action="store_true",
                   help="Use a calibrated grid of point prompts (see --calibrate_grid) fed one at a "
                        "time to SAM2's point-prompted predictor, instead of the automatic mask "
                        "generator's blind uniform point grid. Use this for a fixed camera + "
                        "regularly-spaced tray, where a missed cap is costlier than a slightly-off box.")
    p.add_argument("--grid_calib_json", default=None,
                   help="Path to the calibration JSON produced by --calibrate_grid (origin/row_vec/"
                        "col_vec/n_rows/n_cols, in ROI-local pixel coords). Required with --grid_predict.")
    p.add_argument("--calibrate_grid", action="store_true",
                   help="Interactive one-time setup: click 3 points on a reference image (a cap, its "
                        "right neighbor, its bottom neighbor) to define the tray's point grid, then "
                        "save it to --calib_output and exit. Does not need --checkpoint/--input_dir/"
                        "--output_dir. Re-run this if the camera or tray fixture changes.")
    p.add_argument("--ref_image", default=None,
                   help="Reference image to click points on. Used with --calibrate_grid.")
    p.add_argument("--n_rows", type=int, default=None, help="Bottle rows in the tray (--calibrate_grid).")
    p.add_argument("--n_cols", type=int, default=None, help="Bottle columns in the tray (--calibrate_grid).")
    p.add_argument("--calib_output", default="grid_calib.json",
                   help="Output path for the grid calibration JSON (--calibrate_grid).")
    p.add_argument("--bbox_only", action="store_true",
                   help="Write plain axis-aligned YOLO detection labels instead of OBB.")
    p.add_argument("--seg", action="store_true",
                   help="Write YOLO polygon segmentation labels instead of OBB.")
    p.add_argument("--poly_epsilon_ratio", type=float, default=0.004,
                   help="cv2.approxPolyDP epsilon as a fraction of contour perimeter, "
                        "to simplify mask polygons for the label file.")
    return p.parse_args()


def load_crop_box(args):
    """--roi_json (fixed-camera ROI file) takes precedence over --crop_box."""
    if args.roi_json:
        with open(args.roi_json) as f:
            d = json.load(f)
        return [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])]
    if args.crop_box:
        return list(args.crop_box)
    return None


def calibrate_grid(args):
    """
    Interactive one-time setup for --grid_predict: click a cap's center, its
    right neighbor, and its bottom neighbor on a reference image. The two
    resulting offset vectors encode both spacing and any tray rotation/tilt,
    so the full n_rows x n_cols grid can be extrapolated from just these 3
    points without assuming the camera is perfectly perpendicular to the tray.
    """
    img = cv2.imread(args.ref_image)
    if img is None:
        print(f"Could not read {args.ref_image}", file=sys.stderr)
        sys.exit(1)

    crop_box = load_crop_box(args)
    proc = img
    if crop_box:
        x1, y1, x2, y2 = crop_box
        proc = img[y1:y2, x1:x2]

    points = []
    window = "Click: 1) a cap center  2) its RIGHT neighbor  3) its BOTTOM neighbor  (r=reset, Enter=confirm, Esc=cancel)"

    def redraw():
        display = proc.copy()
        for i, p in enumerate(points):
            cv2.circle(display, p, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(points) >= 2:
            cv2.line(display, points[0], points[1], (255, 0, 0), 1)
        if len(points) >= 3:
            cv2.line(display, points[0], points[2], (0, 255, 255), 1)
        cv2.imshow(window, display)

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))
            redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
            redraw()
        elif key in (13, 10) and len(points) == 3:
            break
        elif key == 27:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    origin = np.array(points[0], dtype=np.float64)
    col_vec = (np.array(points[1], dtype=np.float64) - origin).tolist()
    row_vec = (np.array(points[2], dtype=np.float64) - origin).tolist()

    calib = {
        "origin": origin.tolist(),
        "col_vec": col_vec,
        "row_vec": row_vec,
        "n_rows": args.n_rows,
        "n_cols": args.n_cols,
        "note": "coordinates are relative to the cropped ROI (roi_json/crop_box), not the full image",
    }
    with open(args.calib_output, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved grid calibration to {args.calib_output}:")
    print(json.dumps(calib, indent=2))


def load_grid_points(grid_calib_json):
    """Extrapolate the full n_rows x n_cols grid from a --calibrate_grid JSON."""
    with open(grid_calib_json) as f:
        calib = json.load(f)
    origin = np.array(calib["origin"], dtype=np.float64)
    col_vec = np.array(calib["col_vec"], dtype=np.float64)
    row_vec = np.array(calib["row_vec"], dtype=np.float64)
    points = [
        origin + j * col_vec + i * row_vec
        for i in range(calib["n_rows"])
        for j in range(calib["n_cols"])
    ]
    return np.array(points)


def build_predictor(args):
    """Lazily import torch/sam2 so --help and --calibrate_grid work without them installed."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    return SAM2ImagePredictor(sam2_model)


def generate_grid_masks(predictor, img_rgb, grid_points, args):
    """
    Prompt SAM2 once per calibrated grid slot instead of a blind uniform grid.
    A slot with no bottle simply won't score above pred_iou_thresh and is
    dropped here -- this is what makes partial/empty trays work automatically,
    with no need to know in advance which slots are occupied.
    """
    predictor.set_image(img_rgb)
    masks = []
    for pt in grid_points:
        point_coords = np.array([pt])
        point_labels = np.array([1])
        pred_masks, scores, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True
        )
        best = int(np.argmax(scores))
        if scores[best] < args.pred_iou_thresh:
            continue  # empty slot: no bottle under this point

        seg = pred_masks[best].astype(bool)
        area = int(seg.sum())
        if area < args.min_mask_region_area:
            continue

        ys, xs = np.where(seg)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
        masks.append({"segmentation": seg, "area": area, "bbox": bbox})

    return masks


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

    if args.calibrate_grid:
        if not args.ref_image or not args.n_rows or not args.n_cols:
            print("--calibrate_grid requires --ref_image, --n_rows, and --n_cols", file=sys.stderr)
            sys.exit(1)
        calibrate_grid(args)
        return

    if not (args.input_dir and args.output_dir and args.checkpoint):
        print("--input_dir, --output_dir, and --checkpoint are required "
              "(unless using --calibrate_grid)", file=sys.stderr)
        sys.exit(1)
    if args.grid_predict and not args.grid_calib_json:
        print("--grid_predict requires --grid_calib_json (see --calibrate_grid)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "viz"), exist_ok=True)

    crop_box = load_crop_box(args)

    print("Loading SAM2 (this can take a minute)...")
    if args.grid_predict:
        predictor = build_predictor(args)
        grid_points = load_grid_points(args.grid_calib_json)
    else:
        mask_generator = build_mask_generator(args)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
    if not image_files:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    for fname in image_files:
        in_path = os.path.join(args.input_dir, fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"  [skip] could not read {fname}")
            continue

        offset_x, offset_y = 0, 0
        full_h, full_w = img_bgr.shape[:2]
        proc_img = img_bgr
        if crop_box:
            x1, y1, x2, y2 = crop_box
            proc_img = img_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        if args.grid_predict:
            raw_masks = generate_grid_masks(predictor, img_rgb, grid_points, args)
        else:
            raw_masks = mask_generator.generate(img_rgb)
        kept = filter_masks(raw_masks, args)

        # Shift mask coordinates back to full-image space if we cropped
        if crop_box:
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
        with open(os.path.join(args.output_dir, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(label_lines))

        shutil.copy2(in_path, os.path.join(args.output_dir, "images", fname))

        viz = draw_viz(img_bgr, kept)
        cv2.imwrite(os.path.join(args.output_dir, "viz", fname), viz)

    print("\nDone. Check the viz/ folder first on a handful of images before")
    if args.grid_predict:
        print("trusting the full batch — tune --min_area_ratio / --max_area_ratio /")
        print("--pred_iou_thresh, or re-run --calibrate_grid, based on what you see there.")
    else:
        print("trusting the full batch — tune --min_area_ratio / --max_area_ratio /")
        print("--points_per_side / --crop_box / --roi_json based on what you see there.")


if __name__ == "__main__":
    main()