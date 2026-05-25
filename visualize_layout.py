"""
Visualize SemLayoutDiff output as a clean 2D floor plan with furniture labels.

Usage:
    python visualize_layout.py --sample_dir sample --room_type bedroom
    python visualize_layout.py --sample_dir /kaggle/working/my_results/test_bedroom --room_type bedroom
"""

import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from scipy import ndimage
from pathlib import Path
import glob
import os


# ── Load metadata ─────────────────────────────────────────────────────────────

METADATA_DIR = "preprocess/metadata"

with open(f"{METADATA_DIR}/unified_idx_to_generic_label.json") as f:
    IDX_TO_LABEL = json.load(f)          # {"0": "void", "1": "floor", ...}

with open(f"{METADATA_DIR}/color_palette.json") as f:
    COLOR_PALETTE = json.load(f)         # {"floor": [211,211,211], ...}

# Arch floor plan IDs (from load_custom_floor_plan w_arch mode)
ARCH_IDS = {
    0: ("background", [255, 255, 255]),
    1: ("floor",      [211, 211, 211]),
    2: ("door",       [153,   0,   0]),
    3: ("window",     [255, 153, 153]),
}

# Labels to skip when adding text (non-furniture)
SKIP_LABELS = {"void", "floor", "door", "window", "pendant_lamp", "ceiling_lamp"}

DISPLAY_NAMES = {
    "kids_bed": "Kids Bed", "single_bed": "Single Bed", "double_bed": "Double Bed",
    "corner_side_table": "Side Table", "round_end_table": "End Table",
    "coffee_table": "Coffee Table", "console_table": "Console", "tv_stand": "TV Stand",
    "desk": "Desk", "dressing_table": "Dressing Table", "table": "Table",
    "dining_table": "Dining Table", "stool": "Stool", "dressing_chair": "Dressing Chair",
    "dining_chair": "Dining Chair", "chinese_chair": "Chinese Chair",
    "armchair": "Armchair", "chair": "Chair", "lounge_chair": "Lounge Chair",
    "loveseat_sofa": "Loveseat", "lazy_sofa": "Lazy Sofa", "sofa": "Sofa",
    "multi_seat_sofa": "Multi Sofa", "chaise_longue_sofa": "Chaise Longue",
    "l_shaped_sofa": "L-Shape Sofa", "nightstand": "Nightstand",
    "shelf": "Shelf", "bookshelf": "Bookshelf", "children_cabinet": "Kids Cabinet",
    "wine_cabinet": "Wine Cabinet", "cabinet": "Cabinet", "wardrobe": "Wardrobe",
    "pendant_lamp": "Lamp", "ceiling_lamp": "Ceiling Lamp",
    "door": "Door", "window": "Window",
}


def get_color(cat_id: int):
    """Return (R, G, B) 0-255 for a category ID."""
    if cat_id in ARCH_IDS:
        return ARCH_IDS[cat_id][1]
    label = IDX_TO_LABEL.get(str(cat_id), "void")
    return COLOR_PALETTE.get(label, [180, 180, 180])


def visualize_sample(label_path: str, output_path: str, title: str = "Floor Plan Layout"):
    """
    Core visualization: reads a composite_label.png and produces a
    clean annotated 2D floor plan.
    """
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        raise FileNotFoundError(f"Cannot read: {label_path}")

    H, W = label_img.shape
    SCALE = 6                                    # 120×120 → 720×720
    label_large = cv2.resize(
        label_img, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST
    )
    LH, LW = label_large.shape

    # ── Build colored canvas ──────────────────────────────────────────────────
    canvas = np.ones((LH, LW, 3), dtype=np.uint8) * 255   # white background

    unique_ids = np.unique(label_large)
    for cat_id in unique_ids:
        color = get_color(int(cat_id))
        mask = label_large == cat_id
        canvas[mask] = color

    # ── Draw room boundary (thick black wall) ─────────────────────────────────
    room_mask = (label_large >= 1).astype(np.uint8)        # everything inside room
    contours, _ = cv2.findContours(
        room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), thickness=4)

    # ── Draw thin outlines around each furniture region ───────────────────────
    for cat_id in unique_ids:
        if cat_id <= 1:
            continue
        mask = (label_large == cat_id).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (50, 50, 50), thickness=1)

    # ── Draw door arcs ────────────────────────────────────────────────────────
    door_mask = (label_large == 36).astype(np.uint8)       # semantic door = 36
    # Also check arch door value = 2 (floor plan without semantic overwrite)
    arch_door_mask = (label_large == 2).astype(np.uint8)
    combined_door = cv2.bitwise_or(door_mask, arch_door_mask)
    door_cnts, _ = cv2.findContours(combined_door, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in door_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        # Draw a small arc to indicate door swing
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.ellipse(canvas, center, (radius, radius), 0, 0, 90, (80, 0, 0), 2)
        cv2.line(canvas, center, (center[0] + radius, center[1]), (80, 0, 0), 2)

    # ── Plot with matplotlib for text labels & legend ─────────────────────────
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(canvas)
    ax.set_xlim(0, LW)
    ax.set_ylim(LH, 0)

    # Add furniture labels at centroid of each connected component
    legend_items = {}                            # label_name → color

    for cat_id in unique_ids:
        cat_id = int(cat_id)
        label_name = IDX_TO_LABEL.get(str(cat_id), "void")

        # Add to legend for all visible categories
        if label_name not in ("void",):
            color_255 = get_color(cat_id)
            legend_items[label_name] = [c / 255 for c in color_255]

        if label_name in SKIP_LABELS:
            continue

        mask = (label_large == cat_id)
        labeled_arr, n = ndimage.label(mask)

        for comp in range(1, n + 1):
            comp_mask = labeled_arr == comp
            pixel_count = comp_mask.sum()
            if pixel_count < 80:               # skip tiny noise
                continue

            ys, xs = np.where(comp_mask)
            cy, cx = ys.mean(), xs.mean()

            display = DISPLAY_NAMES.get(label_name, label_name.replace("_", " ").title())

            ax.text(
                cx, cy, display,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="black",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white", alpha=0.75,
                    edgecolor="gray", linewidth=0.8,
                ),
                zorder=5,
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="gray",
            linewidth=0.5,
            label=DISPLAY_NAMES.get(lbl, lbl.replace("_", " ").title()),
        )
        for lbl, color in sorted(legend_items.items())
        if lbl not in ("void",)
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Legend",
        title_fontsize=10,
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="gray",
    )

    # ── Scale bar (assuming 120px = 12m, so 1px = 0.1m, SCALE=6 → 6px = 0.1m) ──
    meter_in_px = SCALE * 10        # 10 original px = 1 m
    bar_x, bar_y = 20, LH - 25
    ax.plot([bar_x, bar_x + meter_in_px], [bar_y, bar_y], "k-", linewidth=3)
    ax.plot([bar_x, bar_x], [bar_y - 5, bar_y + 5], "k-", linewidth=2)
    ax.plot([bar_x + meter_in_px, bar_x + meter_in_px], [bar_y - 5, bar_y + 5], "k-", linewidth=2)
    ax.text(bar_x + meter_in_px / 2, bar_y - 12, "1 m", ha="center", fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.show()
    plt.close()


def run(sample_dir: str, room_type: str):
    """Process all composite_label.png files in a directory."""
    sample_dir = Path(sample_dir)
    label_files = sorted(sample_dir.glob("*composite_label*.png"))

    if not label_files:
        # Fallback: just label files
        label_files = sorted(sample_dir.glob("*label*.png"))

    if not label_files:
        print(f"❌ No label PNG files found in {sample_dir}")
        return

    print(f"Found {len(label_files)} label file(s) in {sample_dir}")

    for label_path in label_files:
        stem = label_path.stem.replace("_composite_label", "").replace("_label", "")
        out_path = sample_dir / f"{stem}_floorplan_2d.png"
        title = f"2D Floor Plan — {room_type.title()} ({stem})"
        try:
            visualize_sample(str(label_path), str(out_path), title)
        except Exception as e:
            print(f"⚠️  Skipped {label_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir",  default="sample",   help="Folder chứa output samples")
    parser.add_argument("--room_type",   default="bedroom",  help="bedroom / livingroom / diningroom")
    args = parser.parse_args()

    run(args.sample_dir, args.room_type)
