"""
Visualize inputs của SemLayoutDiff để dùng trong thuyết trình.

Input của hệ thống gồm 2 thứ:
  1. Room type  : bedroom / livingroom / diningroom (số nguyên 0/1/2)
  2. Floor plan : ảnh grayscale 120×120 thể hiện hình dạng phòng
                  (0=ngoài, 1=sàn, 2=cửa ra vào, 3=cửa sổ)

Script này:
  - Đọc trực tiếp datasets/unified_w_arch_120x120.npy
  - Hiển thị N mẫu floor plan thực tế cho mỗi loại phòng
  - Tạo ảnh tổng hợp dạng "Input → Output" để trình bày

Usage:
    python visualize_inputs.py
    python visualize_inputs.py --npy_path datasets/unified_w_arch_120x120.npy --n_samples 4
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Màu cho từng class trong floor plan ───────────────────────────────────────
FLOOR_PLAN_COLORS = {
    0: (1.00, 1.00, 1.00),   # background  → trắng
    1: (0.83, 0.83, 0.83),   # floor       → xám nhạt
    2: (0.60, 0.00, 0.00),   # door        → đỏ sẫm
    3: (1.00, 0.60, 0.60),   # window      → đỏ nhạt
}
FLOOR_PLAN_LABELS = {0: "Background", 1: "Floor", 2: "Door", 3: "Window"}

ROOM_NAMES  = {0: "Bedroom", 1: "Living Room", 2: "Dining Room"}
ROOM_COLORS = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52"}


def render_floor_plan(arch_map: np.ndarray) -> np.ndarray:
    """Chuyển arch_map (H×W, giá trị 0-3) → ảnh RGB."""
    H, W = arch_map.shape
    rgb = np.ones((H, W, 3), dtype=np.float32)
    for val, color in FLOOR_PLAN_COLORS.items():
        mask = arch_map == val
        rgb[mask] = color
    return rgb


def extract_arch_map(raw_item: np.ndarray, door_id: int, window_id: int) -> np.ndarray:
    """
    Từ semantic map gốc → arch map (0=bg, 1=floor, 2=door, 3=window).
    raw_item shape: [H, W], pixel value = semantic category ID
    """
    arch = np.zeros_like(raw_item, dtype=np.uint8)
    arch[raw_item != 0] = 1          # tất cả non-zero = sàn
    arch[raw_item == door_id]   = 2  # door
    arch[raw_item == window_id] = 3  # window
    return arch


def load_samples(npy_path: str, n_samples: int = 4):
    """
    Đọc file .npy và lấy mẫu floor plan theo room type.
    Returns dict: {room_type_id: [arch_map, ...]}
    """
    print(f"Loading {npy_path} ...")
    data = np.load(npy_path, allow_pickle=True)
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")

    # Load label mapping để tìm door_id, window_id
    with open(f"{ROOT}/preprocess/metadata/unified_idx_to_generic_label.json") as f:
        idx_to_label = json.load(f)
    door_id   = int(next(k for k, v in idx_to_label.items() if v == "door"))
    window_id = int(next(k for k, v in idx_to_label.items() if v == "window"))

    # data[i] = [room_type_channel, semantic_map_channel] khi room_type_condition=True
    # Mỗi item shape [2, H, W]: [0]=room_type_id (uniform), [1]=semantic_map
    samples = {0: [], 1: [], 2: []}

    for item in data:
        if item.ndim == 3 and item.shape[0] == 2:
            room_type_id = int(np.unique(item[0])[0])
            sem_map      = item[1]
        else:
            # Fallback nếu format khác
            room_type_id = -1
            sem_map      = item

        if room_type_id not in samples:
            continue
        if len(samples[room_type_id]) >= n_samples:
            continue

        arch = extract_arch_map(sem_map, door_id, window_id)
        samples[room_type_id].append(arch)

        if all(len(v) >= n_samples for v in samples.values()):
            break

    return samples


# ── Figure 1: Grid floor plans theo room type ─────────────────────────────────

def plot_input_grid(samples: dict, out_path: str = "output_inputs_grid.png"):
    """
    Tạo ảnh grid:
      Cột = room type (Bedroom / Living Room / Dining Room)
      Hàng = các mẫu floor plan khác nhau
    """
    room_types = [0, 1, 2]
    n_rows = max(len(v) for v in samples.values())
    n_cols = len(room_types)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.2),
        squeeze=False
    )
    fig.patch.set_facecolor("#F8F8F8")

    for col, rt in enumerate(room_types):
        maps = samples.get(rt, [])
        color = ROOM_COLORS[rt]

        # Tiêu đề cột
        axes[0][col].set_title(
            f"Input: {ROOM_NAMES[rt]}\n(room_type = {rt})",
            fontsize=12, fontweight="bold", color=color, pad=10
        )

        for row in range(n_rows):
            ax = axes[row][col]
            if row < len(maps):
                rgb = render_floor_plan(maps[row])
                ax.imshow(rgb, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                if col == 0:
                    ax.set_ylabel(f"Sample {row+1}", fontsize=9, color="gray")
            else:
                ax.axis("off")

    # Legend chung
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="gray", label=FLOOR_PLAN_LABELS[v])
        for v, c in FLOOR_PLAN_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=4, fontsize=10, title="Floor Plan Legend",
        title_fontsize=11, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        "SemLayoutDiff — System Inputs\n"
        "Each floor plan = shape of room + door/window positions (120×120 px, 1px = 10cm)",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Saved: {out_path}")
    plt.show()
    plt.close()


# ── Figure 2: Sơ đồ Input → Output ───────────────────────────────────────────

def plot_pipeline_diagram(samples: dict, out_path: str = "output_pipeline.png"):
    """
    Tạo sơ đồ trực quan:
      [Room Type label]  +  [Floor Plan]  →  [SLDN]  →  [Semantic Map]
    Dùng 1 mẫu livingroom làm ví dụ.
    """
    # Lấy 1 mẫu living room
    sample_arch = samples.get(1, samples.get(0, [None]))[0]
    if sample_arch is None:
        print("Không có sample để vẽ pipeline.")
        return

    fig = plt.figure(figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 7, figure=fig,
                           width_ratios=[1.8, 0.4, 1.8, 0.4, 1.5, 0.4, 1.8])

    # ── Box 1: Room Type ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#EEF4FF")
    ax1.text(0.5, 0.65, "Room Type", ha="center", va="center",
             fontsize=11, fontweight="bold", transform=ax1.transAxes)
    ax1.text(0.5, 0.35,
             "0 = Bedroom\n1 = Living Room\n2 = Dining Room",
             ha="center", va="center", fontsize=9.5,
             color="#333333", transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#4C72B0", linewidth=1.5))
    ax1.set_title("Input ①", fontsize=10, color="#4C72B0", pad=6)
    ax1.axis("off")
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#4C72B0")
        spine.set_linewidth(2)

    # ── Arrow ─────────────────────────────────────────────────────────────────
    for col_arrow in [1, 3, 5]:
        ax_arr = fig.add_subplot(gs[col_arrow])
        ax_arr.annotate("", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="gray",
                                        lw=2, mutation_scale=18))
        ax_arr.axis("off")

    # ── Box 2: Floor Plan ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    rgb = render_floor_plan(sample_arch)
    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title("Input ②\nFloor Plan (120×120)", fontsize=10,
                  color="#55A868", pad=6)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor("#55A868")
        spine.set_linewidth(2)

    # Chú thích màu
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="gray", label=FLOOR_PLAN_LABELS[v], linewidth=0.5)
        for v, c in FLOOR_PLAN_COLORS.items() if v > 0
    ]
    ax2.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.38), ncol=3,
               fontsize=7.5, framealpha=0.9, edgecolor="gray")

    # ── Box 3: Model SLDN ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[4])
    ax3.set_facecolor("#FFF8EE")
    ax3.text(0.5, 0.6, "SLDN", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#E07B00",
             transform=ax3.transAxes)
    ax3.text(0.5, 0.3,
             "Semantic Layout\nDiffusion Network\n(Discrete Diffusion)",
             ha="center", va="center", fontsize=8.5,
             color="#555555", transform=ax3.transAxes)
    ax3.set_title("Model", fontsize=10, color="#E07B00", pad=6)
    ax3.axis("off")
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#E07B00")
        spine.set_linewidth(2)

    # ── Box 4: Output Semantic Map ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[6])
    ax4.set_facecolor("#F0FFF0")
    ax4.text(0.5, 0.62, "Semantic Layout", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#2E7D32",
             transform=ax4.transAxes)
    ax4.text(0.5, 0.35,
             "Ảnh 120×120\nmỗi pixel = loại\nđồ nội thất",
             ha="center", va="center", fontsize=9,
             color="#333333", transform=ax4.transAxes)
    ax4.set_title("Output (Stage 1)", fontsize=10, color="#2E7D32", pad=6)
    ax4.axis("off")
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2E7D32")
        spine.set_linewidth(2)

    fig.suptitle(
        "SemLayoutDiff — Stage 1: Input → Semantic Layout Generation",
        fontsize=13, fontweight="bold", y=1.04
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out_path}")
    plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path",  default="datasets/unified_w_arch_120x120.npy")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Số mẫu mỗi room type")
    parser.add_argument("--out_dir",   default=".")
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"❌ Không tìm thấy: {args.npy_path}")
        print("   Chạy trên máy có dataset, hoặc dùng Kaggle.")
        return

    samples = load_samples(args.npy_path, n_samples=args.n_samples)

    for rt, maps in samples.items():
        print(f"  {ROOM_NAMES[rt]}: {len(maps)} mẫu")

    plot_input_grid(
        samples,
        out_path=os.path.join(args.out_dir, "output_inputs_grid.png")
    )
    plot_pipeline_diagram(
        samples,
        out_path=os.path.join(args.out_dir, "output_pipeline.png")
    )


if __name__ == "__main__":
    main()
