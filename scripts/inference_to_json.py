"""
Simplified inference script: Semantic map → JSON layout
Không cần libsg, không cần threed_future_model_unified.json.

Output JSON mỗi sample:
{
  "image_id": "sample_000",
  "room_type": "bedroom",
  "room_size_meters": 12.0,
  "furniture": [
    {
      "instance_id": 1,
      "category": "double_bed",
      "position_m": {"x": -1.2, "y": 0.25, "z": 0.8},
      "size_m":     {"w": 1.8,  "h": 0.5,  "d": 2.0},
      "rotation_deg": 90
    },
    ...
  ]
}

Usage:
    python scripts/inference_to_json.py \
        --semantic_map_dir datasets/results/my_layouts \
        --checkpoint      checkpoints/apm_checkpoint.ckpt \
        --room_type       bedroom \
        --output_dir      datasets/results/json_output
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Thêm root vào path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from semlayoutdiff.apm.attr_module import FurnitureAttributesModel
from semlayoutdiff.apm.mask_utils import get_instance_masks
from semlayoutdiff.apm.geometry_utils import convert_2d_to_3d


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_WIDTH  = 1200   # pixel
IMAGE_HEIGHT = 1200   # pixel
SCALE        = 0.01   # 1 px = 0.01 m  →  1200 px = 12 m
ROOM_SIZE_M  = IMAGE_WIDTH * SCALE   # 12.0 m

METADATA_DIR = os.path.join(ROOT, "preprocess/metadata")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def orientation_class_to_degrees(cls: int) -> float:
    """APM predicts 4 classes: 0→0°, 1→90°, 2→180°, 3→270°."""
    return float(cls * 90)


def predict_one_sample(model, semantic_map_tensor, instances_data,
                       idx_to_label, image_width, image_height):
    """
    Chạy APM cho 1 semantic map.
    Trả về list dict mỗi phần tử là 1 món đồ nội thất.
    """
    furniture = []

    for inst_id, data in instances_data.items():
        with torch.no_grad():
            size_pred, offset_pred, orientation_pred = model(
                semantic_map_tensor.unsqueeze(0),
                data['mask'],
                data['category'].unsqueeze(0)
            )

        # ── Kích thước ────────────────────────────────────────────────────────
        size = size_pred.squeeze().tolist()   # [w, h, d]

        # ── Hướng ─────────────────────────────────────────────────────────────
        orient_class = torch.argmax(orientation_pred, dim=-1).item()
        rotation_deg = orientation_class_to_degrees(orient_class)

        # ── Vị trí ────────────────────────────────────────────────────────────
        mask_np = data['mask'].squeeze().numpy()
        loc = convert_2d_to_3d(mask_np, image_width=image_width,
                                image_height=image_height)
        x_3d = loc[0]["x"]
        z_3d = loc[0]["y"]   # hàm trả về y nhưng thực tế là trục Z
        y_3d = offset_pred.item()   # chiều cao (Y)

        # ── Category ──────────────────────────────────────────────────────────
        cat_id   = data['category'].argmax().item()
        cat_name = idx_to_label.get(str(cat_id), f"unknown_{cat_id}")

        furniture.append({
            "instance_id":  inst_id,
            "category":     cat_name,
            "category_id":  cat_id,
            "position_m": {
                "x": round(x_3d, 4),
                "y": round(y_3d, 4),
                "z": round(z_3d, 4),
            },
            "size_m": {
                "w": round(size[0], 4),
                "h": round(size[1], 4),
                "d": round(size[2], 4),
            },
            "rotation_deg": rotation_deg,
        })

    return furniture


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_map_dir", required=True,
                        help="Thư mục chứa *label*.png từ sample_layout.py")
    parser.add_argument("--checkpoint", default="checkpoints/apm_checkpoint.ckpt",
                        help="Đường dẫn tới APM checkpoint")
    parser.add_argument("--room_type", default="bedroom",
                        choices=["bedroom", "livingroom", "diningroom"])
    parser.add_argument("--output_dir", default="datasets/results/json_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────────────────────
    idx_to_label      = load_json(f"{METADATA_DIR}/unified_idx_to_generic_label.json")
    pix_ratio_thresh  = load_json(f"{METADATA_DIR}/unified_pix_ratio_threshold.json")
    # Đảo key↔value để dùng trong get_instance_masks
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    new_label_to_generic = idx_to_label   # {str_id → label_name}

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading APM checkpoint: {args.checkpoint}")

    # Tạo minimal config cho model
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model": {
            "floor_id": 1,
            "num_orientation_class": 4,
            "num_categories": 36,
            "embedding_dim": 128,
            "semantic_encoder": {"conv1": 32, "conv2": 64, "conv3": 128, "attn_dim": 64},
            "prediction_network": {"fc1": 128, "fc2": 64},
            "attn_dim": 64,
        },
        "image_width":  IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "for_scenestate": False,
        "loss_weights":  {"size": 0.2, "offset": 0.01, "orient": 1},
        "trainer":       {"lr": 1e-4, "max_epochs": 100, "l1_lambda": 0, "l2_lambda": 1e-3},
    })

    model = FurnitureAttributesModel.load_from_checkpoint(
        args.checkpoint, config=cfg, strict=False
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")

    # ── Process semantic maps ─────────────────────────────────────────────────
    label_files = sorted(glob.glob(f"{args.semantic_map_dir}/**/*label*.png",
                                   recursive=True))
    if not label_files:
        label_files = sorted(glob.glob(f"{args.semantic_map_dir}/*label*.png"))

    print(f"Found {len(label_files)} label file(s)")

    all_results = []

    for path in tqdm(label_files, desc="Inference"):
        image_id = os.path.basename(path).replace("_label.png", "") \
                                         .replace("_composite_label.png", "")

        # Load semantic map
        sem_img = Image.open(path).convert("L")
        sem_tensor = transforms.ToTensor()(sem_img) * 255  # [1, H, W], value=category_id

        # Resize nếu khác 1200×1200
        if sem_tensor.shape[-1] != IMAGE_WIDTH or sem_tensor.shape[-2] != IMAGE_HEIGHT:
            sem_tensor = torch.nn.functional.interpolate(
                sem_tensor.unsqueeze(0),
                size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                mode="nearest"
            ).squeeze(0)

        sem_tensor_dev = sem_tensor.to(device)

        # Tách instances
        instances = get_instance_masks(sem_tensor, pix_ratio_thresh, new_label_to_generic)
        if not instances:
            print(f"  No furniture found in {image_id}, skipping.")
            continue

        # Đưa mask lên device
        for inst in instances.values():
            inst['mask']     = inst['mask'].to(device)
            inst['category'] = inst['category'].to(device)

        # Predict
        furniture = predict_one_sample(
            model, sem_tensor_dev, instances,
            idx_to_label, IMAGE_WIDTH, IMAGE_HEIGHT
        )

        result = {
            "image_id":         image_id,
            "room_type":        args.room_type,
            "room_size_meters": ROOM_SIZE_M,
            "coordinate_note":  "origin=center of room, X=right, Y=up, Z=forward, unit=meters",
            "furniture":        furniture,
        }
        all_results.append(result)

        # Lưu từng file riêng
        out_path = os.path.join(args.output_dir, f"{image_id}_layout.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Lưu tất cả vào 1 file tổng hợp
    summary_path = os.path.join(args.output_dir, "all_layouts.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! {len(all_results)} layout(s) saved to: {args.output_dir}")
    print(f"   Summary: {summary_path}")


if __name__ == "__main__":
    main()
