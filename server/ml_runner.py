"""
ML inference runners for the two-stage layout generation pipeline:
  - SLDNRunner: floor plan image → semantic layout map (diffusion model)
  - APMRunner:  semantic layout map → furniture attribute predictions

Both classes are designed to be instantiated once at server startup and
reused across requests.  Model loading is lazy — it happens on first use,
or you can force it by calling `load()`.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

def _backend_new_root() -> Path:
    """Absolute path to the backend_new directory."""
    return Path(__file__).parent.parent.resolve()


def _add_src_to_path() -> None:
    root = _backend_new_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ── SLDN Runner ───────────────────────────────────────────────────────────────

class SLDNRunner:
    """
    Generates semantic layout maps from binary floor plan tensors.

    Parameters
    ----------
    model_dir : str
        Directory containing ``args.pickle`` and ``check/checkpoint.pt``.
    metadata_dir : str
        Directory containing ``color_palette.json`` and
        ``unified_idx_to_generic_label.json``.
    room_type_condition : bool
        Whether the model uses room-type conditioning (set from training config).
    mixed_condition : bool
        Whether the model uses mixed conditioning (floor / arch / uncon).
    condition_type : str
        Active mixed-condition type: "floor", "arch", or "uncon".
    """

    def __init__(
        self,
        model_dir: str,
        metadata_dir: str,
        room_type_condition: bool = True,
        mixed_condition: bool = True,
        condition_type: str = "floor",
    ) -> None:
        self._model_dir = model_dir
        self._metadata_dir = metadata_dir
        self._room_type_condition = room_type_condition
        self._mixed_condition = mixed_condition
        self._condition_type = condition_type
        self._model: Any = None
        self._colormap: dict[int, tuple[int, int, int]] | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load the SLDN model from checkpoint (idempotent)."""
        if self._model is not None:
            return
        _add_src_to_path()
        from semlayoutdiff.sldn.model import get_model
        from semlayoutdiff.sldn.sampling_utils import ImageProcessor

        args_path = os.path.join(self._model_dir, "args.pickle")
        checkpoint_path = os.path.join(self._model_dir, "check", "checkpoint.pt")

        logger.info("Loading SLDN args from %s", args_path)
        with open(args_path, "rb") as f:
            args = pickle.load(f)

        # Derive data_shape from args — no dataset access required
        data_shape = (1, args.data_size, args.data_size)

        logger.info("Building SLDN model (data_size=%d)", args.data_size)
        model = get_model(args, data_shape)

        logger.info("Loading SLDN checkpoint from %s", checkpoint_path)
        ckpt = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=self.device,
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        model = model.to(self.device)

        # Build colormap for the image processor
        colormap = self._build_colormap()
        self._model = model
        self._colormap = colormap
        self._image_processor = ImageProcessor(colormap)
        logger.info(
            "SLDN model ready (epoch %s)", ckpt.get("current_epoch", "?")
        )

    def generate(
        self,
        floor_plan_tensor: torch.Tensor,
        room_type_id: int,
        num_samples: int = 3,
    ) -> list[np.ndarray]:
        """
        Generate ``num_samples`` semantic layout maps.

        Parameters
        ----------
        floor_plan_tensor : torch.Tensor  shape [1, 1, 120, 120]
        room_type_id : int  0=bedroom, 1=livingroom, 2=diningroom
        num_samples : int

        Returns
        -------
        list of np.ndarray, each shape [120, 120], dtype int (category IDs)
        """
        if self._model is None:
            raise RuntimeError("SLDNRunner is not loaded. Call load() first.")

        batch = floor_plan_tensor.repeat(num_samples, 1, 1, 1).to(self.device)

        room_type_batch: torch.Tensor | None = None
        if self._room_type_condition:
            rt = torch.tensor(room_type_id, device=self.device)
            room_type_batch = rt.unsqueeze(0).repeat(num_samples)

        mixed_cond_batch: torch.Tensor | None = None
        if self._mixed_condition:
            cond_map = {"uncon": 0, "floor": 1, "arch": 2}
            cid = torch.tensor(
                cond_map.get(self._condition_type, 1), device=self.device
            )
            mixed_cond_batch = cid.unsqueeze(0).repeat(num_samples)

        with torch.no_grad():
            samples_chain = self._model.sample_chain(
                num_samples,
                floor_plan=batch,
                room_type=room_type_batch,
                text_condition=None,
                mixed_condition_id=mixed_cond_batch,
            )

        # samples_chain: [T, N, 1, H, W]  → permute to [N, T, 1, H, W]
        samples_chain = samples_chain.permute(1, 0, 2, 3, 4)

        semantic_maps: list[np.ndarray] = []
        for sample_chain_i in samples_chain:
            _, raw_layout = self._image_processor.process_samples(sample_chain_i)
            semantic_maps.append(raw_layout.squeeze().cpu().numpy())

        return semantic_maps

    def _build_colormap(self) -> dict[int, tuple[int, int, int]]:
        palette_path = os.path.join(self._metadata_dir, "color_palette.json")
        label_path = os.path.join(
            self._metadata_dir, "unified_idx_to_generic_label.json"
        )
        with open(palette_path) as f:
            palette: dict[str, list[int]] = json.load(f)
        with open(label_path) as f:
            idx_to_label: dict[str, str] = json.load(f)

        colormap: dict[int, tuple[int, int, int]] = {}
        for idx_str, label in idx_to_label.items():
            if label in palette:
                colormap[int(idx_str)] = tuple(palette[label])  # type: ignore[assignment]
        return colormap


# ── APM Runner ────────────────────────────────────────────────────────────────

class APMRunner:
    """
    Predicts furniture attributes (size, orientation, position) from
    a semantic layout map.

    Parameters
    ----------
    checkpoint_path : str
        Path to the APM ``*.ckpt`` checkpoint file.
    metadata_dir : str
        Directory containing ``unified_idx_to_generic_label.json``
        and ``unified_pix_ratio_threshold.json``.
    """

    IMAGE_SIZE = 1200

    def __init__(self, checkpoint_path: str, metadata_dir: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._metadata_dir = metadata_dir
        self._model: Any = None
        self._idx_to_label: dict[str, str] | None = None
        self._pix_ratio_thresh: dict[str, float] | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load the APM model from checkpoint (idempotent)."""
        if self._model is not None:
            return
        _add_src_to_path()
        from semlayoutdiff.apm.attr_module import FurnitureAttributesModel

        cfg = OmegaConf.create(
            {
                "model": {
                    "floor_id": 1,
                    "num_orientation_class": 4,
                    "num_categories": 36,
                    "embedding_dim": 128,
                    "semantic_encoder": {
                        "conv1": 32,
                        "conv2": 64,
                        "conv3": 128,
                        "attn_dim": 64,
                    },
                    "prediction_network": {"fc1": 128, "fc2": 64},
                    "attn_dim": 64,
                },
                "image_width": self.IMAGE_SIZE,
                "image_height": self.IMAGE_SIZE,
                "for_scenestate": False,
                "loss_weights": {"size": 0.2, "offset": 0.01, "orient": 1},
                "trainer": {
                    "lr": 1e-4,
                    "max_epochs": 100,
                    "l1_lambda": 0,
                    "l2_lambda": 1e-3,
                },
            }
        )

        logger.info("Loading APM checkpoint from %s", self._checkpoint_path)
        model = FurnitureAttributesModel.load_from_checkpoint(
            self._checkpoint_path, config=cfg, strict=False
        )
        model.eval()
        model = model.to(self.device)
        self._model = model

        label_path = os.path.join(
            self._metadata_dir, "unified_idx_to_generic_label.json"
        )
        thresh_path = os.path.join(
            self._metadata_dir, "unified_pix_ratio_threshold.json"
        )
        with open(label_path) as f:
            self._idx_to_label = json.load(f)
        with open(thresh_path) as f:
            self._pix_ratio_thresh = json.load(f)

        logger.info("APM model ready")

    def predict(self, semantic_map_120: np.ndarray) -> list[dict]:
        """
        Predict furniture attributes from a 120×120 semantic map.

        Upscales the map to 1200×1200 before inference (nearest-neighbour).

        Parameters
        ----------
        semantic_map_120 : np.ndarray  shape [120, 120], dtype int

        Returns
        -------
        list of dicts, each with keys:
            instance_id, category, category_id,
            position_m: {x, y, z},
            size_m: {w, h, d},
            rotation_deg: float
        """
        if self._model is None:
            raise RuntimeError("APMRunner is not loaded. Call load() first.")

        _add_src_to_path()
        from semlayoutdiff.apm.mask_utils import get_instance_masks
        from semlayoutdiff.apm.geometry_utils import convert_2d_to_3d

        # Upscale 120→1200 using nearest-neighbour (preserves category IDs)
        sem_img = Image.fromarray(semantic_map_120.astype(np.uint8))
        sem_img = sem_img.resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.NEAREST
        )
        sem_tensor = transforms.ToTensor()(sem_img) * 255  # [1, H, W]

        # Ensure correct size
        if (
            sem_tensor.shape[-1] != self.IMAGE_SIZE
            or sem_tensor.shape[-2] != self.IMAGE_SIZE
        ):
            sem_tensor = torch.nn.functional.interpolate(
                sem_tensor.unsqueeze(0),
                size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                mode="nearest",
            ).squeeze(0)

        sem_tensor_dev = sem_tensor.to(self.device)
        assert self._idx_to_label is not None
        assert self._pix_ratio_thresh is not None

        instances = get_instance_masks(
            sem_tensor, self._pix_ratio_thresh, self._idx_to_label
        )
        if not instances:
            return []

        for inst in instances.values():
            inst["mask"] = inst["mask"].to(self.device)
            inst["category"] = inst["category"].to(self.device)

        furniture: list[dict] = []
        for inst_id, data in instances.items():
            with torch.no_grad():
                size_pred, offset_pred, orientation_pred = self._model(
                    sem_tensor_dev.unsqueeze(0),
                    data["mask"],
                    data["category"].unsqueeze(0),
                )

            size = size_pred.squeeze().tolist()
            orient_cls = torch.argmax(orientation_pred, dim=-1).item()
            rotation_deg = float(orient_cls * 90)

            mask_np = data["mask"].squeeze().cpu().numpy()
            loc = convert_2d_to_3d(
                mask_np,
                image_width=self.IMAGE_SIZE,
                image_height=self.IMAGE_SIZE,
            )
            x_3d = float(loc[0]["x"])
            z_3d = float(loc[0]["y"])  # APM's "y" is world Z
            y_3d = float(offset_pred.item())

            cat_id = int(data["category"].cpu().argmax().item())
            cat_name = self._idx_to_label.get(str(cat_id), f"unknown_{cat_id}")

            furniture.append(
                {
                    "instance_id": inst_id,
                    "category": cat_name,
                    "category_id": cat_id,
                    "position_apm": {"x": x_3d, "y": y_3d, "z": z_3d},
                    "size_m": {
                        "w": round(float(size[0]), 4),
                        "h": round(float(size[1]), 4),
                        "d": round(float(size[2]), 4),
                    },
                    "rotation_deg": rotation_deg,
                }
            )

        return furniture
