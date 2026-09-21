"""Shared mammography preprocessing and explicit augmentation policies."""

from __future__ import annotations
import random
import math
from typing import Any
import numpy as np
import torch
from torch import nn
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as TF
from .config import ExperimentConfig, deep_get


class ConfigurableMGAugmentation(nn.Module):
    """Config-driven mammography augmentation.

    Supports the v2 JSON keys used in mg_lejepa_aug_v2.json. Operations that
    require OpenCV, such as CLAHE, are skipped with a one-time warning if cv2 is
    not installed. All transforms operate on one-channel float tensors in [0, 1].
    """

    _warned_no_cv2 = False

    def __init__(self, aug_cfg: dict[str, Any], image_size: int, train: bool):
        super().__init__()
        self.cfg = aug_cfg
        self.image_size = image_size
        self.training_views = bool(train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W] in [0,1]
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)

        if self.training_views:
            # v2 config can resize to a larger intermediate canvas, then crop to output size.
            pre_resize_cfg = deep_get(self.cfg, ["preprocessing", "resize_after_foreground_crop"], {})
            if pre_resize_cfg.get("enabled", True):
                x = self._resize(x, int(pre_resize_cfg.get("size", max(self.image_size, 256))))
            else:
                x = self._resize(x, self.image_size)

            x = self._random_resized_crop(x)
            x = self._horizontal_flip(x)
            x = self._vertical_flip(x)
            x = self._large_rotation(x)
            x = self._random_affine(x)
            x = self._gamma(x)
            x = self._brightness_contrast(x)
            x = self._noise(x)
            x = self._blur(x)
            x = self._sharpen(x)
            x = self._histogram_equalization(x)
            x = self._clahe(x)
            x = self._intensity_inversion(x)
            x = self._posterization(x)
            x = self._random_erasing(x)
            x = self._cutout(x)
            return x.clamp(0, 1)

        # Evaluation uses the same foreground crop and corner mask, followed by resize.
        x = self._resize(x, self.image_size)
        return x.clamp(0, 1)

    def _foreground_crop(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["preprocessing", "foreground_crop"], {})
        if not c.get("enabled", False):
            return x
        threshold = float(c.get("threshold_abs", 1e-6))
        margin_frac = float(c.get("margin_frac", 0.05))
        min_area_frac = float(c.get("min_foreground_area_frac", 0.01))
        fallback = bool(c.get("fallback_to_original", True))
        mask = x[0] > threshold
        ys, xs = torch.where(mask)
        h, w = x.shape[-2:]
        if len(xs) < int(h * w * min_area_frac):
            return x if fallback else x[:, :h, :w]
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mh, mw = int((y1 - y0) * margin_frac), int((x1 - x0) * margin_frac)
        return x[:, max(0, y0 - mh) : min(h, y1 + mh), max(0, x0 - mw) : min(w, x1 + mw)]

    def _mask_top_corner(self, x: torch.Tensor) -> torch.Tensor:
        """Mask likely watermark / metadata text in the top corner after foreground crop.

        Enabled by this augmentation config block:
        {
          "preprocessing": {
            "top_right_corner_mask": {
              "enabled": true,
              "frac_x": 0.30,
              "frac_y": 0.12,
              "value": 0.0,
              "foreground_threshold": 0.0001,
              "min_component_area_frac": 0.0002,
              "skip_if_single_component": true
            }
          }
        }

        The side is chosen dynamically: the mask is applied to the top side with
        less foreground tissue, so left/right mammograms are handled consistently.
        """
        c = deep_get(self.cfg, ["preprocessing", "top_right_corner_mask"], {})
        if not c.get("enabled", False):
            return x

        frac_x = float(c.get("frac_x", 0.30))
        frac_y = float(c.get("frac_y", 0.12))
        value = float(c.get("value", 0.0))
        foreground_threshold = float(c.get("foreground_threshold", 1e-4))
        min_component_area_frac = float(c.get("min_component_area_frac", 0.0002))
        skip_if_single_component = bool(c.get("skip_if_single_component", True))

        _, h, w = x.shape
        mh = max(1, int(round(h * frac_y)))
        mw = max(1, int(round(w * frac_x)))

        foreground = x[0] > foreground_threshold

        if skip_if_single_component:
            try:
                import cv2  # type: ignore

                mask_np = foreground.detach().cpu().numpy().astype("uint8")
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask_np,
                    connectivity=8,
                )

                min_area = max(1, int(round(h * w * min_component_area_frac)))

                relevant_components = 0
                for label_idx in range(1, num_labels):
                    area = int(stats[label_idx, cv2.CC_STAT_AREA])
                    if area >= min_area:
                        relevant_components += 1

                if relevant_components <= 1:
                    return x

            except Exception:
                # If OpenCV is not available or connected components fail, still apply
                # the deterministic top-corner side heuristic below.
                pass

        left_half = foreground[:, : w // 2]
        right_half = foreground[:, w // 2 :]

        left_foreground = left_half.float().sum().item()
        right_foreground = right_half.float().sum().item()

        x = x.clone()
        if left_foreground <= right_foreground:
            x[:, :mh, :mw] = value
        else:
            x[:, :mh, w - mw :] = value
        return x

    @staticmethod
    def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
        return TF.resize(x, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "random_resized_crop"], {})
        if not c.get("enabled", False):
            return self._resize(x, self.image_size)
        scale = tuple(c.get("scale", [0.85, 1.0]))
        ratio = tuple(c.get("ratio", [0.9, 1.1]))
        size = int(c.get("size", self.image_size))
        i, j, h, w = RandomResizedCrop.get_params(x, scale=scale, ratio=ratio)
        return TF.resized_crop(x, i, j, h, w, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "horizontal_flip"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.5)):
            return TF.hflip(x)
        return x

    def _vertical_flip(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "vertical_flip"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            return TF.vflip(x)
        return x

    def _large_rotation(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "large_rotation_90_180"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            angle = random.choice(c.get("angles", [90, 180, 270]))
            return TF.rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        return x

    def _random_affine(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "random_affine"], {})
        if not c.get("enabled", False) or random.random() > float(c.get("p", 0.5)):
            return x
        degrees = float(c.get("degrees", 3.0))
        tr = c.get("translate", [0.02, 0.02])
        sc = c.get("scale", [0.97, 1.03])
        sh = c.get("shear", [0.0, 0.0])
        angle = random.uniform(-degrees, degrees)
        h, w = x.shape[-2:]
        tx = int(random.uniform(-float(tr[0]), float(tr[0])) * w)
        ty = int(random.uniform(-float(tr[1]), float(tr[1])) * h)
        scale = random.uniform(float(sc[0]), float(sc[1]))
        shear = [random.uniform(float(sh[0]), float(sh[1])), 0.0]
        return TF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=[float(c.get("fill", 0.0))],
        )

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "random_gamma"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.5)):
            g = c.get("gamma", [0.9, 1.1])
            return x.clamp(0, 1).pow(random.uniform(float(g[0]), float(g[1])))
        return x

    def _brightness_contrast(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "brightness_contrast"], {})
        if not c.get("enabled", False) or random.random() > float(c.get("p", 0.5)):
            return x
        b, co = c.get("brightness", [0.95, 1.05]), c.get("contrast", [0.9, 1.1])
        brightness = random.uniform(float(b[0]), float(b[1]))
        contrast = random.uniform(float(co[0]), float(co[1]))
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return ((x - mean) * contrast + mean).mul(brightness).clamp(0, 1)

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "gaussian_noise"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.2)):
            sr = c.get("std", [0.0, 0.01])
            std = random.uniform(float(sr[0]), float(sr[1]))
            out = x + torch.randn_like(x) * std
            return out.clamp(0, 1) if c.get("clip", True) else out
        return x

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "gaussian_blur"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.1)):
            k = int(c.get("kernel_size", 3))
            if k % 2 == 0:
                k += 1
            return TF.gaussian_blur(x, kernel_size=[k, k], sigma=tuple(c.get("sigma", [0.1, 0.6])))
        return x

    def _sharpen(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "sharpen"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            factors = c.get("sharpness_factor", [1.0, 1.2])
            factor = random.uniform(float(factors[0]), float(factors[1]))
            return TF.adjust_sharpness(x, sharpness_factor=factor).clamp(0, 1)
        return x

    @staticmethod
    def _to_uint8(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(0, 1) * 255.0).round().to(torch.uint8)

    @staticmethod
    def _from_uint8(x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def _histogram_equalization(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "histogram_equalization"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            return self._from_uint8(TF.equalize(self._to_uint8(x))).clamp(0, 1)
        return x

    def _clahe(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "clahe"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 1.0))):
            return x
        try:
            import cv2  # type: ignore

            arr = (x.squeeze(0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            clip_limit = float(c.get("clip_limit", 2.0))
            tile_grid_size = tuple(c.get("tile_grid_size", [8, 8]))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            out = clahe.apply(arr).astype(np.float32) / 255.0
            return torch.from_numpy(out).unsqueeze(0).to(dtype=x.dtype)
        except Exception:
            if not ConfigurableMGAugmentation._warned_no_cv2:
                print("WARNING: CLAHE requested but OpenCV/cv2 is unavailable or failed. Skipping CLAHE.", flush=True)
                ConfigurableMGAugmentation._warned_no_cv2 = True
            return x

    def _intensity_inversion(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "intensity_inversion"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            return 1.0 - x
        return x

    def _posterization(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "posterization"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            bits = int(c.get("bits", 6))
            bits = max(1, min(8, bits))
            return self._from_uint8(TF.posterize(self._to_uint8(x), bits=bits)).clamp(0, 1)
        return x

    def _random_erasing(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["occlusion", "random_erasing"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 0.0))):
            return x
        scale = c.get("scale", [0.01, 0.03])
        ratio = c.get("ratio", [0.3, 3.3])
        value = float(c.get("value", 0.0))
        _, h, w = x.shape
        area = h * w
        for _ in range(10):
            target = random.uniform(float(scale[0]), float(scale[1])) * area
            aspect = math.exp(random.uniform(math.log(float(ratio[0])), math.log(float(ratio[1]))))
            erase_h = int(round(math.sqrt(target * aspect)))
            erase_w = int(round(math.sqrt(target / aspect)))
            if erase_h < h and erase_w < w:
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                x = x.clone()
                x[:, i : i + erase_h, j : j + erase_w] = value
                return x
        return x

    def _cutout(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["occlusion", "cutout"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 0.0))):
            return x
        size_frac = float(c.get("size_frac", 0.05))
        value = float(c.get("value", 0.0))
        _, h, w = x.shape
        ch = max(1, int(h * size_frac))
        cw = max(1, int(w * size_frac))
        i = random.randint(0, max(0, h - ch))
        j = random.randint(0, max(0, w - cw))
        x = x.clone()
        x[:, i : i + ch, j : j + cw] = value
        return x


class MildSupervisedMGAugmentation(ConfigurableMGAugmentation):
    def __init__(
        self,
        aug_cfg: dict[str, Any],
        image_size: int,
        train: bool,
        hflip_p: float = 0.5,
        max_rotation_deg: float = 5.0,
    ):
        super().__init__(aug_cfg, image_size, train)
        self.hflip_p = float(hflip_p)
        self.max_rotation_deg = float(max_rotation_deg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)
        x = TF.resize(x, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.training_views:
            if self.hflip_p > 0 and random.random() < self.hflip_p:
                x = TF.hflip(x)
            if self.max_rotation_deg > 0:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                x = TF.rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        return x.float().clamp(0, 1)


def make_supervised_transform(aug_cfg: dict[str, Any], cfg: ExperimentConfig, train: bool) -> nn.Module:
    mode = str(cfg.supervised_aug_mode).lower().strip()
    if mode == "config":
        return ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=train)
    if mode == "mild":
        return MildSupervisedMGAugmentation(
            aug_cfg, cfg.image_size, train=train, hflip_p=cfg.hflip_p, max_rotation_deg=cfg.max_rotation_deg
        )
    if mode == "none":
        return MildSupervisedMGAugmentation(aug_cfg, cfg.image_size, train=train, hflip_p=0.0, max_rotation_deg=0.0)
    raise ValueError(f"Unknown supervised_aug_mode: {cfg.supervised_aug_mode}")


def evaluation_transform(aug_cfg: dict[str, Any], image_size: int) -> nn.Module:
    return ConfigurableMGAugmentation(aug_cfg, image_size, train=False)
