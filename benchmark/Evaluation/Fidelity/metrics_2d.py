from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_METRICS = ("psnr_full", "ssim", "mask_iou", "lpips", "clip")


def clip_weights(cache: Path, offline: bool) -> Path:
    import open_clip
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    config = open_clip.get_pretrained_cfg("ViT-B-32-quickgelu", "openai")
    repository, filename = (
        config["hf_hub"].rstrip("/").rsplit("/", 1)
        if not config["hf_hub"].endswith("/")
        else (config["hf_hub"].rstrip("/"), "open_clip_pytorch_model.bin")
    )
    filenames = ("open_clip_model.safetensors", filename)
    for filename in filenames:
        cached = try_to_load_from_cache(repository, filename, cache_dir=str(cache))
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
    if offline:
        raise FileNotFoundError(f"CLIP ViT-B-32-quickgelu/openai weights are not cached in {cache}")
    return Path(
        hf_hub_download(
            repo_id=repository,
            filename="open_clip_model.safetensors",
            cache_dir=str(cache),
            token=False,
        )
    )


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
            alpha = rgba[..., 3:4]
            return rgba[..., :3] * alpha + (1.0 - alpha)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return rgb


def load_mask(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with Image.open(path) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            mask = np.asarray(image.convert("RGBA"))[..., 3] > 127
        else:
            mask = np.asarray(image.convert("L")) > 127
    if mask.shape != shape:
        raise ValueError(f"Mask shape mismatch for {path}: {mask.shape} != {shape}")
    return mask


class ImageScorer:
    def __init__(
        self,
        metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
        device: str = "auto",
        cache_dir: Path | None = None,
        *,
        clip_checkpoint: Path | None = None,
        offline: bool = False,
    ) -> None:
        if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(DEFAULT_METRICS):
            raise ValueError(f"Choose distinct metrics from {DEFAULT_METRICS}")
        self.metrics = tuple(metrics)
        self.models: dict[str, Any] = {}
        self.transforms: dict[str, Any] = {}
        self.device = device
        if not set(metrics) & {"clip", "lpips"}:
            return
        import torch

        self.torch = torch
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if cache_dir is None:
            from artisan.config import cache_dir as project_cache_dir

            cache_dir = project_cache_dir() / "fidelity_models"
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        hub = cache_dir / "torch" / "hub"
        torch.hub.set_dir(str(hub))
        if "clip" in metrics:
            try:
                import open_clip

                checkpoint = (
                    Path(clip_checkpoint)
                    if clip_checkpoint is not None
                    else clip_weights(cache_dir / "open_clip", offline)
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                config = open_clip.get_pretrained_cfg("ViT-B-32-quickgelu", "openai")
                model, _, transform = open_clip.create_model_and_transforms(
                    "ViT-B-32-quickgelu",
                    pretrained=str(checkpoint),
                    cache_dir=str(cache_dir / "open_clip"),
                    image_mean=config["mean"],
                    image_std=config["std"],
                    image_interpolation=config["interpolation"],
                    image_resize_mode=config["resize_mode"],
                )
                self.models["clip"] = model.eval().to(self.device)
                self.transforms["clip"] = transform
            except Exception as exc:
                raise RuntimeError(f"Cannot load CLIP ViT-B-32-quickgelu/openai: {exc}") from exc
        if "lpips" in metrics:
            try:
                checkpoint = hub / "checkpoints" / "alexnet-owt-7be5be79.pth"
                if offline and not checkpoint.is_file():
                    raise FileNotFoundError(f"LPIPS AlexNet weights not found: {checkpoint}")
                import lpips

                self.models["lpips"] = lpips.LPIPS(net="alex", verbose=False).eval().to(self.device)
            except Exception as exc:
                raise RuntimeError(f"Cannot load LPIPS AlexNet: {exc}") from exc

    def compare(self, prediction: Path, reference: Path) -> dict[str, float]:
        pred = load_image(prediction)
        gt = load_image(reference)
        if pred.shape != gt.shape:
            raise ValueError(f"Image shape mismatch: {pred.shape} != {gt.shape}")
        pred_mask_path = prediction.with_name(f"{prediction.stem}_mask.png")
        gt_mask_path = reference.with_name(f"{reference.stem}_mask.png")
        if "mask_iou" in self.metrics and not (pred_mask_path.is_file() and gt_mask_path.is_file()):
            raise ValueError("Geometric masks are required for IoU; rerender with --refresh")
        if pred_mask_path.is_file() != gt_mask_path.is_file():
            raise ValueError("Independent masks must be supplied for both prediction and reference")
        if pred_mask_path.is_file():
            pred_mask = load_mask(pred_mask_path, pred.shape[:2])
            gt_mask = load_mask(gt_mask_path, gt.shape[:2])
        result: dict[str, float] = {}
        if "psnr_full" in self.metrics:
            result["psnr_full"] = float(
                -10 * np.log10(max(float(np.square(pred - gt).mean()), 1e-12))
            )
        if "ssim" in self.metrics:
            from skimage.metrics import structural_similarity

            result["ssim"] = float(structural_similarity(pred, gt, channel_axis=2, data_range=1.0))
        if "mask_iou" in self.metrics:
            union = pred_mask | gt_mask
            result["mask_iou"] = (
                float((pred_mask & gt_mask).sum() / union.sum()) if union.any() else 1.0
            )
        if not self.models:
            return result
        torch = self.torch
        import torch.nn.functional as functional

        images = [Image.fromarray(np.round(rgb * 255).astype(np.uint8)) for rgb in (pred, gt)]
        with torch.inference_mode():
            if "clip" in self.models:
                batch = torch.stack([self.transforms["clip"](image) for image in images]).to(
                    self.device
                )
                embeddings = self.models["clip"].encode_image(batch)
                embeddings = functional.normalize(embeddings.float(), dim=-1)
                result["clip"] = float((embeddings[0] * embeddings[1]).sum().item())
            if "lpips" in self.models:
                batch = (
                    torch.from_numpy(np.stack([pred, gt]))
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(self.device)
                    * 2
                    - 1
                )
                batch = functional.interpolate(
                    batch, size=(256, 256), mode="bilinear", align_corners=False
                )
                result["lpips"] = float(self.models["lpips"](batch[:1], batch[1:]).mean().item())
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Image metrics produced a non-finite value")
        return result

    def score_pairs(self, pairs: list[tuple[Path, Path]]) -> dict[str, Any]:
        if not pairs:
            raise ValueError("At least one prediction/reference pair is required")
        scores = [self.compare(pred, gt) for pred, gt in pairs]
        return {
            "mean": {
                name: float(np.mean([score[name] for score in scores])) for name in self.metrics
            },
        }


def score_pairs(
    pairs: list[tuple[Path, Path]],
    metrics: list[str],
    device: str = "auto",
    cache_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return ImageScorer(metrics, device, cache_dir, **kwargs).score_pairs(pairs)
