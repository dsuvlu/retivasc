"""Optional PyTorch segmentation comparators for local demos.

These models are intentionally small CPU-capable comparators. They are not vendored
copies of the official OCTA-Net or nnU-Net projects. The official nnU-Net path remains
the adapter in :mod:`retivasc.external.nnunet`; this module gives the ROSE notebook a
way to train local U-Net-family masks when PyTorch is available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from skimage import filters, transform
from skimage import io as skio

from retivasc.evaluation import evaluate_score_postprocessing, tune_score_postprocessing
from retivasc.preprocess import ensure_grayscale, normalize_image

if TYPE_CHECKING:
    from collections.abc import Sequence

try:  # pragma: no cover - exercised when the optional dependency is installed.
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - default lightweight environment.
    torch = None
    nn = None
    F = None


PREDICTION_COLUMNS = {
    "unet_lite": "unet_lite_prediction_path",
    "octa_net_lite": "octa_net_lite_prediction_path",
    "nnunet_lite": "nnunet_lite_prediction_path",
}
DEFAULT_DEEP_METHODS = tuple(PREDICTION_COLUMNS)
DEFAULT_POSTPROCESS_PARAM_GRID = tuple(
    {
        "threshold": threshold,
        "min_size": min_size,
        "closing_radius": closing_radius,
        "dilation_radius": 0,
    }
    for threshold in (0.45, 0.50, 0.55, 0.65, 0.75)
    for min_size in (1, 8, 24)
    for closing_radius in (0, 1)
)


@dataclass(frozen=True)
class DeepSegmenterConfig:
    """Training settings for the small in-repo deep segmentation comparators."""

    image_size: int = 128
    epochs: int = 6
    batch_size: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    threshold: float = 0.5
    base_channels: int = 8
    device: str = "auto"
    seed: int = 0


def torch_available() -> bool:
    """Return whether the optional PyTorch dependency can be imported."""
    return torch is not None


def require_torch() -> None:
    """Raise a clear error if PyTorch is unavailable."""
    if torch is None:
        msg = (
            "PyTorch is required for in-repo deep segmentation comparators. "
            "Install a PyTorch-enabled environment, then rerun this command."
        )
        raise RuntimeError(msg)


if nn is not None:  # pragma: no cover - covered only in PyTorch-enabled envs.

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, norm: str = "group") -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                _norm_layer(out_channels, norm),
                _activation(norm),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                _norm_layer(out_channels, norm),
                _activation(norm),
            )

        def forward(self, x):
            return self.net(x)


    class _ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = _ConvBlock(in_channels, out_channels, norm="group")
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        def forward(self, x):
            return self.block(x) + self.skip(x)


    class _TinyUNet(nn.Module):
        def __init__(
            self,
            in_channels: int,
            *,
            base_channels: int,
            norm: str = "group",
            residual: bool = False,
            dilation_context: bool = False,
        ) -> None:
            super().__init__()
            block = _ResidualBlock if residual else lambda i, o: _ConvBlock(i, o, norm=norm)
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            self.enc1 = block(in_channels, c1)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = block(c1, c2)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge = block(c2, c3)
            self.context = (
                nn.Sequential(
                    nn.Conv2d(c3, c3, kernel_size=3, padding=2, dilation=2, bias=False),
                    _norm_layer(c3, norm),
                    _activation(norm),
                )
                if dilation_context
                else nn.Identity()
            )
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.dec2 = block(c2 + c2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.dec1 = block(c1 + c1, c1)
            self.out = nn.Conv2d(c1, 1, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            bridge = self.context(self.bridge(self.pool2(e2)))
            d2 = self.up2(bridge)
            d2 = self.dec2(torch.cat([_crop_or_pad(d2, e2), e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([_crop_or_pad(d1, e1), e1], dim=1))
            return self.out(d1)


def build_model(method: str, *, base_channels: int = 8):
    """Build one small optional PyTorch segmentation model."""
    require_torch()
    if method == "unet_lite":
        return _TinyUNet(1, base_channels=base_channels, norm="group")
    if method == "octa_net_lite":
        return _TinyUNet(
            2,
            base_channels=base_channels,
            norm="group",
            residual=True,
            dilation_context=True,
        )
    if method == "nnunet_lite":
        return _TinyUNet(1, base_channels=base_channels * 2, norm="instance")
    msg = f"Unknown deep segmentation method: {method}"
    raise ValueError(msg)


def train_predict_deep_segmenters(
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    output_root: str | Path,
    *,
    methods: Sequence[str] = DEFAULT_DEEP_METHODS,
    config: DeepSegmenterConfig | None = None,
    tune_manifest: pd.DataFrame | None = None,
    postprocess_param_grid: Sequence[dict[str, object]] | None = DEFAULT_POSTPROCESS_PARAM_GRID,
    image_col: str = "image_path",
    mask_col: str = "mask_path",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train small local comparators and write tuned prediction masks for eval rows."""
    require_torch()
    cfg = config or DeepSegmenterConfig()
    _validate_config(cfg)
    _require_columns(train_manifest, [image_col, mask_col])
    _require_columns(eval_manifest, [image_col, mask_col])
    if tune_manifest is not None:
        _require_columns(tune_manifest, [image_col, mask_col])
    method_names = tuple(methods)
    unknown = sorted(set(method_names) - set(PREDICTION_COLUMNS))
    if unknown:
        msg = f"Unknown deep segmentation method(s): {', '.join(unknown)}"
        raise ValueError(msg)
    if train_manifest.empty:
        raise ValueError("train_manifest must contain at least one row.")
    if eval_manifest.empty:
        raise ValueError("eval_manifest must contain at least one row.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    _set_seed(cfg.seed)

    predictions = eval_manifest.copy()
    tune_rows = train_manifest if tune_manifest is None else tune_manifest
    history_rows: list[dict[str, object]] = []
    for method in method_names:
        method_root = output_root / method

        x_train, y_train = _manifest_tensors(
            train_manifest,
            method,
            image_col=image_col,
            mask_col=mask_col,
            image_size=cfg.image_size,
        )
        model = build_model(method, base_channels=cfg.base_channels).to(device)
        history_rows.extend(
            _train_model(
                model,
                x_train,
                y_train,
                method=method,
                cfg=cfg,
                device=device,
            )
        )
        tune_predictions = _predict_manifest_for_method(
            model,
            tune_rows,
            method,
            method_root / "tuning",
            cfg=cfg,
            image_col=image_col,
            threshold=cfg.threshold,
            device=device,
        )
        eval_predictions = _predict_manifest_for_method(
            model,
            eval_manifest,
            method,
            method_root / "eval",
            cfg=cfg,
            image_col=image_col,
            threshold=cfg.threshold,
            device=device,
        )

        prediction_col = PREDICTION_COLUMNS[method]
        score_col = f"{method}_score_path"
        predictions[prediction_col] = eval_predictions[prediction_col].to_list()
        predictions[score_col] = eval_predictions[score_col].to_list()

        if postprocess_param_grid:
            tuning_table, tuning_summary, best_params = tune_score_postprocessing(
                tune_predictions,
                {method: score_col},
                postprocess_param_grid,
                image_col=image_col,
                mask_col=mask_col,
            )
            tuned_eval = evaluate_score_postprocessing(
                eval_predictions,
                {method: score_col},
                best_params,
                image_col=image_col,
                mask_col=mask_col,
                output_root=method_root / "tuned_masks",
            )
            if not tuned_eval.empty and tuned_eval["pred_mask_path"].notna().all():
                predictions[prediction_col] = tuned_eval["pred_mask_path"].to_list()
            predictions[f"{method}_postprocess_params_json"] = [
                json.dumps(best_params.get(method, {}), sort_keys=True)
            ] * len(predictions)
            tuning_table.to_csv(method_root / "postprocess_tuning_metrics.csv", index=False)
            tuning_summary.to_csv(method_root / "postprocess_tuning_summary.csv", index=False)
            for summary_row in tuning_summary.to_dict(orient="records"):
                history_rows.append(
                    {
                        **summary_row,
                        "method": method,
                        "stage": "postprocess_tuning",
                        "config_json": json.dumps(best_params.get(method, {}), sort_keys=True),
                    }
                )


    history = pd.DataFrame(history_rows)
    return predictions, history


def _predict_manifest_for_method(
    model,
    manifest: pd.DataFrame,
    method: str,
    output_root: Path,
    *,
    cfg: DeepSegmenterConfig,
    image_col: str,
    threshold: float,
    device,
) -> pd.DataFrame:
    mask_root = output_root / "masks"
    score_root = output_root / "scores"
    mask_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)
    predictions = manifest.copy()
    prediction_paths = []
    score_paths = []
    for row_index, row in manifest.iterrows():
        image_path = Path(str(row[image_col]))
        original_shape = ensure_grayscale(skio.imread(image_path)).shape
        prob = _predict_probability(
            model,
            row,
            method,
            image_col=image_col,
            image_size=cfg.image_size,
            original_shape=original_shape,
            device=device,
        )
        pred_mask = prob >= threshold
        stem = _safe_stem(f"{row_index}_{row.get('image_id', image_path.stem)}_{method}")
        pred_path = mask_root / f"{stem}.png"
        score_path = score_root / f"{stem}.npy"
        skio.imsave(pred_path, pred_mask.astype(np.uint8) * 255, check_contrast=False)
        np.save(score_path, prob.astype(np.float32, copy=False))
        prediction_paths.append(str(pred_path))
        score_paths.append(str(score_path))
    predictions[PREDICTION_COLUMNS[method]] = prediction_paths
    predictions[f"{method}_score_path"] = score_paths
    return predictions


def select_rose_deep_demo_rows(
    manifest: pd.DataFrame,
    *,
    train_max_rows: int = 8,
    eval_max_rows: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the same small ROSE-1/SVC demo rows used by the notebook."""
    candidates = manifest.copy()
    if "dataset" in candidates.columns:
        rose1 = candidates.loc[candidates["dataset"].astype("string") == "ROSE-1"]
        if not rose1.empty:
            candidates = rose1
    if "layer" in candidates.columns:
        svc = candidates.loc[candidates["layer"].astype("string") == "SVC"]
        if not svc.empty:
            candidates = svc
    candidates = candidates.reset_index(drop=True)
    tuning = _balanced_sample(candidates, max_rows=train_max_rows, split="train")
    excluded = (
        set(tuning["subject_id"].astype("string")) if "subject_id" in tuning.columns else set()
    )
    comparison = _balanced_sample(
        candidates,
        max_rows=eval_max_rows,
        split="test",
        exclude_subjects=excluded,
    )
    if comparison.empty:
        comparison = _balanced_sample(candidates, max_rows=eval_max_rows, exclude_subjects=excluded)
    return tuning, comparison


def _train_model(
    model,
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    method: str,
    cfg,
    device,
) -> list[dict]:
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    records = []
    model.train()
    for epoch in range(cfg.epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        losses = []
        for start in range(0, x.shape[0], cfg.batch_size):
            batch_ids = permutation[start : start + cfg.batch_size]
            logits = model(x[batch_ids])
            loss = _segmentation_loss(logits, y[batch_ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        records.append(
            {
                "method": method,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "config_json": json.dumps(asdict(cfg), sort_keys=True),
            }
        )
    return records


def _predict_probability(
    model,
    row: pd.Series,
    method: str,
    *,
    image_col: str,
    image_size: int,
    original_shape: tuple[int, int],
    device,
) -> np.ndarray:
    x = _row_input(row, method, image_col=image_col, image_size=image_size)
    tensor = torch.as_tensor(x[None, ...], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    if prob.shape != original_shape:
        prob = transform.resize(
            prob,
            original_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
    return np.clip(prob, 0.0, 1.0)


def _manifest_tensors(
    manifest: pd.DataFrame,
    method: str,
    *,
    image_col: str,
    mask_col: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    masks = []
    for _, row in manifest.iterrows():
        inputs.append(_row_input(row, method, image_col=image_col, image_size=image_size))
        mask = ensure_grayscale(skio.imread(row[mask_col])) > 0
        masks.append(_resize_mask(mask, image_size)[None, ...])
    return np.stack(inputs, axis=0), np.stack(masks, axis=0).astype(np.float32)


def _row_input(
    row: pd.Series,
    method: str,
    *,
    image_col: str,
    image_size: int,
) -> np.ndarray:
    image = normalize_image(ensure_grayscale(skio.imread(row[image_col])))
    image = _resize_image(image, image_size)
    channels = [image.astype(np.float32, copy=False)]
    if method == "octa_net_lite":
        vesselness = filters.frangi(image, black_ridges=False)
        channels.append(normalize_image(vesselness).astype(np.float32, copy=False))
    return np.stack(channels, axis=0)


def _resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.shape == (image_size, image_size):
        return image.astype(np.float32, copy=False)
    return transform.resize(
        image,
        (image_size, image_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32, copy=False)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    if mask.shape == (image_size, image_size):
        return mask.astype(np.float32, copy=False)
    return transform.resize(
        mask.astype(float),
        (image_size, image_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32, copy=False)


def _segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    prob = torch.sigmoid(logits)
    smooth = 1.0
    intersection = torch.sum(prob * targets, dim=(1, 2, 3))
    denominator = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = 1.0 - torch.mean((2.0 * intersection + smooth) / (denominator + smooth))
    return bce + dice


def _norm_layer(channels: int, norm: str):
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    groups = min(4, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _activation(norm: str):
    return nn.LeakyReLU(negative_slope=0.01, inplace=True) if norm == "instance" else nn.ReLU(True)


def _crop_or_pad(source, target):
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(cfg: DeepSegmenterConfig) -> None:
    if cfg.image_size < 16:
        raise ValueError("image_size must be at least 16.")
    if cfg.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if cfg.base_channels < 2:
        raise ValueError("base_channels must be at least 2.")
    if not 0.0 < cfg.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")


def _balanced_sample(
    candidates: pd.DataFrame,
    *,
    max_rows: int,
    split: str | None = None,
    exclude_subjects=(),
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows = candidates.copy()
    if split is not None and "official_split" in rows.columns:
        split_rows = rows.loc[rows["official_split"].astype("string").str.lower() == split]
        if not split_rows.empty:
            rows = split_rows
    if exclude_subjects and "subject_id" in rows.columns:
        rows = rows.loc[~rows["subject_id"].astype("string").isin(set(exclude_subjects))]
    if rows.empty:
        return pd.DataFrame()

    selected_indexes = []
    if "label" in rows.columns:
        labels = rows["label"].astype("string").str.lower()
        for label in ("disease", "control"):
            label_rows = rows.loc[labels == label]
            if not label_rows.empty:
                selected_indexes.extend(label_rows.head(max(1, max_rows // 2)).index)
    for row_index in rows.index:
        if row_index not in selected_indexes:
            selected_indexes.append(row_index)
        if len(selected_indexes) >= max_rows:
            break

    selected = rows.loc[selected_indexes].copy()
    if "image_id" in selected.columns:
        selected = selected.drop_duplicates("image_id")
    return selected.head(max_rows).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required manifest columns: {', '.join(missing)}"
        raise ValueError(msg)


def _safe_stem(value: str) -> str:
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(cleaned).strip("_") or "image"


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train small local deep segmentation comparators.")
    parser.add_argument("--train-manifest", help="CSV manifest for training rows.")
    parser.add_argument("--eval-manifest", help="CSV manifest for evaluation rows.")
    parser.add_argument(
        "--rose-root",
        help="Load ROSE data and select the demo rows automatically.",
    )
    parser.add_argument(
        "--out-manifest",
        default="data/interim/rose_inrepo_deep_predictions.csv",
        help="CSV path for prediction columns.",
    )
    parser.add_argument(
        "--history-out",
        default="data/interim/rose_inrepo_deep_training_history.csv",
        help="CSV path for training history.",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim/rose_inrepo_deep_comparison",
        help="Folder for predicted masks and probability maps.",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_DEEP_METHODS))
    parser.add_argument("--epochs", type=int, default=DeepSegmenterConfig.epochs)
    parser.add_argument("--image-size", type=int, default=DeepSegmenterConfig.image_size)
    parser.add_argument("--base-channels", type=int, default=DeepSegmenterConfig.base_channels)
    parser.add_argument("--batch-size", type=int, default=DeepSegmenterConfig.batch_size)
    parser.add_argument("--device", default=DeepSegmenterConfig.device)
    args = parser.parse_args(argv)

    if args.rose_root:
        from retivasc.io import load_rose_manifest

        manifest = load_rose_manifest(args.rose_root, require_split_safe=False)
        train_manifest, eval_manifest = select_rose_deep_demo_rows(manifest)
    else:
        if not args.train_manifest or not args.eval_manifest:
            parser.error("Provide --rose-root or both --train-manifest and --eval-manifest.")
        train_manifest = pd.read_csv(args.train_manifest)
        eval_manifest = pd.read_csv(args.eval_manifest)

    cfg = DeepSegmenterConfig(
        epochs=args.epochs,
        image_size=args.image_size,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        device=args.device,
    )
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    predictions, history = train_predict_deep_segmenters(
        train_manifest,
        eval_manifest,
        args.output_root,
        methods=methods,
        config=cfg,
    )
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_manifest, index=False)
    history_out = Path(args.history_out)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
