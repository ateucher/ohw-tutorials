"""Visualisation helpers for the underwater vision tutorial."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def _require_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib import patches

    return plt, patches


def _open_image(image_path: str | Path):
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def _class_name(class_names, class_id: int) -> str:
    if class_names is None:
        return str(class_id)
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_names.get(str(class_id), class_id)))
    if class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


def show_image_grid(
    image_paths: Sequence[str | Path],
    *,
    titles: Sequence[str] | None = None,
    max_images: int = 12,
    columns: int = 4,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Display a compact grid of images."""

    plt, _ = _require_matplotlib()
    image_paths = list(image_paths)[:max_images]
    if not image_paths:
        print("No images to display.")
        return

    rows = (len(image_paths) + columns - 1) // columns
    figsize = figsize or (4 * columns, 3 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    if rows == 1 and columns == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flat", axes))

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index >= len(image_paths):
            continue
        ax.imshow(_open_image(image_paths[index]))
        if titles and index < len(titles):
            ax.set_title(titles[index], fontsize=9)
    plt.tight_layout()
    plt.show()


def _read_label_lines(label_path: str | Path) -> list[list[float]]:
    path = Path(label_path)
    if not path.exists():
        return []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(part) for part in line.split()])
    return rows


def draw_yolo_boxes(
    image_path: str | Path,
    label_path: str | Path,
    *,
    class_names=None,
    ax=None,
    color: str = "lime",
    linewidth: float = 2.0,
):
    """Draw YOLO detection labels on one image."""

    plt, patches = _require_matplotlib()
    image = _open_image(image_path)
    width, height = image.size
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image)
    ax.axis("off")

    for row in _read_label_lines(label_path):
        if len(row) != 5:
            continue
        class_id, cx, cy, box_w, box_h = row
        x0 = (cx - box_w / 2) * width
        y0 = (cy - box_h / 2) * height
        rect = patches.Rectangle(
            (x0, y0),
            box_w * width,
            box_h * height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(x0, y0, _class_name(class_names, int(class_id)), color="black", fontsize=8, backgroundcolor=color)
    return ax


def draw_yolo_masks(
    image_path: str | Path,
    label_path: str | Path,
    *,
    class_names=None,
    ax=None,
    alpha: float = 0.28,
    linewidth: float = 1.5,
):
    """Draw YOLO segmentation polygon labels on one image."""

    plt, patches = _require_matplotlib()
    image = _open_image(image_path)
    width, height = image.size
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image)
    ax.axis("off")

    for index, row in enumerate(_read_label_lines(label_path)):
        if len(row) < 7:
            continue
        class_id = int(row[0])
        coords = row[1:]
        points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
        color = f"C{index % 10}"
        polygon = patches.Polygon(points, closed=True, fill=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=linewidth)
        outline = patches.Polygon(points, closed=True, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(polygon)
        ax.add_patch(outline)
        if points:
            ax.text(points[0][0], points[0][1], _class_name(class_names, class_id), color="white", fontsize=8)
    return ax


def plot_training_curves(
    results_csv_path: str | Path,
    *,
    metric_columns: Sequence[str] | None = None,
    include_training: bool = False,
    title: str | None = None,
):
    """Plot selected metrics from an Ultralytics-style training CSV.

    Set `include_training=True` to add matching `train/...` columns for
    requested `val/...` columns when those training columns are present.
    """

    plt, _ = _require_matplotlib()
    path = Path(results_csv_path)
    if not path.exists():
        print(f"No results CSV found at {path}")
        return None

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"No rows found in {path}")
        return None

    columns = list(rows[0].keys())
    if metric_columns is None:
        validation_metrics = [
            column
            for column in columns
            if column.startswith("metrics/") or column in {"accuracy_top1", "val/accuracy_top1"}
        ][:6]
        metric_columns = validation_metrics
        if include_training:
            loss_pairs: list[str] = []
            for column in columns:
                if not column.startswith("val/") or "loss" not in column:
                    continue
                train_column = "train/" + column.removeprefix("val/")
                if train_column in columns and train_column not in loss_pairs:
                    loss_pairs.append(train_column)
                if column not in loss_pairs:
                    loss_pairs.append(column)
            metric_columns = loss_pairs + list(metric_columns)
    else:
        metric_columns = list(metric_columns)
        if include_training:
            expanded_columns: list[str] = []
            for column in metric_columns:
                if column.startswith("val/"):
                    train_column = "train/" + column.removeprefix("val/")
                    if train_column in columns and train_column not in expanded_columns:
                        expanded_columns.append(train_column)
                if column not in expanded_columns:
                    expanded_columns.append(column)
            metric_columns = expanded_columns

    metric_columns = [column for column in metric_columns if column in columns]
    if not metric_columns:
        print(f"No requested metric columns found in {path}")
        return None

    epochs = [float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for column in metric_columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except Exception:
                values.append(float("nan"))
        ax.plot(epochs, values, marker="o", linewidth=1.5, label=column)
    ax.set_xlabel("epoch")
    ax.set_title(title or path.name)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return ax


def plot_detection_training_summary(
    results_csv_path: str | Path,
    final_metrics: dict[str, object],
    *,
    title: str = "Detection fine-tuning summary",
):
    """Plot detection loss history beside the final validation metrics."""

    plt, _ = _require_matplotlib()
    path = Path(results_csv_path)
    if not path.exists():
        print(f"No results CSV found at {path}")
        return None

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"No rows found in {path}")
        return None

    epochs = [float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]
    loss_specs = [
        ("train/box_loss", "box loss", "C0"),
        ("train/cls_loss", "classification loss", "C1"),
        ("train/dfl_loss", "distribution focal loss", "C2"),
    ]
    metric_specs = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("mAP50", "mAP@0.50"),
        ("mAP50-95", "mAP@0.50–0.95"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for column, label, color in loss_specs:
        if column not in rows[0]:
            continue
        values = [float(row[column]) for row in rows]
        axes[0].plot(epochs, values, color=color, linewidth=2, label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss (lower is better on the training set)")
    axes[0].set_title("What the optimizer reduced")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    metric_labels = [label for key, label in metric_specs if key in final_metrics]
    metric_values = [float(final_metrics[key]) for key, _ in metric_specs if key in final_metrics]
    bars = axes[1].bar(metric_labels, metric_values, color=["C0", "C1", "C2", "C3"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("score")
    axes[1].set_title("Final held-out validation scores")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metric_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            min(0.97, value + 0.03),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return axes


def plot_confusion_matrix(
    matrix,
    class_names: Sequence[str],
    *,
    normalize: bool = False,
    title: str = "Confusion matrix",
):
    """Plot a confusion matrix from a nested list or NumPy array."""

    plt, _ = _require_matplotlib()
    import numpy as np

    values = np.asarray(matrix, dtype=float)
    if normalize:
        denom = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denom, out=np.zeros_like(values), where=denom != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.show()
    return ax


def plot_sam3_result(
    image_path: str | Path,
    result: str | Path | dict,
    *,
    score_threshold: float = 0.0,
    title: str | None = None,
):
    """Plot a cached reference or live SAM3 result with masks, polygons, and boxes."""

    if not isinstance(result, dict):
        with Path(result).open("r", encoding="utf-8") as handle:
            result = json.load(handle)

    plt, patches = _require_matplotlib()
    import numpy as np
    image = _open_image(image_path)
    width, height = image.size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image)
    ax.axis("off")
    scores = result.get("scores", [])

    raw_masks = result.get("masks", [])
    if raw_masks is not None and len(raw_masks):
        masks = np.asarray(raw_masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        for index, mask in enumerate(masks):
            score = float(scores[index]) if index < len(scores) else 1.0
            if score < score_threshold:
                continue
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                continue
            if mask.shape != (height, width):
                from PIL import Image

                mask = np.asarray(
                    Image.fromarray(mask.astype("float32"), mode="F").resize(
                        (width, height),
                        resample=Image.Resampling.NEAREST,
                    )
                )
            color = np.asarray(plt.get_cmap("tab10")(index % 10))
            overlay = np.zeros((height, width, 4), dtype=float)
            overlay[..., :3] = color[:3]
            overlay[..., 3] = (mask > 0.5) * 0.30
            ax.imshow(overlay)

    for index, box in enumerate(result.get("boxes", [])):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < score_threshold:
            continue
        x0, y0, x1, y1 = box
        if max(box) <= 1.0:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        color = f"C{index % 10}"
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, f"{score:.2f}", color="black", fontsize=8, backgroundcolor=color)

    for index, polygon in enumerate(result.get("polygons", [])):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < score_threshold:
            continue
        points = [(x * width, y * height) if max(x, y) <= 1.0 else (x, y) for x, y in polygon]
        if len(points) >= 3:
            color = f"C{index % 10}"
            ax.add_patch(patches.Polygon(points, closed=True, fill=True, facecolor=color, edgecolor=color, alpha=0.25))
            ax.add_patch(patches.Polygon(points, closed=True, fill=False, edgecolor=color, linewidth=1.5))

    ax.set_title(title or result.get("prompt", "SAM3 result"))
    plt.tight_layout()
    plt.show()
    return ax
