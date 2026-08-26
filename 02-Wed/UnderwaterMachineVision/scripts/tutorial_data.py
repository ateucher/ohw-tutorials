"""Dataset helpers for the underwater vision tutorial."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Broad visual groups for the optional multi-class detection appendix. These
# classes make a small workshop dataset trainable; they are not a taxonomic
# hierarchy and should be replaced with project-specific expert labels in a
# scientific study.
FATHOMNET_COARSE_DETECTION_GROUPS = {
    "fish": {
        "Anoplopoma",
        "Bathyraja kincaidii",
        "Lamprogrammus",
        "Pleuronectiformes",
        "Sebastes",
        "Sebastes melanostomus",
    },
    "crustacean": {
        "Chionoecetes tanneri",
        "Euphausia",
        "Eusergestes similis",
        "Phronima sedentaria",
        "shrimp",
        "Sternostylus perarmatus",
    },
    "echinoderm": {
        "Abyssocucumis abyssorum",
        "Actinopyga echinites",
        "Apostichopus leukothele",
        "Asteroidea",
        "Benthodytes",
        "Crinoidea",
        "Cystocrepis setigera",
        "Echinocrepis rostrata",
        "Elpidia",
        "Ophiocreas oedipus",
        "Ophiuroidea",
        "Peniagone",
        "Peniagone vitrea",
        "Psolus squamatus",
        "Pterasteridae",
        "Rathbunaster californicus",
        "Strongylocentrotus fragilis",
        "Synallactidae",
    },
    "gelatinous": {
        "Aegina",
        "Aglantha digitale",
        "Apolemia",
        "Atolla",
        "Bathochordaeus",
        "Bathochordaeus stygius",
        "Bathocyroe fosteri",
        "Benthocodon",
        "Beroe",
        "Beroe cucumis",
        "Deepstaria enigmatica",
        "Desmophyes haematogaster",
        "Haliscera conica",
        "Lampocteis cruentiventer",
        "Nanomia bijuga",
        "Physophora hydrostatica",
        "Poralia rufescens",
        "Prayidae",
        "Pyrosoma",
        "Salpida",
        "Sphaeronectes haddocki",
        "Stellamedusa ventana",
    },
    "sponge/cnidarian": {
        "Actiniaria",
        "Actiniidae",
        "Actinernus",
        "Alcyonacea",
        "black coral",
        "Dofleinia",
        "Farrea",
        "Heterochone calyx",
        "Hexactinellida",
        "Iosactis vagabunda",
        "Isididae",
        "Isosicyonis",
        "Keratoisis",
        "Keratoisididae",
        "Keratoisidinae",
        "Metallogorgia melanotrichos",
        "Narella",
        "Paragorgia arborea",
        "Parazoanthidae",
        "Pennatulacea",
        "Porifera",
        "Psamminidae",
        "sponge",
        "Staurocalyptus",
        "Umbellula",
    },
}


def load_manifest(bundle_root: str | Path) -> dict:
    """Load the bundle manifest."""

    manifest_path = Path(bundle_root) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_runtime_yolo_yaml(dataset_dir: str | Path, output_name: str = "dataset.runtime.yaml") -> Path:
    """Write a portable Ultralytics YAML with an absolute `path` entry."""

    dataset_path = Path(dataset_dir).resolve()
    source_yaml = dataset_path / "dataset.yaml"
    with source_yaml.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["path"] = str(dataset_path)

    runtime_yaml = dataset_path / output_name
    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return runtime_yaml


def get_task_paths(task: str, bundle_root: str | Path) -> dict[str, Path]:
    """Return important paths for a tutorial task.

    `task` may be `classification`, `detect`, `segment`, `coco`, `sam3`, or
    `cached_training`.
    """

    root = Path(bundle_root).resolve()
    normalized = task.lower().strip()
    if normalized in {"classification", "classify", "cls"}:
        return {"root": root / "classification_crops"}
    if normalized in {"detect", "detection", "bbox"}:
        dataset_dir = root / "yolo_detect_binary"
        return {"root": dataset_dir, "yaml": make_runtime_yolo_yaml(dataset_dir)}
    if normalized in {"segment", "segmentation", "seg"}:
        dataset_dir = root / "yolo_segment_binary"
        return {"root": dataset_dir, "yaml": make_runtime_yolo_yaml(dataset_dir)}
    if normalized == "coco":
        return {"root": root / "coco", "json": root / "coco" / "subset.json"}
    if normalized == "sam3":
        return {"root": root / "sam3_cached_outputs", "index": root / "sam3_cached_outputs" / "index.json"}
    if normalized == "cached_training":
        return {"root": root / "cached_training"}
    if normalized == "licenses":
        return {"root": root / "licenses", "attribution": root / "licenses" / "attribution.csv"}
    raise ValueError(f"Unknown task: {task}")


def summarize_classification_dataset(dataset_root: str | Path) -> dict[str, object]:
    """Count images by split and class in an ImageFolder-style dataset."""

    root = Path(dataset_root)
    summary: dict[str, object] = {"root": str(root), "splits": {}}
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_counts = {
            class_dir.name: len(_image_files(class_dir))
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir())
        }
        summary["splits"][split_dir.name] = class_counts
    return summary


def classification_examples_by_class(
    dataset_root: str | Path,
    *,
    split: str = "val",
    examples_per_class: int = 1,
    max_classes: int | None = None,
) -> list[dict[str, object]]:
    """Select labelled classification images from an ImageFolder-style split.

    Each returned item has an `image_path` and `class_name`. The helper keeps
    the notebook focused on the modelling question: here the target is one class
    label per image, with no boxes or masks.
    """

    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        return []

    rows: list[dict[str, object]] = []
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    for class_dir in class_dirs:
        for image_path in _image_files(class_dir)[:examples_per_class]:
            rows.append({"image_path": image_path, "class_name": class_dir.name})
    return rows


def yolo_label_instance_count(label_path: str | Path) -> int:
    """Count non-empty instance rows in one YOLO label file."""

    path = Path(label_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _matching_image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    matches = [
        image_path
        for image_path in image_dir.glob(f"{label_path.stem}.*")
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(matches)[0] if matches else None


def select_yolo_examples(
    dataset_root: str | Path,
    *,
    split: str = "val",
    min_instances: int = 2,
    limit: int = 4,
) -> list[dict[str, object]]:
    """Select YOLO examples with several labelled objects in one image.

    Multi-object examples are useful early in the tutorial because they make
    the distinction between image classification, object detection, and
    instance segmentation visible without needing a long explanation.
    """

    root = Path(dataset_root)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return []

    examples: list[dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        instance_count = yolo_label_instance_count(label_path)
        if instance_count < min_instances:
            continue
        image_path = _matching_image_for_label(label_path, image_dir)
        if image_path is None:
            continue
        examples.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "instance_count": instance_count,
            }
        )

    examples.sort(key=lambda item: (-int(item["instance_count"]), str(item["label_path"])))
    return examples[:limit]


def select_coco_category_examples(
    coco_json_path: str | Path,
    image_roots: str | Path | list[str | Path],
    *,
    limit: int = 6,
    min_annotations: int = 1,
) -> list[dict[str, object]]:
    """Select full images with their ground-truth COCO category labels.

    This is a classification-style view of full images: it shows which taxa or
    object categories are annotated as present, but it intentionally hides the
    boxes and masks so the geometric tasks remain visually distinct.
    """

    with Path(coco_json_path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    if isinstance(image_roots, (str, Path)):
        roots = [Path(image_roots)]
    else:
        roots = [Path(root) for root in image_roots]

    images_by_stem: dict[str, Path] = {}
    for root in roots:
        for image_path in _image_files(root):
            images_by_stem.setdefault(image_path.stem, image_path)

    names_by_category_id = {category["id"]: category["name"] for category in coco.get("categories", [])}
    image_records = {image["id"]: image for image in coco.get("images", [])}
    category_names_by_image: dict[int, list[str]] = {}
    annotation_count_by_image: Counter[int] = Counter()

    for annotation in coco.get("annotations", []):
        image_id = annotation["image_id"]
        category_name = names_by_category_id.get(annotation["category_id"], str(annotation["category_id"]))
        category_names_by_image.setdefault(image_id, []).append(category_name)
        annotation_count_by_image[image_id] += 1

    rows: list[dict[str, object]] = []
    for image_id, category_names in category_names_by_image.items():
        if annotation_count_by_image[image_id] < min_annotations:
            continue
        image_record = image_records.get(image_id)
        if not image_record:
            continue
        image_stem = Path(image_record["file_name"]).stem
        image_path = images_by_stem.get(image_stem)
        if image_path is None:
            continue
        rows.append(
            {
                "image_path": image_path,
                "category_names": sorted(set(category_names)),
                "annotation_count": annotation_count_by_image[image_id],
            }
        )

    rows.sort(key=lambda item: (-int(item["annotation_count"]), str(item["image_path"])))
    return rows[:limit]


def _load_dataset_yaml(dataset_yaml: str | Path) -> tuple[Path, dict]:
    yaml_path = Path(dataset_yaml).resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    root = Path(payload.get("path", yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, payload


def validate_yolo_dataset(dataset_yaml: str | Path, *, task: str = "detect") -> dict[str, object]:
    """Check that a YOLO dataset has image/label pairs and sane labels."""

    root, payload = _load_dataset_yaml(dataset_yaml)
    names = payload.get("names", {})
    class_count = len(names) if isinstance(names, (list, dict)) else int(payload.get("nc", 0))
    line_lengths = Counter()
    invalid_lines: list[str] = []
    split_summaries: dict[str, dict[str, object]] = {}

    for split in ["train", "val", "test"]:
        split_value = payload.get(split)
        if not split_value:
            continue
        image_dir = root / split_value
        label_dir = root / str(split_value).replace("images", "labels", 1)
        image_paths = _image_files(image_dir) if image_dir.exists() else []
        label_paths = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        empty_labels = 0

        for label_path in label_paths:
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                line_lengths[len(parts)] += 1
                try:
                    class_id = int(float(parts[0]))
                    values = [float(value) for value in parts[1:]]
                except Exception:
                    invalid_lines.append(f"{label_path}:{line_number}: non-numeric label")
                    continue
                if class_id < 0 or (class_count and class_id >= class_count):
                    invalid_lines.append(f"{label_path}:{line_number}: bad class id {class_id}")
                if any(value < 0 or value > 1 for value in values):
                    invalid_lines.append(f"{label_path}:{line_number}: coordinates outside [0, 1]")
                if task == "detect" and len(parts) != 5:
                    invalid_lines.append(f"{label_path}:{line_number}: detection labels need 5 values")
                if task == "segment" and (len(parts) < 7 or len(values) % 2 != 0):
                    invalid_lines.append(f"{label_path}:{line_number}: segmentation labels need class plus >= 3 xy points")

        split_summaries[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "empty_label_files": empty_labels,
            "missing_label_files": sorted(image_stems - label_stems)[:10],
            "labels_without_images": sorted(label_stems - image_stems)[:10],
        }

    return {
        "root": str(root),
        "class_count": class_count,
        "names": names,
        "splits": split_summaries,
        "label_line_lengths": dict(line_lengths),
        "invalid_lines": invalid_lines[:25],
        "valid": not invalid_lines,
    }


def summarize_dataset(bundle_root: str | Path) -> dict[str, object]:
    """Summarise all major bundle components."""

    root = Path(bundle_root)
    return {
        "manifest": load_manifest(root),
        "classification": summarize_classification_dataset(root / "classification_crops"),
        "detection": validate_yolo_dataset(make_runtime_yolo_yaml(root / "yolo_detect_binary"), task="detect"),
        "segmentation": validate_yolo_dataset(make_runtime_yolo_yaml(root / "yolo_segment_binary"), task="segment"),
    }


def _read_detection_rows(label_path: Path) -> list[list[float]]:
    """Read YOLO detection rows as floats, skipping malformed lines."""

    rows: list[list[float]] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    return rows


def _detection_example_score(label_path: Path) -> float:
    """Score a YOLO label file for a tiny overfit/debugging subset.

    Larger boxes and images with several labelled objects make better debugging
    examples because the learner can visually confirm that labels and images
    line up. Empty label files are given a negative score and are avoided by
    the `easy` selection strategy.
    """

    rows = _read_detection_rows(label_path)
    if not rows:
        return -1.0
    areas = [max(0.0, row[3]) * max(0.0, row[4]) for row in rows]
    return max(areas) + 0.15 * sum(areas) + 0.08 * len(areas)


def _largest_detection_area(label_path: Path) -> tuple[float, int]:
    """Return `(largest_area, instance_count)` for a detection label file."""

    rows = _read_detection_rows(label_path)
    if not rows:
        return -1.0, 0
    areas = [max(0.0, row[3]) * max(0.0, row[4]) for row in rows]
    return max(areas), len(rows)


def make_tiny_detection_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    train_images: int = 8,
    val_images: int = 8,
    selection_strategy: str = "first",
    val_from_train: bool = False,
) -> Path:
    """Copy a tiny YOLO detection dataset for quick overfit/debugging labs.

    The tiny dataset is intentionally created under `tmp/` by the notebook. It
    should not be committed; it is a disposable local teaching artefact.

    Parameters
    ----------
    selection_strategy:
        Use `"first"` for deterministic alphabetical selection, `"easy"` to
        prefer visible multi-object examples, or `"large"` to prefer examples
        with one very large labelled object. The large-object strategy is useful
        for overfit checks because tiny objects can produce weak gradients and
        hard-to-interpret visual results.
    val_from_train:
        If true, copy the selected training examples into the validation split
        too. That is not a valid generalisation estimate, but it is the right
        setup for the specific question "can this model memorise a few images?"
    """

    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    def select_labels(split: str, count: int) -> list[Path]:
        label_paths = sorted((source_root / "labels" / split).glob("*.txt"))
        if selection_strategy == "first":
            return label_paths[:count]
        if selection_strategy == "easy":
            scored = [
                (_detection_example_score(label_path), label_path)
                for label_path in label_paths
                if _matching_image_for_label(label_path, source_root / "images" / split) is not None
            ]
            return [label_path for _, label_path in sorted(scored, key=lambda item: (-item[0], item[1].name))[:count]]
        if selection_strategy == "large":
            scored = [
                (*_largest_detection_area(label_path), label_path)
                for label_path in label_paths
                if _matching_image_for_label(label_path, source_root / "images" / split) is not None
            ]
            return [
                label_path
                for _, _, label_path in sorted(scored, key=lambda item: (-item[0], item[1], item[2].name))[:count]
            ]
        raise ValueError(f"Unknown selection_strategy: {selection_strategy!r}")

    def copy_split(label_paths: list[Path], *, source_split: str, destination_split: str) -> None:
        (output_root / "images" / destination_split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / destination_split).mkdir(parents=True, exist_ok=True)
        for label_path in label_paths:
            image_path = _matching_image_for_label(label_path, source_root / "images" / source_split)
            if image_path is None:
                continue
            shutil.copy2(image_path, output_root / "images" / destination_split / image_path.name)
            shutil.copy2(label_path, output_root / "labels" / destination_split / label_path.name)

    train_labels = select_labels("train", train_images)
    copy_split(train_labels, source_split="train", destination_split="train")

    if val_from_train:
        copy_split(train_labels[:val_images], source_split="train", destination_split="val")
    else:
        val_labels = select_labels("val", val_images)
        copy_split(val_labels, source_split="val", destination_split="val")

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "underwater organism"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def make_detection_finetune_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    positive_train_images: int = 24,
    negative_train_images: int = 8,
    val_images: int | None = None,
) -> Path:
    """Build the deterministic detection subset used in the object-detection tutorial.

    Positive examples are ranked using the same visible-object heuristic as
    ``make_tiny_detection_dataset(..., selection_strategy="easy")``. Empty
    label files can be included separately when the source dataset contains
    background-only frames. Validation examples always come from the original
    validation split; ``None`` keeps that complete split.

    The output is disposable and is intended to live under ``tmp/``.
    """

    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    train_label_dir = source_root / "labels" / "train"
    train_image_dir = source_root / "images" / "train"
    train_labels = sorted(train_label_dir.glob("*.txt"))

    positive_candidates = [
        label_path
        for label_path in train_labels
        if _read_detection_rows(label_path)
        and _matching_image_for_label(label_path, train_image_dir) is not None
    ]
    positive_candidates.sort(
        key=lambda label_path: (-_detection_example_score(label_path), label_path.name)
    )
    negative_candidates = [
        label_path
        for label_path in train_labels
        if not _read_detection_rows(label_path)
        and _matching_image_for_label(label_path, train_image_dir) is not None
    ]

    selected_train = (
        positive_candidates[:positive_train_images]
        + negative_candidates[:negative_train_images]
    )

    val_label_dir = source_root / "labels" / "val"
    val_image_dir = source_root / "images" / "val"
    selected_val = [
        label_path
        for label_path in sorted(val_label_dir.glob("*.txt"))
        if _matching_image_for_label(label_path, val_image_dir) is not None
    ]
    if val_images is not None:
        selected_val = selected_val[:val_images]

    def copy_split(
        label_paths: list[Path],
        *,
        source_image_dir: Path,
        destination_split: str,
    ) -> None:
        image_output = output_root / "images" / destination_split
        label_output = output_root / "labels" / destination_split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for label_path in label_paths:
            image_path = _matching_image_for_label(label_path, source_image_dir)
            if image_path is None:
                continue
            shutil.copy2(image_path, image_output / image_path.name)
            shutil.copy2(label_path, label_output / label_path.name)

    copy_split(
        selected_train,
        source_image_dir=train_image_dir,
        destination_split="train",
    )
    copy_split(
        selected_val,
        source_image_dir=val_image_dir,
        destination_split="val",
    )

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "underwater organism"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def make_coarse_multiclass_detection_dataset(
    coco_json_path: str | Path,
    source_detection_root: str | Path,
    output_root: str | Path,
    *,
    min_box_area: float = 0.005,
) -> dict[str, object]:
    """Build a five-class YOLO detection dataset from the tutorial COCO data.

    The source bundle stores one-class YOLO labels for the main lesson and the
    original concept names in COCO format. This helper maps selected concepts
    into five broad visual groups, preserves the original train/validation
    split, converts COCO ``[x, y, width, height]`` boxes into normalized YOLO
    rows, and writes a disposable dataset under ``tmp/``.

    Categories outside the five workshop groups are deliberately omitted.
    Images that contain no retained categories remain in the dataset with an
    empty label file, where they act as background examples for these targets.
    """

    coco_json_path = Path(coco_json_path)
    source_root = Path(source_detection_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    with coco_json_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    class_names = {
        class_id: class_name
        for class_id, class_name in enumerate(FATHOMNET_COARSE_DETECTION_GROUPS)
    }
    class_ids = {class_name: class_id for class_id, class_name in class_names.items()}
    category_to_class = {
        category_name: class_name
        for class_name, category_names in FATHOMNET_COARSE_DETECTION_GROUPS.items()
        for category_name in category_names
    }
    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    image_records = {
        Path(image["file_name"]).stem: image
        for image in coco.get("images", [])
    }
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    instance_counts = {split: Counter() for split in ("train", "val")}
    image_counts = {split: Counter() for split in ("train", "val")}
    empty_label_files = Counter()
    filtered_small_instances = Counter()
    unmapped_categories = Counter()
    reviewed_negative_images = Counter()

    for split in ("train", "val"):
        source_image_dir = source_root / "images" / split
        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in _image_files(source_image_dir):
            shutil.copy2(image_path, output_image_dir / image_path.name)
            rows: list[str] = []
            classes_in_image: set[str] = set()
            image_record = image_records.get(image_path.stem)
            if image_record is None:
                # Reviewed negative frames were added outside the original COCO
                # subset, so they intentionally have no annotations to remap.
                reviewed_negative_images[split] += 1
                empty_label_files[split] += 1
                (output_label_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
                continue

            image_width = float(image_record["width"])
            image_height = float(image_record["height"])

            for annotation in annotations_by_image_id.get(int(image_record["id"]), []):
                category_name = category_names.get(int(annotation["category_id"]), "unknown")
                class_name = category_to_class.get(category_name)
                if class_name is None:
                    unmapped_categories[category_name] += 1
                    continue

                x, y, width, height = (float(value) for value in annotation["bbox"])
                x0 = max(0.0, min(image_width, x))
                y0 = max(0.0, min(image_height, y))
                x1 = max(0.0, min(image_width, x + width))
                y1 = max(0.0, min(image_height, y + height))
                box_width = x1 - x0
                box_height = y1 - y0
                normalized_width = box_width / image_width
                normalized_height = box_height / image_height
                if box_width <= 0 or box_height <= 0:
                    continue
                if normalized_width * normalized_height < min_box_area:
                    filtered_small_instances[class_name] += 1
                    continue

                x_center = ((x0 + x1) / 2.0) / image_width
                y_center = ((y0 + y1) / 2.0) / image_height
                rows.append(
                    f"{class_ids[class_name]} {x_center:.6f} {y_center:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )
                instance_counts[split][class_name] += 1
                classes_in_image.add(class_name)

            for class_name in classes_in_image:
                image_counts[split][class_name] += 1
            if not rows:
                empty_label_files[split] += 1
            (output_label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(rows),
                encoding="utf-8",
            )

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": class_names,
                "tutorial_task": "coarse multiclass detection",
                "tutorial_note": (
                    "Broad visual workshop labels derived from selected FathomNet concepts; "
                    "not a taxonomic hierarchy."
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return {
        "yaml_path": str(yaml_path),
        "class_names": class_names,
        "instance_counts": {
            split: {class_name: instance_counts[split][class_name] for class_name in class_names.values()}
            for split in ("train", "val")
        },
        "image_counts": {
            split: {class_name: image_counts[split][class_name] for class_name in class_names.values()}
            for split in ("train", "val")
        },
        "empty_label_files": dict(empty_label_files),
        "reviewed_negative_images": dict(reviewed_negative_images),
        "filtered_small_instances": dict(filtered_small_instances),
        "unmapped_annotation_counts": dict(sorted(unmapped_categories.items())),
    }
