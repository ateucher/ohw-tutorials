"""Runtime setup helpers for the underwater vision tutorial.

The notebook is designed for Google Colab, but these helpers also work on a
local workstation. Heavy packages such as torch are imported lazily so the
module can be inspected even before the teaching environment is fully ready.
"""

from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


PINNED_TORCH_STACK = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
}

PINNED_PIP_PACKAGES = {
    "ultralytics": "ultralytics==8.4.41",
}

PYTORCH_CU128_INDEX_URL = "https://download.pytorch.org/whl/cu128"

DEFAULT_IMPORTS = {
    "torch": "torch==2.11.0",
    "torchvision": "torchvision==0.26.0",
    "ultralytics": "ultralytics==8.4.41",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "huggingface_hub": "huggingface_hub",
}

GITHUB_REPO_URL = "https://github.com/oceanhackweek/ohw-tutorials"
GITHUB_BRANCH = "OHW26"
PROJECT_DIR_NAME = "ohw-tutorials"
TUTORIAL_RELATIVE_DIR = Path("02-Wed") / "UnderwaterMachineVision"
BUNDLE_URL = (
    "https://raw.githubusercontent.com/oceanhackweek/ohw-tutorials/OHW26/"
    "02-Wed/UnderwaterMachineVision/data/fathomnet_underwater_tutorial_bundle.zip"
)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the nearest directory containing this tutorial's local files."""

    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (
            (candidate / "scripts" / "tutorial_setup.py").exists()
            and (candidate / "data" / "fathomnet_underwater_tutorial_bundle.zip").exists()
        ):
            return candidate
        nested = candidate / TUTORIAL_RELATIVE_DIR
        if (
            (nested / "scripts" / "tutorial_setup.py").exists()
            and (nested / "data" / "fathomnet_underwater_tutorial_bundle.zip").exists()
        ):
            return nested
    return start_path


def set_reproducible_seed(seed: int = 42) -> int:
    """Set common pseudo-random seeds and return the seed that was used."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed


def _run_text(command: list[str], timeout: int = 10) -> str | None:
    """Run a short diagnostic command and return stdout if it succeeds."""

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _installed_distribution_version(distribution_name: str) -> str | None:
    """Return an installed package version without importing the package."""

    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_matches(installed: str | None, expected: str) -> bool:
    """Return true when `installed` has the expected public version prefix."""

    if installed is None:
        return False
    return installed == expected or installed.startswith(f"{expected}+")


def dependency_versions() -> dict[str, object]:
    """Report the package versions most likely to cause Colab incompatibilities."""

    payload: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": _installed_distribution_version("torch"),
        "torchvision": _installed_distribution_version("torchvision"),
        "ultralytics": _installed_distribution_version("ultralytics"),
        "expected": {
            "torch": PINNED_TORCH_STACK["torch"],
            "torchvision": PINNED_TORCH_STACK["torchvision"],
            "ultralytics": "8.4.41",
        },
        "torch_cuda": None,
        "cuda_available": False,
        "status": "unchecked",
        "mismatches": [],
    }

    for package_name, expected_version in PINNED_TORCH_STACK.items():
        installed = payload.get(package_name)
        if not _version_matches(str(installed) if installed else None, expected_version):
            payload["mismatches"].append(
                f"{package_name}: expected {expected_version}, found {installed or 'not installed'}"
            )
    ultralytics_version = payload.get("ultralytics")
    if not _version_matches(str(ultralytics_version) if ultralytics_version else None, "8.4.41"):
        payload["mismatches"].append(
            f"ultralytics: expected 8.4.41, found {ultralytics_version or 'not installed'}"
        )

    if importlib.util.find_spec("torch") is not None:
        import torch

        payload["torch_cuda"] = getattr(torch.version, "cuda", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())

    payload["status"] = "ok" if not payload["mismatches"] else "mismatch"
    return payload


def detect_runtime() -> dict[str, object]:
    """Summarise the active notebook/runtime environment."""

    in_colab = "google.colab" in sys.modules
    payload: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "in_colab": in_colab,
        "has_cuda": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "torch_version": None,
        "torchvision_version": _installed_distribution_version("torchvision"),
        "ultralytics_version": _installed_distribution_version("ultralytics"),
        "nvidia_smi": None,
    }

    if importlib.util.find_spec("torch") is not None:
        import torch

        payload["torch_version"] = torch.__version__
        payload["has_cuda"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)

    payload["nvidia_smi"] = _run_text(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        timeout=5,
    )
    return payload


def print_runtime_summary(runtime: dict[str, object] | None = None) -> dict[str, object]:
    """Pretty-print the runtime summary and return it for later cells."""

    runtime = runtime or detect_runtime()
    print(json.dumps(runtime, indent=2))
    return runtime


def ensure_dependencies(
    packages: dict[str, str] | None = None,
    *,
    install: bool = False,
    extra_pip_args: Iterable[str] = (),
) -> dict[str, object]:
    """Check or install notebook dependencies.

    Parameters
    ----------
    packages:
        Mapping from import name to pip package name.
    install:
        If True, missing packages are installed into the active interpreter.
        Keep this False while merely inspecting the repository.
    extra_pip_args:
        Extra arguments passed to pip, for example `("--quiet",)`.
    """

    packages = dict(packages or DEFAULT_IMPORTS)
    pinned_imports = {"torch", "torchvision", *PINNED_PIP_PACKAGES.keys()}
    result: dict[str, object] = {
        "missing": [],
        "installed": [],
        "restart_recommended": False,
        "version_report_before": {
            "torch": _installed_distribution_version("torch"),
            "torchvision": _installed_distribution_version("torchvision"),
            "ultralytics": _installed_distribution_version("ultralytics"),
        },
    }

    if install:
        torch_stack_mismatch = [
            package_name
            for package_name, expected_version in PINNED_TORCH_STACK.items()
            if not _version_matches(_installed_distribution_version(package_name), expected_version)
        ]
        if torch_stack_mismatch:
            if any(package_name in sys.modules for package_name in PINNED_TORCH_STACK):
                result["restart_recommended"] = True
                print(
                    "Torch or Torchvision is already imported in this runtime. "
                    "If pip changes these packages, restart the runtime before training."
                )
            torch_specs = [
                f"{package_name}=={expected_version}"
                for package_name, expected_version in PINNED_TORCH_STACK.items()
            ]
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                *extra_pip_args,
                "--index-url",
                PYTORCH_CU128_INDEX_URL,
                *torch_specs,
            ]
            print("Installing pinned PyTorch stack:", " ".join(torch_specs))
            subprocess.check_call(command)
            result["installed"].extend(torch_specs)
            importlib.invalidate_caches()

        for import_name, pip_spec in PINNED_PIP_PACKAGES.items():
            expected_version = pip_spec.split("==", 1)[1]
            if not _version_matches(_installed_distribution_version(import_name), expected_version):
                command = [sys.executable, "-m", "pip", "install", *extra_pip_args, pip_spec]
                print("Installing pinned package:", pip_spec)
                subprocess.check_call(command)
                result["installed"].append(pip_spec)
                importlib.invalidate_caches()

    missing = [
        pip_name
        for import_name, pip_name in packages.items()
        if import_name not in pinned_imports and importlib.util.find_spec(import_name) is None
    ]
    if missing and install:
        command = [sys.executable, "-m", "pip", "install", *extra_pip_args, *missing]
        print("Installing:", " ".join(missing))
        subprocess.check_call(command)
        result["installed"].extend(missing)
        importlib.invalidate_caches()

    result["missing"] = [
        pip_name
        for import_name, pip_name in packages.items()
        if importlib.util.find_spec(import_name) is None
    ]
    result["version_report_after"] = dependency_versions()

    if result["version_report_after"]["mismatches"]:
        print("Pinned dependency mismatches:")
        for mismatch in result["version_report_after"]["mismatches"]:
            print(" -", mismatch)
    elif result["missing"]:
        print("Missing optional dependencies:", ", ".join(result["missing"]))
    else:
        print("All requested dependencies are importable and pinned versions match.")
    if result["restart_recommended"]:
        print("Restart the runtime before running training cells, then rerun setup from the top.")
    return result


def download_tutorial_bundle(
    *,
    bundle_url: str | None = None,
    bundle_zip_path: str | Path | None = None,
    output_dir: str | Path = "data/fathomnet_underwater_tutorial_bundle",
    force: bool = False,
) -> Path:
    """Download or unpack the prebuilt tutorial data bundle.

    The lookup order is:
    1. Existing `output_dir` with a manifest, unless `force=True`.
    2. Explicit `bundle_zip_path`.
    3. Local repo zip at `data/fathomnet_underwater_tutorial_bundle.zip`.
    4. `bundle_url`.
    """

    output_path = Path(output_dir).expanduser().resolve()
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists() and not force:
        return output_path

    if force and output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = find_repo_root()
    candidates: list[Path] = []
    if bundle_zip_path is not None:
        candidates.append(Path(bundle_zip_path).expanduser())
    candidates.append(repo_root / "data" / "fathomnet_underwater_tutorial_bundle.zip")

    zip_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if zip_path is None and bundle_url:
        zip_path = output_path.parent / "fathomnet_underwater_tutorial_bundle.zip"
        print(f"Downloading tutorial bundle from {bundle_url}")
        urllib.request.urlretrieve(bundle_url, zip_path)

    if zip_path is None:
        raise FileNotFoundError(
            "Could not find a tutorial bundle zip. Set BUNDLE_URL or build "
            "data/fathomnet_underwater_tutorial_bundle.zip first."
        )

    print(f"Unpacking {zip_path} -> {output_path.parent}")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_path.parent)

    if not manifest_path.exists():
        nested = output_path.parent / "fathomnet_underwater_tutorial_bundle" / "manifest.json"
        if nested.exists() and nested.parent != output_path:
            if output_path.exists():
                shutil.rmtree(output_path)
            nested.parent.rename(output_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle unpacked, but no manifest was found at {manifest_path}")
    return output_path


def default_train_args(
    *,
    n_epochs: int = 10,
    imgsz: int = 320,
    batch: int = 8,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    patience: int = 3,
    workers: int = 0,
    project: str | Path = "runs/tutorial",
    name: str = "experiment",
    seed: int = 42,
) -> dict[str, object]:
    """Return the compact training-argument dictionary used by notebook cells.

    The notebook exposes the values passed into this helper instead of the
    helper body itself. That keeps the notebook focused on the modelling knobs:
    `n_epochs`, `imgsz`, `batch`, `lr0`, `optimizer`, and `patience`.
    """

    return {
        "epochs": int(n_epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "lr0": float(lr0),
        "optimizer": str(optimizer),
        "patience": int(patience),
        "workers": int(workers),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        "save": True,
        "verbose": False,
    }


def _resolve_repo_root(namespace: dict[str, object]) -> Path:
    """Find the tutorial repo root, cloning in Colab if needed."""

    repo_root = namespace.get("REPO_ROOT")
    if repo_root is not None:
        return Path(repo_root).resolve()

    candidate = find_repo_root()
    if (
        (candidate / "scripts" / "tutorial_setup.py").exists()
        and (candidate / "data" / "fathomnet_underwater_tutorial_bundle.zip").exists()
    ):
        return candidate

    if "google.colab" in sys.modules:
        clone_dir = Path("/content") / PROJECT_DIR_NAME
        if not clone_dir.exists():
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    GITHUB_BRANCH,
                    f"{GITHUB_REPO_URL}.git",
                    str(clone_dir),
                ]
            )
        tutorial_dir = clone_dir / TUTORIAL_RELATIVE_DIR
        if tutorial_dir.exists():
            return tutorial_dir

    raise RuntimeError("Could not find the tutorial files. Start Jupyter from the cloned OHW26 repository.")


def bootstrap_section(section: str, namespace: dict[str, object] | None = None) -> dict[str, object]:
    """Populate globals needed to run one notebook section independently.

    The notebook calls this helper from each skip-ahead cell. Passing
    `globals()` lets the helper avoid overwriting visible functions that a
    participant may have edited earlier in the notebook.
    """

    namespace = namespace if namespace is not None else {}
    section_key = section.lower().strip()

    repo_root = _resolve_repo_root(namespace)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if not namespace.get("_TUTORIAL_DEPENDENCIES_CHECKED", False):
        ensure_dependencies(install=("google.colab" in sys.modules), extra_pip_args=("--quiet",))
        namespace["_TUTORIAL_DEPENDENCIES_CHECKED"] = True

    local_bundle_zip = Path(namespace.get("LOCAL_BUNDLE_ZIP", repo_root / "data" / "fathomnet_underwater_tutorial_bundle.zip"))
    bundle_root = namespace.get("BUNDLE_ROOT")
    if bundle_root is None:
        bundle_root = download_tutorial_bundle(
            bundle_url=str(namespace.get("BUNDLE_URL", BUNDLE_URL)) or None,
            bundle_zip_path=local_bundle_zip if local_bundle_zip.exists() else None,
            output_dir=repo_root / "data" / "fathomnet_underwater_tutorial_bundle",
        )
    bundle_root = Path(bundle_root)

    namespace.update(
        {
            "Path": Path,
            "json": json,
            "REPO_ROOT": repo_root,
            "GITHUB_REPO_URL": GITHUB_REPO_URL,
            "GITHUB_BRANCH": GITHUB_BRANCH,
            "PROJECT_DIR_NAME": PROJECT_DIR_NAME,
            "BUNDLE_URL": str(namespace.get("BUNDLE_URL", BUNDLE_URL)),
            "LOCAL_BUNDLE_ZIP": local_bundle_zip,
            "BUNDLE_ROOT": bundle_root,
        }
    )
    namespace.setdefault("RUN_LIVE_TRAINING", bool(detect_runtime().get("has_cuda", False)))
    namespace.setdefault("build_train_args", default_train_args)

    if section_key == "classification":
        from scripts.tutorial_data import get_task_paths, summarize_classification_dataset
        from scripts.tutorial_models import run_classification_lr_trial
        from scripts.tutorial_viz import plot_confusion_matrix, plot_training_curves, show_image_grid

        classify_paths = get_task_paths("classification", bundle_root)
        namespace.update(
            {
                "get_task_paths": get_task_paths,
                "summarize_classification_dataset": summarize_classification_dataset,
                "run_classification_lr_trial": run_classification_lr_trial,
                "plot_confusion_matrix": plot_confusion_matrix,
                "plot_training_curves": plot_training_curves,
                "show_image_grid": show_image_grid,
                "CLASSIFY_PATHS": classify_paths,
                "CLASSIFY_ROOT": classify_paths["root"],
                "CACHED_CLASSIFY_RESULTS": bundle_root / "cached_training" / "classification" / "results.csv",
            }
        )
        print(json.dumps(summarize_classification_dataset(classify_paths["root"]), indent=2))
        return namespace

    if section_key == "detection":
        from scripts.tutorial_data import (
            get_task_paths,
            make_tiny_detection_dataset,
            select_yolo_examples,
            validate_yolo_dataset,
        )
        from scripts.tutorial_models import prediction_count_at_threshold
        from scripts.tutorial_viz import draw_yolo_boxes, plot_training_curves

        detect_paths = get_task_paths("detect", bundle_root)
        namespace.update(
            {
                "get_task_paths": get_task_paths,
                "make_tiny_detection_dataset": make_tiny_detection_dataset,
                "select_yolo_examples": select_yolo_examples,
                "validate_yolo_dataset": validate_yolo_dataset,
                "prediction_count_at_threshold": prediction_count_at_threshold,
                "draw_yolo_boxes": draw_yolo_boxes,
                "plot_training_curves": plot_training_curves,
                "DETECT_PATHS": detect_paths,
                "DETECT_ROOT": detect_paths["root"],
                "DETECT_YAML": detect_paths["yaml"],
                "CACHED_DETECT_RESULTS": bundle_root / "cached_training" / "detection" / "results.csv",
            }
        )
        print(json.dumps(validate_yolo_dataset(detect_paths["yaml"], task="detect"), indent=2)[:3000])
        return namespace

    if section_key == "segmentation":
        from scripts.tutorial_data import (
            get_task_paths,
            select_yolo_examples,
            validate_yolo_dataset,
            yolo_label_instance_count,
        )
        from scripts.tutorial_viz import draw_yolo_masks, plot_training_curves

        segment_paths = get_task_paths("segment", bundle_root)
        namespace.update(
            {
                "get_task_paths": get_task_paths,
                "select_yolo_examples": select_yolo_examples,
                "validate_yolo_dataset": validate_yolo_dataset,
                "yolo_label_instance_count": yolo_label_instance_count,
                "draw_yolo_masks": draw_yolo_masks,
                "plot_training_curves": plot_training_curves,
                "SEGMENT_PATHS": segment_paths,
                "SEGMENT_ROOT": segment_paths["root"],
                "SEGMENT_YAML": segment_paths["yaml"],
                "CACHED_SEGMENT_RESULTS": bundle_root / "cached_training" / "segmentation" / "results.csv",
            }
        )
        print(json.dumps(validate_yolo_dataset(segment_paths["yaml"], task="segment"), indent=2)[:3000])
        return namespace

    if section_key == "sam3":
        from scripts.tutorial_sam3 import (
            available_cached_prompts,
            configure_huggingface_token,
            install_sam3_package,
            load_cached_sam3_result,
            resolve_sam3_image_path,
            run_sam3_text_prompt,
            sam3_can_run_live,
        )
        from scripts.tutorial_viz import plot_sam3_result

        sam3_status = sam3_can_run_live()
        cached_prompts = available_cached_prompts(bundle_root)
        namespace.update(
            {
                "available_cached_prompts": available_cached_prompts,
                "configure_huggingface_token": configure_huggingface_token,
                "install_sam3_package": install_sam3_package,
                "load_cached_sam3_result": load_cached_sam3_result,
                "resolve_sam3_image_path": resolve_sam3_image_path,
                "run_sam3_text_prompt": run_sam3_text_prompt,
                "sam3_can_run_live": sam3_can_run_live,
                "plot_sam3_result": plot_sam3_result,
                "SAM3_STATUS": sam3_status,
                "USE_LIVE_SAM3": False,
            }
        )
        print(json.dumps(sam3_status, indent=2))
        print(json.dumps(cached_prompts, indent=2)[:3000])
        return namespace

    raise ValueError(f"Unknown tutorial section: {section!r}")
