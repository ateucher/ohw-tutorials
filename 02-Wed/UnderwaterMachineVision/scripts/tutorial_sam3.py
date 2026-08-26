"""SAM3 helpers with cached fallbacks for classroom use."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path


def _version_at_least(version: str | None, minimum: tuple[int, int]) -> bool:
    """Return whether a dotted version string is at least `(major, minor)`."""

    if version is None:
        return False
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return False
    major, minor = (int(match.group(1)), int(match.group(2)))
    return (major, minor) >= minimum


def sam3_can_run_live() -> dict[str, object]:
    """Check whether this runtime is likely able to run live SAM3 inference."""

    python_ok = sys.version_info >= (3, 12)
    sam3_importable = importlib.util.find_spec("sam3") is not None
    torch_importable = importlib.util.find_spec("torch") is not None
    torch_ok = False
    cuda_toolkit_ok = False
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    info: dict[str, object] = {
        "python_ok": python_ok,
        "sam3_importable": sam3_importable,
        "torch_importable": torch_importable,
        "torch_ok": torch_ok,
        "torch_version": None,
        "cuda_available": False,
        "torch_cuda_version": None,
        "cuda_toolkit_ok": cuda_toolkit_ok,
        "hf_token_present": hf_token_present,
        "blockers": [],
        "can_run": False,
    }
    if info["torch_importable"]:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        torch_ok = _version_at_least(info["torch_version"], (2, 7))
        cuda_toolkit_ok = _version_at_least(info["torch_cuda_version"], (12, 6))
        info["torch_ok"] = torch_ok
        info["cuda_toolkit_ok"] = cuda_toolkit_ok
        info["cuda_available"] = bool(torch.cuda.is_available())

    if not python_ok:
        info["blockers"].append("SAM3 requires Python 3.12 or newer.")
    if not sam3_importable:
        info["blockers"].append("The `sam3` package is not installed or not importable.")
    if not torch_importable:
        info["blockers"].append("PyTorch is not installed or not importable.")
    if torch_importable and not torch_ok:
        info["blockers"].append("SAM3 requires PyTorch 2.7 or newer.")
    if not info["cuda_available"]:
        info["blockers"].append("A CUDA GPU is required for the live SAM3 path.")
    if torch_importable and not cuda_toolkit_ok:
        info["blockers"].append("SAM3 requires a PyTorch build with CUDA 12.6 or newer.")
    if not hf_token_present:
        info["blockers"].append("No Hugging Face token is configured in HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")

    info["can_run"] = bool(
        info["python_ok"]
        and info["sam3_importable"]
        and info["torch_ok"]
        and info["cuda_available"]
        and info["cuda_toolkit_ok"]
        and info["hf_token_present"]
    )
    return info


def install_sam3_package(
    package_spec: str = "git+https://github.com/facebookresearch/sam3.git",
    *,
    quiet: bool = True,
) -> dict[str, object]:
    """Install the SAM3 package into the current notebook runtime."""

    command = [sys.executable, "-m", "pip", "install"]
    if quiet:
        command.append("--quiet")
    command.append(package_spec)
    subprocess.check_call(command)
    importlib.invalidate_caches()
    return {
        "package_spec": package_spec,
        "sam3_importable": importlib.util.find_spec("sam3") is not None,
    }


def configure_huggingface_token(
    token: str | None = None,
    *,
    prompt_if_missing: bool = True,
) -> dict[str, object]:
    """Store a Hugging Face token in the current runtime without printing it.

    The token is kept in environment variables used by `huggingface_hub`,
    Transformers, and SAM3 checkpoint-loading utilities. It is not written to
    the notebook, git credentials, or any repository file.
    """

    if token is None and prompt_if_missing:
        token = getpass("Paste a Hugging Face read token. Input is hidden: ")

    token = (token or "").strip()
    if not token:
        return {"configured": False, "reason": "no token provided"}

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    login_result = "huggingface_hub not importable"
    if importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import login

        try:
            login(token=token, add_to_git_credential=False)
            login_result = "logged in for this runtime"
        except Exception as exc:
            login_result = f"login failed: {type(exc).__name__}: {exc}"

    return {
        "configured": True,
        "environment_variables": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
        "login": login_result,
    }


def _load_index(bundle_root: str | Path) -> dict:
    index_path = Path(bundle_root) / "sam3_cached_outputs" / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_cached_prompts(bundle_root: str | Path, image_id: str | None = None) -> dict[str, list[str]]:
    """List cached prompts available in the tutorial bundle."""

    index = _load_index(bundle_root)
    if image_id is not None:
        return {image_id: sorted(index["images"].get(image_id, {}).get("prompts", {}).keys())}
    return {
        key: sorted(value.get("prompts", {}).keys())
        for key, value in index.get("images", {}).items()
    }


def resolve_sam3_image_path(bundle_root: str | Path, image_id_or_path: str | Path) -> Path:
    """Resolve a cached SAM3 image id or a direct local image path.

    Live SAM3 can run on arbitrary local images. Cached fallback outputs are
    only available for the small set of image ids listed in `index.json`, but
    this resolver lets the notebook use one simple variable for both modes.
    """

    candidate = Path(image_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    root = Path(bundle_root)
    image_key = str(image_id_or_path)
    index = _load_index(root)
    image_entry = index.get("images", {}).get(image_key)
    if image_entry is not None:
        return (root / image_entry["image"]).resolve()

    raise ValueError(
        f"Could not resolve {image_id_or_path!r} as a local image path or cached SAM3 image id."
    )


def load_cached_sam3_result(
    bundle_root: str | Path,
    image_id: str,
    prompt: str | None = None,
) -> dict:
    """Load one cached SAM3-like output from the bundle."""

    root = Path(bundle_root)
    index = _load_index(root)
    image_entry = index["images"][image_id]
    prompts = image_entry["prompts"]
    chosen_prompt = prompt or next(iter(prompts))
    if chosen_prompt not in prompts:
        raise KeyError(f"No cached prompt {chosen_prompt!r} for image {image_id!r}")
    result_path = root / "sam3_cached_outputs" / prompts[chosen_prompt]
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["image_path"] = str((root / image_entry["image"]).resolve())
    return result


def run_sam3_text_prompt(
    image_path: str | Path,
    prompt: str,
    *,
    confidence_threshold: float = 0.5,
) -> dict:
    """Run live SAM3 image inference for one text prompt.

    This intentionally follows the official image predictor example. It is not
    called by default in the workshop because checkpoint access and GPU support
    can vary across Colab sessions.
    """

    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import torch

    model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    image = Image.open(image_path).convert("RGB")

    # The current public SAM3 image path can mix BF16 activations with FP32
    # weights unless processor calls run under CUDA BF16 autocast. Keeping the
    # context here means the notebook cells can stay simple.
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    def _tolist(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    return {
        "prompt": prompt,
        "image_path": str(image_path),
        "boxes": _tolist(output.get("boxes", [])),
        "scores": _tolist(output.get("scores", [])),
        "masks": _tolist(output.get("masks", [])),
        "source": "live_sam3",
    }
