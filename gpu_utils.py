from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb


CUDA_TORCH_HINT = (
    "Install a CUDA-enabled PyTorch wheel before running TFT scripts, for example:\n"
    r"  .\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
)


def tft_trainer_settings(requested_gpus: int = 1) -> tuple[dict, str]:
    if requested_gpus <= 0:
        return {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
        }, "Using CPU for TFT (--gpus 0)."

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU was requested for TFT, but torch.cuda.is_available() is False.\n"
            f"Current torch build: {torch.__version__}\n"
            f"{CUDA_TORCH_HINT}"
        )

    devices = min(requested_gpus, torch.cuda.device_count())
    gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(devices))
    return {
        "accelerator": "gpu",
        "devices": devices,
        "precision": "16-mixed",
    }, f"Using CUDA for TFT on {gpu_names}."


def tft_predict_trainer_kwargs(requested_gpus: int = 1) -> tuple[dict, str]:
    if requested_gpus <= 0:
        return {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "logger": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
        }, "Using CPU for TFT prediction (--gpus 0)."

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU was requested for TFT prediction, but torch.cuda.is_available() is False.\n"
            f"Current torch build: {torch.__version__}\n"
            f"{CUDA_TORCH_HINT}"
        )

    devices = min(requested_gpus, torch.cuda.device_count())
    gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(devices))
    return {
        "accelerator": "gpu",
        "devices": devices,
        "precision": "32-true",
        "logger": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }, f"Using CUDA for TFT prediction on {gpu_names} (full precision)."


def xgb_training_settings(requested_gpus: int = 1) -> tuple[dict, str]:
    if requested_gpus <= 0:
        return {
            "tree_method": "hist",
            "device": "cpu",
        }, "Using CPU for XGBoost (--gpus 0)."

    build_info = xgb.build_info()
    if not build_info.get("USE_CUDA", False):
        raise RuntimeError(
            "The installed xgboost build does not include CUDA support. "
            "Reinstall xgboost with GPU support before training."
        )

    try:
        X_probe = np.random.rand(64, 4).astype("float32")
        y_probe = np.random.rand(64).astype("float32")
        probe = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        probe.fit(X_probe, y_probe)
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        raise RuntimeError(
            "XGBoost CUDA initialization failed. Check that the NVIDIA driver is installed "
            "and the GPU is visible to Python."
        ) from exc

    return {
        "tree_method": "hist",
        "device": "cuda",
    }, "Using CUDA for XGBoost on GPU 0."
