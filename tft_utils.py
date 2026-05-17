from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


# Named checkpoint families living under models/:
#   original  : tft_best.ckpt, tft_best-v1.ckpt ... (trained to Dec 2020)
#   step1     : tft_best_2022*.ckpt             (retrained to Dec 2021)
#   step5     : tft_best_xgbfused*.ckpt         (XGB-as-feature variant)
CHECKPOINT_PATTERNS = {
    "original": ["tft_best.ckpt", "tft_best-v*.ckpt"],
    "step1":    ["tft_best_2022.ckpt", "tft_best_2022-v*.ckpt"],
    "step5":    ["tft_best_xgbfused.ckpt", "tft_best_xgbfused-v*.ckpt"],
}


def checkpoint_score(checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    callbacks = checkpoint.get("callbacks", {})
    for callback_state in callbacks.values():
        if isinstance(callback_state, dict) and "best_model_score" in callback_state:
            score = callback_state["best_model_score"]
            return float(score.item() if hasattr(score, "item") else score)
    return float("inf")


def list_checkpoints(models_dir: Path, family: str = "original") -> list[Path]:
    patterns = CHECKPOINT_PATTERNS.get(family, [])
    found: list[Path] = []
    for pat in patterns:
        found.extend(models_dir.glob(pat))
    return sorted(set(found))


def find_best_checkpoint(models_dir: Path, family: str = "original") -> Path | None:
    candidates = list_checkpoints(models_dir, family)
    if not candidates:
        return None
    return min(candidates, key=checkpoint_score)


def load_tft_from_checkpoint(training, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    loss = hparams.get("loss")
    quantiles = getattr(loss, "quantiles", [0.1, 0.5, 0.9])

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hparams["hidden_size"],
        attention_head_size=hparams["attention_head_size"],
        lstm_layers=hparams["lstm_layers"],
        hidden_continuous_size=hparams["hidden_continuous_size"],
        dropout=hparams["dropout"],
        loss=QuantileLoss(quantiles=quantiles),
        learning_rate=hparams["learning_rate"],
        optimizer=hparams["optimizer"],
        reduce_on_plateau_patience=hparams["reduce_on_plateau_patience"],
        log_interval=hparams["log_interval"],
        mask_bias=hparams.get("mask_bias", -1e9),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, quantiles


def normalize_prediction_output(predictions):
    if torch.is_tensor(predictions):
        return predictions.detach().cpu()
    if isinstance(predictions, (list, tuple)):
        if not predictions:
            return torch.empty(0)
        batches = [normalize_prediction_output(batch) for batch in predictions]
        if len(batches) == 1:
            return batches[0]
        return torch.cat(batches, dim=0)
    raise TypeError(f"Unsupported prediction output type: {type(predictions)!r}")


def extract_raw_prediction_output(predictions):
    if predictions is None:
        raise ValueError("No raw predictions were returned")
    if hasattr(predictions, "output"):
        return predictions.output
    if hasattr(predictions, "prediction"):
        return predictions
    if isinstance(predictions, (list, tuple)):
        if not predictions:
            raise ValueError("No raw predictions were returned")
        last_error = None
        for item in predictions:
            try:
                return extract_raw_prediction_output(item)
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise ValueError("No raw predictions were returned") from last_error
    raise TypeError(f"Unsupported raw prediction type: {type(predictions)!r}")
