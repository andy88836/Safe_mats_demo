"""Benzene/Toluene adsorption prediction using pre-trained models."""
from pathlib import Path

import joblib
import numpy as np
import yaml


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "paths.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_model_path(path_str: str, label: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{label} model file not found: {p}")
    return p


def predict_benzene(eigenvalues: np.ndarray) -> dict:
    """Predict benzene adsorption uptake from SCM eigenvalues.

    Args:
        eigenvalues: np.ndarray of shape (520,) — SCM eigenvalues.

    Returns:
        {"uptake_mg_g": float, "model_version": str, "applicability_warning": str | None}

    Raises:
        ValueError: if input shape is not (520,).
        FileNotFoundError: if model file is missing.
    """
    cfg = _load_config()["adsorption_models"]
    expected_dim = cfg["benzene_target_dim"]

    if eigenvalues.shape != (expected_dim,):
        raise ValueError(f"Expected shape ({expected_dim},), got {eigenvalues.shape}")

    model_path = _validate_model_path(cfg["benzene_model"], "Benzene")
    model = joblib.load(str(model_path))

    X = eigenvalues.reshape(1, -1)
    prediction = float(model.predict(X)[0])

    warning = None
    if prediction < 0:
        warning = f"Predicted negative uptake ({prediction:.2f}), clipped to 0."
        prediction = 0.0

    return {
        "uptake_mg_g": round(prediction, 2),
        "model_version": "RF_tuned_pipeline_seed8",
        "applicability_warning": warning,
    }


def predict_toluene(eigenvalues: np.ndarray) -> dict:
    """Predict toluene adsorption uptake from SCM eigenvalues.

    Args:
        eigenvalues: np.ndarray of shape (584,) — SCM eigenvalues.

    Returns:
        {"uptake_mg_g": float, "model_version": str, "applicability_warning": str | None}

    Raises:
        ValueError: if input shape is not (584,).
        FileNotFoundError: if model file is missing.
    """
    cfg = _load_config()["adsorption_models"]
    expected_dim = cfg["toluene_target_dim"]

    if eigenvalues.shape != (expected_dim,):
        raise ValueError(f"Expected shape ({expected_dim},), got {eigenvalues.shape}")

    model_path = _validate_model_path(cfg["toluene_model"], "Toluene")
    scaler_path = _validate_model_path(cfg["toluene_scaler"], "Toluene scaler")

    scaler = joblib.load(str(scaler_path))
    model = joblib.load(str(model_path))

    X = eigenvalues.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = float(model.predict(X_scaled)[0])

    warning = None
    if prediction < 0:
        warning = f"Predicted negative uptake ({prediction:.2f}), clipped to 0."
        prediction = 0.0

    return {
        "uptake_mg_g": round(prediction, 2),
        "model_version": "XGBoost_seed42",
        "applicability_warning": warning,
    }
