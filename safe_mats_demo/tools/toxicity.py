"""Aquatic toxicity prediction using trained RF + Morgan fingerprint models."""
from pathlib import Path

import joblib
import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "paths.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _smiles_to_features(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    fp_arr = np.array(fp, dtype=np.float32)
    desc = [
        Descriptors.MolLogP(mol), Descriptors.MolWt(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.HeavyAtomCount(mol),
    ]
    return np.concatenate([fp_arr, np.array(desc, dtype=np.float32)])


_ENDPOINT_MAP = {
    "LC50": "rf_lc50.pkl",
    "LC50DM": "rf_lc50dm.pkl",
    "IGC50": "rf_igc50.pkl",
    "IBC50": "rf_ibc50.pkl",
}

_UNIT = "-log10(mol/L)"

_CONFIDENCE = {
    "LC50": "high",
    "LC50DM": "medium",
    "IGC50": "high",
    "IBC50": "low",
}


def predict_toxicity(linker_smiles: str | None) -> dict:
    """Predict aquatic toxicity for 4 endpoints from a linker SMILES.

    Args:
        linker_smiles: SMILES string or None (Level 3 linker extraction).

    Returns:
        dict with keys: LC50_Pimephales, LC50_Daphnia, IGC50_Tetrahymena,
        IBC50_Vibrio (float|None), plus _unit and _confidence for each,
        and applicability_domain_flag (bool).
        Values are -log10(mol/L); higher = more toxic.
    """
    result = {
        "LC50_Pimephales": None, "LC50_Pimephales_unit": _UNIT, "LC50_Pimephales_confidence": None,
        "LC50_Daphnia": None, "LC50_Daphnia_unit": _UNIT, "LC50_Daphnia_confidence": None,
        "IGC50_Tetrahymena": None, "IGC50_Tetrahymena_unit": _UNIT, "IGC50_Tetrahymena_confidence": None,
        "IBC50_Vibrio": None, "IBC50_Vibrio_unit": _UNIT, "IBC50_Vibrio_confidence": None,
        "applicability_domain_flag": False,
    }

    if linker_smiles is None:
        return result

    cfg = _load_config()
    model_dir = Path(cfg["toxicity_models"]["model_dir"])
    fp_cfg = cfg["toxicity_models"]["fingerprint"]

    try:
        features = _smiles_to_features(
            linker_smiles, radius=fp_cfg["radius"], n_bits=fp_cfg["n_bits"]
        )
    except ValueError:
        return result

    X = features.reshape(1, -1)
    key_map = {
        "LC50": "LC50_Pimephales",
        "LC50DM": "LC50_Daphnia",
        "IGC50": "IGC50_Tetrahymena",
        "IBC50": "IBC50_Vibrio",
    }

    all_ok = True
    for endpoint, filename in _ENDPOINT_MAP.items():
        model_path = model_dir / filename
        if not model_path.exists():
            all_ok = False
            continue
        model = joblib.load(str(model_path))
        pred = float(model.predict(X)[0])
        out_key = key_map[endpoint]
        result[out_key] = round(pred, 4)
        result[f"{out_key}_confidence"] = _CONFIDENCE[endpoint]

    result["applicability_domain_flag"] = all_ok
    return result
