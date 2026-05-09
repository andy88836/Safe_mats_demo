"""Train 4 AquaTox RF models from SDF data, save as pkl."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def smiles_to_features(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    fp_arr = np.array(fp, dtype=np.float32)
    desc = [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.HeavyAtomCount(mol),
    ]
    return np.concatenate([fp_arr, np.array(desc, dtype=np.float32)])


def load_sdf_data(sdf_path: str) -> tuple[list[str], list[float]]:
    suppl = Chem.SDMolSupplier(sdf_path)
    smiles_list = []
    tox_list = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            tox = float(mol.GetProp("Tox"))
        except (KeyError, ValueError):
            continue
        smi = Chem.MolToSmiles(mol)
        smiles_list.append(smi)
        tox_list.append(tox)
    return smiles_list, tox_list


def train_endpoint(name: str, sdf_path: str, output_dir: Path, params: dict) -> dict:
    print(f"\n{'='*50}")
    print(f"Training {name}...")

    smiles_list, tox_list = load_sdf_data(sdf_path)
    print(f"  Loaded {len(smiles_list)} molecules from SDF")

    features = []
    valid_tox = []
    valid_smiles = []
    for smi, tox in zip(smiles_list, tox_list):
        feat = smiles_to_features(smi, radius=params["radius"], n_bits=params["n_bits"])
        if feat is not None:
            features.append(feat)
            valid_tox.append(tox)
            valid_smiles.append(smi)

    X = np.array(features)
    y = np.array(valid_tox)
    print(f"  Valid features: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "endpoint": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "train_r2": round(r2_score(y_train, y_pred_train), 4),
        "test_r2": round(r2_score(y_test, y_pred_test), 4),
        "train_mae": round(mean_absolute_error(y_train, y_pred_train), 4),
        "test_mae": round(mean_absolute_error(y_test, y_pred_test), 4),
        "test_rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
    }

    model_path = output_dir / f"rf_{name.lower()}.pkl"
    joblib.dump(model, model_path)
    print(f"  Train R2: {metrics['train_r2']}, Test R2: {metrics['test_r2']}")
    print(f"  Test MAE: {metrics['test_mae']}, Test RMSE: {metrics['test_rmse']}")
    print(f"  Saved: {model_path}")

    return metrics


def main():
    config_path = Path("D:/Doc_Projects/All_des_cal/safe_mats_demo/configs/paths.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tox_cfg = cfg["toxicity_models"]
    output_dir = Path(tox_cfg["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fp_cfg = tox_cfg["fingerprint"]
    model_params = tox_cfg["model_params"]
    params = {
        "radius": fp_cfg["radius"],
        "n_bits": fp_cfg["n_bits"],
        "n_estimators": model_params["n_estimators"],
        "max_depth": model_params["max_depth"],
        "min_samples_leaf": model_params["min_samples_leaf"],
        "random_state": model_params["random_state"],
        "test_size": model_params["test_size"],
    }

    all_metrics = []
    for ep in tox_cfg["endpoints"]:
        sdf_path = ep["training_sdf"]
        if not Path(sdf_path).exists():
            print(f"WARNING: SDF not found: {sdf_path}, skipping {ep['name']}")
            continue
        metrics = train_endpoint(ep["name"], sdf_path, output_dir, params)
        metrics["organism"] = ep["organism"]
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    summary_path = output_dir / "training_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(df[["endpoint", "organism", "n_train", "n_test", "test_r2", "test_mae"]].to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
