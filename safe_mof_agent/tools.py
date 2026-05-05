from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BENIGN_METALS, DEFAULT_WEIGHTS, TOXIC_METALS, SafeMOFPaths
from .schemas import CandidateRecord, PredictionRecord, ToxicityRecord, utc_timestamp


TARGETS = ("benzene", "toluene")
ADSORPTION_BENCHMARKS_MG_G = {"benzene": 1000.0, "toluene": 600.0}


def read_model_summary(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {k: _coerce_number(v) for k, v in row.items()}


def load_feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["feature_names"])


def load_manuscript_records(path: Path) -> tuple[list[CandidateRecord], list[PredictionRecord], list[ToxicityRecord]]:
    df = pd.read_csv(path)
    candidates: list[CandidateRecord] = []
    predictions: list[PredictionRecord] = []
    toxicities: list[ToxicityRecord] = []

    for _, row in df.iterrows():
        candidate_id = str(row["candidate_id"])
        metal_centers = _split_cell(row.get("metal_centers", ""))
        linker_names = _split_cell(row.get("linker_names", ""))
        evidence = {
            "benzene_aqueous_mg_g": _maybe_float(row.get("benzene_aqueous_mg_g")),
            "toluene_aqueous_mg_g": _maybe_float(row.get("toluene_aqueous_mg_g")),
            "benzene_vvs_mg_g": _maybe_float(row.get("benzene_vvs_mg_g")),
            "toluene_vvs_mg_g": _maybe_float(row.get("toluene_vvs_mg_g")),
            "water_stability_score": _maybe_float(row.get("water_stability_score")),
            "synthesis_feasibility_score": _maybe_float(row.get("synthesis_feasibility_score")),
            "membrane_processability_score": _maybe_float(row.get("membrane_processability_score")),
            "notes": str(row.get("notes", "")),
        }
        candidates.append(
            CandidateRecord(
                candidate_id=candidate_id,
                source=str(row.get("source", "manuscript")),
                metal_centers=metal_centers,
                linker_names=linker_names,
                evidence=evidence,
                provenance={
                    "created_by": "manuscript_candidate_loader",
                    "input_file": str(path),
                    "timestamp": utc_timestamp(),
                },
            )
        )
        predictions.append(
            PredictionRecord(
                candidate_id=candidate_id,
                predictions={
                    "benzene": evidence["benzene_aqueous_mg_g"],
                    "toluene": evidence["toluene_aqueous_mg_g"],
                },
                uncertainty={"benzene": None, "toluene": None},
                observed={
                    "benzene_vvs_mg_g": evidence["benzene_vvs_mg_g"],
                    "toluene_vvs_mg_g": evidence["toluene_vvs_mg_g"],
                },
                model_info={
                    "source": "manuscript experimental evidence",
                    "interpretation": "Aqueous adsorption values are used as validation evidence in this demo mode.",
                },
            )
        )
        toxicities.append(
            ToxicityRecord(
                candidate_id=candidate_id,
                metal_toxicity_flag=_bool_cell(row.get("metal_toxicity_flag")),
                ligand_toxicity_risk=str(row.get("ligand_toxicity_risk", "unknown")).lower(),
                pmt_vpvm_risk=str(row.get("pmt_vpvm_risk", "unknown")).lower(),
                leaching_risk=str(row.get("leaching_risk", "unknown")).lower(),
                risk_level=str(row.get("risk_level", "unknown")).lower(),
                notes=_split_cell(row.get("toxicity_notes", "")),
            )
        )

    return candidates, predictions, toxicities


class SineMatrixToolbox:
    def __init__(self, paths: SafeMOFPaths):
        self.paths = paths
        self._datasets: dict[str, pd.DataFrame] = {}
        self._features: dict[str, list[str]] = {}
        self._models: dict[str, Any] = {}
        self._scalers: dict[str, Any] = {}
        self._summaries: dict[str, dict[str, Any]] = {}

    def common_candidate_ids(self) -> list[str]:
        benzene_ids = set(self._dataset("benzene")["MOF"].astype(str))
        toluene_ids = set(self._dataset("toluene")["MOF"].astype(str))
        return sorted(benzene_ids & toluene_ids)

    def predict_many(self, candidate_ids: list[str]) -> tuple[list[CandidateRecord], list[PredictionRecord], list[ToxicityRecord]]:
        by_id: dict[str, dict[str, Any]] = {candidate_id: {} for candidate_id in candidate_ids}
        records: dict[str, CandidateRecord] = {}

        for target in TARGETS:
            df = self._dataset(target)
            selected = df[df["MOF"].astype(str).isin(candidate_ids)].copy()
            selected["MOF"] = selected["MOF"].astype(str)
            selected = selected.drop_duplicates(subset=["MOF"])
            feature_names = self._feature_names(target)
            x = selected[feature_names]
            scaler = self._scaler(target)
            model = self._model(target)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_scaled = scaler.transform(x)
                y_pred = model.predict(x_scaled)

            target_col = f"{target}_adsorption"
            summary = self._summary(target)
            rmse = _maybe_float(summary.get("Test_RMSE")) or _maybe_float(summary.get("CV_RMSE_mean"))
            for idx, candidate_id in enumerate(selected["MOF"].tolist()):
                observed = _maybe_float(selected.iloc[idx][target_col])
                by_id[candidate_id][target] = float(y_pred[idx])
                by_id[candidate_id][f"{target}_observed_gcmc"] = observed
                by_id[candidate_id][f"{target}_uncertainty_rmse"] = rmse
                by_id[candidate_id][f"{target}_ood_mean_abs_z"] = self._mean_abs_z(target, x_scaled[idx])

                records.setdefault(
                    candidate_id,
                    CandidateRecord(
                        candidate_id=candidate_id,
                        source="CoRE_DDEC_GCMC_sine_matrix",
                        evidence={},
                        provenance={
                            "created_by": "sine_matrix_toolbox",
                            "timestamp": utc_timestamp(),
                        },
                    ),
                )
                records[candidate_id].evidence[f"{target}_observed_gcmc_mg_g"] = observed

        predictions: list[PredictionRecord] = []
        toxicities: list[ToxicityRecord] = []
        candidates: list[CandidateRecord] = []
        for candidate_id in candidate_ids:
            payload = by_id[candidate_id]
            if not all(target in payload for target in TARGETS):
                continue
            ood_score = float(np.mean([payload.get(f"{t}_ood_mean_abs_z", 0.0) for t in TARGETS]))
            records[candidate_id].evidence["ood_mean_abs_z"] = ood_score
            records[candidate_id].evidence["ood_risk"] = classify_ood(ood_score)
            records[candidate_id].evidence["water_stability_score"] = 0.55
            records[candidate_id].evidence["synthesis_feasibility_score"] = 0.50
            records[candidate_id].evidence["membrane_processability_score"] = 0.50
            candidates.append(records[candidate_id])
            predictions.append(
                PredictionRecord(
                    candidate_id=candidate_id,
                    predictions={target: _maybe_float(payload.get(target)) for target in TARGETS},
                    uncertainty={target: _maybe_float(payload.get(f"{target}_uncertainty_rmse")) for target in TARGETS},
                    observed={
                        target: _maybe_float(payload.get(f"{target}_observed_gcmc")) for target in TARGETS
                    },
                    model_info={
                        "descriptor": "sine_coulomb_matrix",
                        "benzene_model": str(self.paths.benzene_model),
                        "toluene_model": str(self.paths.toluene_model),
                        "benzene_metrics": self._summary("benzene"),
                        "toluene_metrics": self._summary("toluene"),
                    },
                )
            )
            toxicities.append(
                ToxicityRecord(
                    candidate_id=candidate_id,
                    metal_toxicity_flag=False,
                    ligand_toxicity_risk="unknown",
                    pmt_vpvm_risk="unknown",
                    leaching_risk="unknown",
                    risk_level="unknown",
                    notes=["Ligand and metal annotations are not available in the sine-matrix table."],
                )
            )

        return candidates, predictions, toxicities

    def _dataset(self, target: str) -> pd.DataFrame:
        if target not in self._datasets:
            path = self.paths.benzene_dataset if target == "benzene" else self.paths.toluene_dataset
            self._datasets[target] = pd.read_csv(path)
        return self._datasets[target]

    def _feature_names(self, target: str) -> list[str]:
        if target not in self._features:
            path = self.paths.benzene_feature_info if target == "benzene" else self.paths.toluene_feature_info
            self._features[target] = load_feature_names(path)
        return self._features[target]

    def _model(self, target: str) -> Any:
        if target not in self._models:
            import joblib

            path = self.paths.benzene_model if target == "benzene" else self.paths.toluene_model
            model = joblib.load(path)
            _force_single_thread(model)
            self._models[target] = model
        return self._models[target]

    def _scaler(self, target: str) -> Any:
        if target not in self._scalers:
            import joblib

            path = self.paths.benzene_scaler if target == "benzene" else self.paths.toluene_scaler
            self._scalers[target] = joblib.load(path)
        return self._scalers[target]

    def _summary(self, target: str) -> dict[str, Any]:
        if target not in self._summaries:
            path = self.paths.benzene_summary if target == "benzene" else self.paths.toluene_summary
            self._summaries[target] = read_model_summary(path)
        return self._summaries[target]

    def _mean_abs_z(self, target: str, scaled_row: np.ndarray) -> float:
        _ = target
        return float(np.mean(np.abs(np.asarray(scaled_row, dtype=float))))


def classify_ood(mean_abs_z: float) -> str:
    if mean_abs_z < 0.75:
        return "low"
    if mean_abs_z < 1.25:
        return "medium"
    return "high"


def toxicity_safety_score(toxicity: ToxicityRecord) -> float:
    penalty = 0.0
    if toxicity.metal_toxicity_flag:
        penalty += 0.35
    penalty += {"low": 0.0, "medium": 0.18, "high": 0.35, "unknown": 0.18}.get(toxicity.ligand_toxicity_risk, 0.18)
    penalty += {"low": 0.0, "medium": 0.12, "high": 0.25, "unknown": 0.12}.get(toxicity.pmt_vpvm_risk, 0.12)
    penalty += {"low": 0.0, "medium": 0.12, "high": 0.30, "unknown": 0.12}.get(toxicity.leaching_risk, 0.12)
    if toxicity.risk_level == "high":
        penalty += 0.20
    return max(0.0, 1.0 - penalty)


def compute_decision_components(
    candidate: CandidateRecord,
    prediction: PredictionRecord,
    toxicity: ToxicityRecord,
    normalized_predictions: dict[str, dict[str, float]],
) -> dict[str, float]:
    benzene = normalized_predictions[candidate.candidate_id].get("benzene", 0.0)
    toluene = normalized_predictions[candidate.candidate_id].get("toluene", 0.0)
    uncertainty = _uncertainty_score(prediction)
    evidence = candidate.evidence
    return {
        "adsorption_capacity": float(np.mean([benzene, toluene])),
        "dual_pollutant_performance": float(min(benzene, toluene)),
        "water_stability": _score01(evidence.get("water_stability_score"), 0.55),
        "toxicity_safety": toxicity_safety_score(toxicity),
        "synthesis_feasibility": _score01(evidence.get("synthesis_feasibility_score"), 0.50),
        "membrane_processability": _score01(evidence.get("membrane_processability_score"), 0.50),
        "uncertainty": uncertainty,
    }


def weighted_score(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    return float(sum(components[key] * weights[key] for key in weights))


def normalize_predictions(predictions: list[PredictionRecord]) -> dict[str, dict[str, float]]:
    raw: dict[str, dict[str, float]] = {}
    for pred in predictions:
        raw[pred.candidate_id] = {}
        for target in TARGETS:
            value = _maybe_float(pred.predictions.get(target))
            raw[pred.candidate_id][target] = value if value is not None else 0.0

    result: dict[str, dict[str, float]] = {}
    for candidate_id, values in raw.items():
        result[candidate_id] = {}
        for target in TARGETS:
            benchmark = ADSORPTION_BENCHMARKS_MG_G[target]
            result[candidate_id][target] = max(0.0, min(1.0, values[target] / benchmark))
    return result


def experimental_next_steps(candidate: CandidateRecord, toxicity: ToxicityRecord, ood_risk: str) -> list[str]:
    steps = [
        "VVS adsorption isotherms for benzene and toluene",
        "Aqueous batch adsorption and kinetic tests",
        "Competitive adsorption in mixed benzene-series pollutants",
    ]
    if toxicity.risk_level in {"high", "unknown"} or toxicity.leaching_risk in {"high", "unknown"}:
        steps.append("ICP-MS metal leaching and linker-release analysis")
    if toxicity.risk_level in {"high", "unknown"}:
        steps.append("Zebrafish embryo toxicity validation")
    if ood_risk in {"medium", "high"}:
        steps.append("Confirmatory GCMC or VVS before scale-up")
    if _score01(candidate.evidence.get("membrane_processability_score"), 0.5) >= 0.6:
        steps.append("Cellulose membrane fabrication and cycling test")
    return steps


def infer_metal_safety(metals: list[str]) -> tuple[bool, str]:
    if any(metal in TOXIC_METALS for metal in metals):
        return True, "high"
    if metals and all(metal in BENIGN_METALS for metal in metals):
        return False, "low"
    return False, "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_decision_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _uncertainty_score(prediction: PredictionRecord) -> float:
    values = [v for v in prediction.uncertainty.values() if v is not None]
    preds = [abs(v) for v in prediction.predictions.values() if v is not None]
    if not values:
        return 0.80
    denom = max(np.mean(preds), 1.0) if preds else 1.0
    relative = float(np.mean(values) / denom)
    return max(0.0, min(1.0, 1.0 - relative))


def _score01(value: Any, default: float) -> float:
    parsed = _maybe_float(value)
    if parsed is None:
        parsed = default
    return max(0.0, min(1.0, float(parsed)))


def _split_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).replace(";", "|").split("|") if part.strip()]


def _bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _coerce_number(value: Any) -> Any:
    parsed = _maybe_float(value)
    return parsed if parsed is not None else value


def _force_single_thread(model: Any) -> None:
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    if hasattr(model, "set_params"):
        try:
            params = model.get_params()
            if "n_jobs" in params:
                model.set_params(n_jobs=1)
        except Exception:
            pass
