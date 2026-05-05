from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_WEIGHTS = {
    "adsorption_capacity": 0.30,
    "dual_pollutant_performance": 0.15,
    "water_stability": 0.15,
    "toxicity_safety": 0.20,
    "synthesis_feasibility": 0.10,
    "membrane_processability": 0.05,
    "uncertainty": 0.05,
}

TOXIC_METALS = {"Pb", "Hg", "Cd", "As", "Cr", "Tl", "Be"}
BENIGN_METALS = {"Mg", "Ca", "Zn", "Fe", "Al", "Zr"}


@dataclass(frozen=True)
class SafeMOFPaths:
    repo_root: Path
    manuscript_candidates: Path
    report_dir: Path
    benzene_dataset: Path
    benzene_model: Path
    benzene_scaler: Path
    benzene_feature_info: Path
    benzene_summary: Path
    toluene_dataset: Path
    toluene_model: Path
    toluene_scaler: Path
    toluene_feature_info: Path
    toluene_summary: Path
    ood_summary: Path


def build_paths(repo_root: str | Path) -> SafeMOFPaths:
    root = Path(repo_root).resolve()
    return SafeMOFPaths(
        repo_root=root,
        manuscript_candidates=root / "safe_mof_agent" / "data" / "manuscript_candidates.csv",
        report_dir=root / "Agentic" / "safe_mof_agent_reports",
        benzene_dataset=root
        / "sine_matrix"
        / "Benzene"
        / "sine_matrix_benzene_IQR_2_modeling_cleaned.csv",
        benzene_model=root / "sine_matrix" / "Benzene" / "best_model_benzene.pkl",
        benzene_scaler=root / "sine_matrix" / "Benzene" / "scaler_benzene.pkl",
        benzene_feature_info=root / "sine_matrix" / "Benzene" / "feature_info_benzene.json",
        benzene_summary=root / "sine_matrix" / "Benzene" / "paper_best_model_summary_benzene.csv",
        toluene_dataset=root
        / "sine_matrix"
        / "Toluene"
        / "Toluene"
        / "sine_matrix_toluene_modeling_cleaned_toluene.csv",
        toluene_model=root / "sine_matrix" / "Toluene" / "Toluene" / "best_model_toluene.pkl",
        toluene_scaler=root / "sine_matrix" / "Toluene" / "Toluene" / "scaler_toluene.pkl",
        toluene_feature_info=root
        / "sine_matrix"
        / "Toluene"
        / "Toluene"
        / "feature_info_toluene.json",
        toluene_summary=root
        / "sine_matrix"
        / "Toluene"
        / "Toluene"
        / "paper_best_model_summary_toluene.csv",
        ood_summary=root / "sine_matrix" / "OOD" / "metal_group_ood_summary.csv",
    )
