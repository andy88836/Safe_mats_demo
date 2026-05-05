from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class TaskCard:
    research_goal: str
    pollutants: list[str]
    material_class: str
    objectives: list[str]
    constraints: dict[str, Any]
    workflow: list[str]
    created_at: str = field(default_factory=utc_timestamp)


@dataclass
class CandidateRecord:
    candidate_id: str
    source: str
    metal_centers: list[str] = field(default_factory=list)
    linker_names: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    candidate_id: str
    predictions: dict[str, float | None]
    uncertainty: dict[str, float | None]
    model_info: dict[str, Any]
    observed: dict[str, float | None] = field(default_factory=dict)


@dataclass
class ToxicityRecord:
    candidate_id: str
    metal_toxicity_flag: bool
    ligand_toxicity_risk: str
    pmt_vpvm_risk: str
    leaching_risk: str
    risk_level: str
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    candidate_id: str
    status: str
    scientific_validity: dict[str, Any]
    environmental_safety: dict[str, Any]
    reviewer_risks: list[str]
    recommended_actions: list[str]


@dataclass
class DecisionRecord:
    candidate_id: str
    decision_class: str
    final_score: float
    component_scores: dict[str, float]
    reasons: list[str]
    next_steps: list[str]
    llm_summary: str | None = None
    manuscript_claim: str | None = None


@dataclass
class TraceEvent:
    agent: str
    action: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)


def to_plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if hasattr(value, "__fspath__"):
        return value.__fspath__()
    return value


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return to_plain(asdict(value))
