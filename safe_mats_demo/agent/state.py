"""LangGraph state definition for MOF screening pipeline."""
from typing import TypedDict


class ScreeningState(TypedDict, total=False):
    cif_path: str
    mof_id: str
    scm_meta: dict | None
    adsorption: dict | None
    linker: dict | None
    toxicity: dict | None
    safety: dict | None
    final_score: float | None
    recommendation: str | None
    explanation: str | None
    llm_provider: str
    llm_api_key: str | None
    warnings: list[str]
    errors: list[str]
