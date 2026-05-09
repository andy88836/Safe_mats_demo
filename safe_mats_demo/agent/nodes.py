"""LangGraph node functions for MOF screening pipeline."""
import os
from pathlib import Path

from agent.state import ScreeningState


def ingest_cif(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))
    cif_path = state["cif_path"]

    try:
        p = Path(cif_path)
        if not p.exists():
            errors.append(f"CIF file not found: {cif_path}")
            return {"errors": errors, "warnings": warnings}
        mof_id = p.stem
    except Exception as e:
        errors.append(f"ingest_cif failed: {e}")
        mof_id = "unknown"

    return {"mof_id": mof_id, "warnings": warnings, "errors": errors}


def compute_descriptor(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if state.get("errors"):
        return {"warnings": warnings, "errors": errors}

    try:
        from tools.descriptors import compute_scm_eigenvalues
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "configs" / "paths.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        benzene_dim = cfg["adsorption_models"]["benzene_target_dim"]
        toluene_dim = cfg["adsorption_models"]["toluene_target_dim"]

        result_b = compute_scm_eigenvalues(state["cif_path"], benzene_dim)
        result_t = compute_scm_eigenvalues(state["cif_path"], toluene_dim)

        scm_meta = {
            "benzene_eigenvalues": result_b["eigenvalues"],
            "toluene_eigenvalues": result_t["eigenvalues"],
            "raw_dim": result_b["raw_dim"],
            "benzene_padded": result_b["padded"],
            "benzene_truncated": result_b["truncated"],
            "toluene_padded": result_t["padded"],
            "toluene_truncated": result_t["truncated"],
        }

        for r in [result_b, result_t]:
            if r["applicability_warning"]:
                warnings.append(r["applicability_warning"])

        return {"scm_meta": scm_meta, "warnings": warnings, "errors": errors}
    except Exception as e:
        errors.append(f"compute_descriptor failed: {e}")
        return {"scm_meta": None, "warnings": warnings, "errors": errors}


def predict_adsorption(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if state.get("scm_meta") is None:
        errors.append("predict_adsorption skipped: no SCM descriptors available.")
        return {"adsorption": None, "warnings": warnings, "errors": errors}

    try:
        from tools.adsorption import predict_benzene, predict_toluene

        meta = state["scm_meta"]
        b = predict_benzene(meta["benzene_eigenvalues"])
        t = predict_toluene(meta["toluene_eigenvalues"])

        adsorption = {
            "benzene_uptake_mg_g": b["uptake_mg_g"],
            "benzene_model": b["model_version"],
            "toluene_uptake_mg_g": t["uptake_mg_g"],
            "toluene_model": t["model_version"],
        }

        for r in [b, t]:
            if r["applicability_warning"]:
                warnings.append(r["applicability_warning"])

        return {"adsorption": adsorption, "warnings": warnings, "errors": errors}
    except Exception as e:
        errors.append(f"predict_adsorption failed: {e}")
        return {"adsorption": None, "warnings": warnings, "errors": errors}


def extract_linker(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    try:
        from tools.linker_extraction import extract_linker as _extract

        result = _extract(state["cif_path"])
        linker = {
            "metals": result["metals"],
            "linker_smiles": result["linker_smiles"],
            "linker_name": result["linker_name"],
            "linker_formula": result["linker_formula"],
            "extraction_level": result["extraction_level"],
        }
        if result["extraction_level"] == 3:
            warnings.append(result["extraction_note"])

        return {"linker": linker, "warnings": warnings, "errors": errors}
    except Exception as e:
        errors.append(f"extract_linker failed: {e}")
        return {"linker": None, "warnings": warnings, "errors": errors}


def predict_toxicity(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    linker = state.get("linker")
    smiles = linker.get("linker_smiles") if linker else None

    try:
        from tools.toxicity import predict_toxicity as _predict

        result = _predict(smiles)
        return {"toxicity": result, "warnings": warnings, "errors": errors}
    except Exception as e:
        errors.append(f"predict_toxicity failed: {e}")
        return {"toxicity": None, "warnings": warnings, "errors": errors}


def apply_safety_rules(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    try:
        from tools.safety_rules import check_metal_safety, check_pmt_pre_filter

        linker = state.get("linker")
        metals = linker.get("metals", []) if linker else []
        smiles = linker.get("linker_smiles") if linker else None

        metal_result = check_metal_safety(metals) if metals else {
            "tier": "unknown", "flagged_metals": [], "details": "No metals found."
        }
        pmt_result = check_pmt_pre_filter(smiles)

        safety = {
            "metal_tier": metal_result["tier"],
            "metal_flagged": metal_result["flagged_metals"],
            "metal_details": metal_result["details"],
            "pmt_pass": pmt_result["pmt_pass"],
            "pmt_flags": pmt_result["flags"],
            "pmt_descriptors": pmt_result["descriptors"],
        }

        return {"safety": safety, "warnings": warnings, "errors": errors}
    except Exception as e:
        errors.append(f"apply_safety_rules failed: {e}")
        return {"safety": None, "warnings": warnings, "errors": errors}


def score_and_explain(state: ScreeningState) -> dict:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    score, recommendation, explanation = _compute_rule_based_score(state)

    llm_explanation = _try_llm_explanation(state, score, recommendation)
    if llm_explanation:
        explanation = llm_explanation

    return {
        "final_score": score,
        "recommendation": recommendation,
        "explanation": explanation,
        "warnings": warnings,
        "errors": errors,
    }


def _compute_rule_based_score(state: ScreeningState) -> tuple[float, str, str]:
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "configs" / "thresholds.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = cfg["scoring"]["weights"]
    safety_scores = cfg["scoring"]["safety_scores"]
    thresholds = cfg["scoring"]["recommendation"]
    benzene_ref = cfg["scoring"]["adsorption_normalization"]["benzene_ref_mg_g"]

    adsorption = state.get("adsorption") or {}
    safety = state.get("safety") or {}
    toxicity = state.get("toxicity") or {}

    benzene_uptake = adsorption.get("benzene_uptake_mg_g", 0)
    adsorption_score = min(benzene_uptake / benzene_ref, 1.0) * 10

    tier = safety.get("metal_tier", "unknown")
    safety_score = safety_scores.get(tier, safety_scores["unknown"])
    if not safety.get("pmt_pass", True):
        safety_score = max(safety_score - 3, 0)

    tox_values = [
        toxicity.get("LC50_Pimephales"),
        toxicity.get("LC50_Daphnia"),
        toxicity.get("IGC50_Tetrahymena"),
        toxicity.get("IBC50_Vibrio"),
    ]
    valid_tox = [v for v in tox_values if v is not None]
    if valid_tox:
        mean_tox = sum(valid_tox) / len(valid_tox)
        tox_score = max(0, min(10, (5.0 - mean_tox) * 2 + 5))
    else:
        tox_score = 5.0

    final_score = (
        weights["adsorption"] * adsorption_score
        + weights["safety"] * safety_score
        + weights["toxicity"] * tox_score
    )
    final_score = round(min(10.0, max(0.0, final_score)), 2)

    if final_score >= thresholds["recommend_threshold"]:
        recommendation = "recommend"
    elif final_score >= thresholds["borderline_threshold"]:
        recommendation = "borderline"
    else:
        recommendation = "reject"

    mof_id = state.get("mof_id", "unknown")
    linker_data = state.get("linker") or {}
    metals_str = ", ".join(linker_data.get("metals", []))
    tox_summary = (
        f"mean -log toxicity = {sum(valid_tox)/len(valid_tox):.2f}"
        if valid_tox else "toxicity data unavailable"
    )

    explanation = (
        f"MOF {mof_id} achieves {benzene_uptake:.1f} mg/g benzene uptake. "
        f"Metal node(s) [{metals_str}] are in the {tier} tier. "
        f"Predicted aquatic toxicity: {tox_summary}. "
        f"Final score: {final_score:.2f} -> {recommendation}."
    )

    return final_score, recommendation, explanation


def _try_llm_explanation(
    state: ScreeningState, score: float, recommendation: str
) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return None

    try:
        from agent.prompts import build_explanation_prompt

        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "dashscope"
        prompt = build_explanation_prompt(state, score, recommendation)

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=300)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="qwen-turbo",
                temperature=0.3,
                max_tokens=300,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.environ["DASHSCOPE_API_KEY"],
            )

        response = llm.invoke(prompt)
        return response.content
    except Exception:
        return None
