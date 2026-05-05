from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_WEIGHTS, build_paths
from .llm import LLMConfigurationError, LLMRequestError, OpenAIResponsesClient
from .llm_schemas import AUDIT_REPORT_SCHEMA, DECISION_NARRATIVE_SCHEMA, EXECUTION_PLAN_SCHEMA, TASK_CARD_SCHEMA
from .prompts import AUDIT_SYSTEM, DECISION_SYSTEM, EXECUTION_SYSTEM, PLANNING_SYSTEM
from .schemas import (
    AuditReport,
    CandidateRecord,
    DecisionRecord,
    PredictionRecord,
    TaskCard,
    ToxicityRecord,
    TraceEvent,
    dataclass_to_dict,
    utc_timestamp,
)
from .tools import (
    SineMatrixToolbox,
    compute_decision_components,
    experimental_next_steps,
    load_manuscript_records,
    normalize_predictions,
    weighted_score,
    write_decision_csv,
    write_json,
)


class PlanningAgent:
    name = "Planning Agent"

    def __init__(self, llm_client: OpenAIResponsesClient | None = None):
        self.llm_client = llm_client
        self.last_meta: dict[str, Any] = {"llm_used": False}

    def plan(self, research_goal: str) -> TaskCard:
        if self.llm_client and self.llm_client.enabled:
            payload = {
                "research_goal": research_goal,
                "domain_context": {
                    "project": "Safe-MOF-Agent",
                    "material_class": "MOF",
                    "available_tools": [
                        "load_manuscript_records",
                        "run_dataset_screen",
                        "adsorption_ml_prediction",
                        "safe_by_design_audit",
                        "multi_objective_ranking",
                    ],
                    "available_evidence": [
                        "GCMC-derived sine Coulomb matrix descriptor tables",
                        "trained benzene Random Forest model",
                        "trained toluene XGBoost model",
                        "Mg-MOF and Ag-MOF validation records",
                    ],
                },
            }
            try:
                parsed, meta = self.llm_client.structured_json(
                    agent_name=self.name,
                    system_prompt=PLANNING_SYSTEM,
                    user_payload=payload,
                    schema_name="safe_mof_task_card",
                    schema=TASK_CARD_SCHEMA,
                )
                self.last_meta = {"llm_used": True, **meta}
                return TaskCard(**parsed)
            except (LLMConfigurationError, LLMRequestError, TypeError, KeyError) as exc:
                self.last_meta = {"llm_used": False, "fallback_reason": str(exc)}

        task = self._fallback_plan(research_goal)
        self.last_meta = {**self.last_meta, "llm_used": False, "fallback": "deterministic_template"}
        return task

    def _fallback_plan(self, research_goal: str) -> TaskCard:
        return TaskCard(
            research_goal=research_goal,
            pollutants=["benzene", "toluene", "aniline"],
            material_class="MOF",
            objectives=[
                "maximize_adsorption_capacity",
                "maximize_dual_pollutant_performance",
                "minimize_aquatic_toxicity",
                "maximize_water_stability",
                "maximize_synthesis_feasibility",
                "maximize_membrane_processability",
            ],
            constraints={
                "exclude_highly_toxic_metals": ["Pb", "Hg", "Cd", "As", "Cr", "Tl", "Be"],
                "require_traceable_descriptor_source": True,
                "require_human_experimental_validation": True,
                "preferred_validation": [
                    "VVS",
                    "aqueous_adsorption",
                    "real_water",
                    "zebrafish",
                    "membrane_cycling",
                ],
            },
            workflow=[
                "load_candidate_records",
                "predict_adsorption_or_load_validation_evidence",
                "estimate_toxicity_and_leaching_risk",
                "audit_scientific_and_safety_risks",
                "multi_objective_ranking",
                "recommend_experiments",
            ],
        )


class ExecutionAgent:
    name = "Execution Agent"

    def __init__(self, paths, llm_client: OpenAIResponsesClient | None = None):
        self.paths = paths
        self.toolbox = SineMatrixToolbox(paths)
        self.llm_client = llm_client
        self.last_meta: dict[str, Any] = {"llm_used": False}

    def create_execution_plan(self, mode: str, dataset_limit: int | None, task: TaskCard) -> dict[str, Any]:
        fallback = {
            "selected_tool": "load_manuscript_records" if mode == "manuscript" else "run_dataset_screen",
            "tool_arguments": {"limit": dataset_limit},
            "rationale": f"CLI mode requested {mode}.",
            "expected_outputs": ["candidate_records", "prediction_records", "toxicity_records"],
        }
        if self.llm_client and self.llm_client.enabled:
            payload = {
                "requested_mode": mode,
                "dataset_limit": dataset_limit,
                "task_card": dataclass_to_dict(task),
                "available_tools": {
                    "load_manuscript_records": {
                        "description": "Load curated Mg-MOF and Ag-MOF validation evidence from manuscript records.",
                        "arguments": {"limit": None},
                    },
                    "run_dataset_screen": {
                        "description": "Run local sine-matrix adsorption ML models over shared benzene/toluene candidates.",
                        "arguments": {"limit": "integer or null"},
                    },
                },
                "instruction": "Respect requested_mode. Select the matching tool and do not invent tool outputs.",
            }
            try:
                parsed, meta = self.llm_client.structured_json(
                    agent_name=self.name,
                    system_prompt=EXECUTION_SYSTEM,
                    user_payload=payload,
                    schema_name="safe_mof_execution_plan",
                    schema=EXECUTION_PLAN_SCHEMA,
                )
                if mode == "manuscript" and parsed["selected_tool"] != "load_manuscript_records":
                    parsed = fallback
                    meta = {**meta, "guardrail": "overrode_tool_to_match_cli_mode"}
                if mode == "dataset" and parsed["selected_tool"] != "run_dataset_screen":
                    parsed = fallback
                    meta = {**meta, "guardrail": "overrode_tool_to_match_cli_mode"}
                self.last_meta = {"llm_used": True, **meta}
                return parsed
            except (LLMConfigurationError, LLMRequestError, TypeError, KeyError) as exc:
                self.last_meta = {"llm_used": False, "fallback_reason": str(exc)}

        self.last_meta = {**self.last_meta, "llm_used": False, "fallback": "deterministic_tool_selection"}
        return fallback

    def execute_plan(self, plan: dict[str, Any]) -> tuple[list[CandidateRecord], list[PredictionRecord], list[ToxicityRecord]]:
        selected_tool = plan["selected_tool"]
        limit = plan.get("tool_arguments", {}).get("limit")
        if selected_tool == "load_manuscript_records":
            return self.run_manuscript_demo()
        if selected_tool == "run_dataset_screen":
            return self.run_dataset_screen(limit)
        raise ValueError(f"Unknown execution tool: {selected_tool}")

    def run_manuscript_demo(self) -> tuple[list[CandidateRecord], list[PredictionRecord], list[ToxicityRecord]]:
        return load_manuscript_records(self.paths.manuscript_candidates)

    def run_dataset_screen(self, limit: int | None = None) -> tuple[list[CandidateRecord], list[PredictionRecord], list[ToxicityRecord]]:
        candidate_ids = self.toolbox.common_candidate_ids()
        if limit is not None:
            candidate_ids = candidate_ids[:limit]
        return self.toolbox.predict_many(candidate_ids)


class AuditAgent:
    name = "Audit Agent"

    def __init__(self, llm_client: OpenAIResponsesClient | None = None):
        self.llm_client = llm_client
        self.llm_calls = 0
        self.llm_failures: list[str] = []

    def audit(
        self,
        candidate: CandidateRecord,
        prediction: PredictionRecord,
        toxicity: ToxicityRecord,
    ) -> AuditReport:
        rule_report = self._rule_audit(candidate, prediction, toxicity)
        if not (self.llm_client and self.llm_client.enabled):
            return rule_report

        payload = {
            "candidate": dataclass_to_dict(candidate),
            "prediction": dataclass_to_dict(prediction),
            "toxicity": dataclass_to_dict(toxicity),
            "rule_audit": dataclass_to_dict(rule_report),
            "instruction": "Return an audit report. Preserve any critical safety flags from rule_audit.",
        }
        try:
            parsed, _meta = self.llm_client.structured_json(
                agent_name=self.name,
                system_prompt=AUDIT_SYSTEM,
                user_payload=payload,
                schema_name="safe_mof_audit_report",
                schema=AUDIT_REPORT_SCHEMA,
            )
            self.llm_calls += 1
            llm_report = AuditReport(**parsed)
            return self._merge_audits(rule_report, llm_report)
        except (LLMConfigurationError, LLMRequestError, TypeError, KeyError) as exc:
            self.llm_failures.append(f"{candidate.candidate_id}: {exc}")
            return rule_report

    def _rule_audit(
        self,
        candidate: CandidateRecord,
        prediction: PredictionRecord,
        toxicity: ToxicityRecord,
    ) -> AuditReport:
        evidence = candidate.evidence
        ood_risk = str(evidence.get("ood_risk", "low"))
        reviewer_risks: list[str] = []
        recommended_actions: list[str] = []

        if toxicity.risk_level == "high":
            reviewer_risks.append("high_environmental_or_leaching_risk")
            recommended_actions.append("Do not advance without toxicity and leaching repair data.")
        if toxicity.risk_level == "unknown":
            reviewer_risks.append("missing_ligand_and_metal_toxicity_annotation")
            recommended_actions.append("Add linker SMILES, metal annotation, QSAR toxicity, PMT/vPvM and leaching checks.")
        if ood_risk in {"medium", "high"}:
            reviewer_risks.append(f"{ood_risk}_descriptor_domain_risk")
            recommended_actions.append("Add confirmatory GCMC/VVS data for this candidate.")
        if "benzene_vvs_mg_g" not in evidence and candidate.source != "manuscript":
            reviewer_risks.append("missing_vapor_to_water_validation_bridge")
            recommended_actions.append("Use VVS isotherms to bridge gas-phase screening and aqueous adsorption.")
        if evidence.get("water_stability_score", 0.0) < 0.65:
            reviewer_risks.append("water_stability_evidence_needs_strengthening")
            recommended_actions.append("Add pH-dependent stability, cycling PXRD/FTIR/BET and real-water tests.")

        status = "pass"
        if toxicity.risk_level == "high":
            status = "reject_or_repair"
        elif reviewer_risks:
            status = "pass_with_actions"

        return AuditReport(
            candidate_id=candidate.candidate_id,
            status=status,
            scientific_validity={
                "prediction_available": all(v is not None for v in prediction.predictions.values()),
                "descriptor_or_evidence_source": candidate.source,
                "ood_risk": ood_risk,
                "model_info": prediction.model_info,
            },
            environmental_safety={
                "metal_toxicity_flag": toxicity.metal_toxicity_flag,
                "ligand_toxicity_risk": toxicity.ligand_toxicity_risk,
                "pmt_vpvm_risk": toxicity.pmt_vpvm_risk,
                "leaching_risk": toxicity.leaching_risk,
                "overall_risk": toxicity.risk_level,
            },
            reviewer_risks=reviewer_risks,
            recommended_actions=_dedupe(recommended_actions),
        )

    def _merge_audits(self, rule_report: AuditReport, llm_report: AuditReport) -> AuditReport:
        status_rank = {"pass": 0, "pass_with_actions": 1, "reject_or_repair": 2}
        status = max([rule_report.status, llm_report.status], key=lambda item: status_rank[item])
        scientific_validity = {**llm_report.scientific_validity}
        scientific_validity.setdefault("rule_model_info", rule_report.scientific_validity.get("model_info", {}))
        environmental_safety = {**rule_report.environmental_safety, **llm_report.environmental_safety}
        return AuditReport(
            candidate_id=rule_report.candidate_id,
            status=status,
            scientific_validity=scientific_validity,
            environmental_safety=environmental_safety,
            reviewer_risks=_dedupe(rule_report.reviewer_risks + llm_report.reviewer_risks),
            recommended_actions=_dedupe(rule_report.recommended_actions + llm_report.recommended_actions),
        )


class DecisionAgent:
    name = "Decision Agent"

    def __init__(self, weights: dict[str, float] | None = None, llm_client: OpenAIResponsesClient | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.llm_client = llm_client
        self.llm_calls = 0
        self.llm_failures: list[str] = []

    def rank(
        self,
        candidates: list[CandidateRecord],
        predictions: list[PredictionRecord],
        toxicities: list[ToxicityRecord],
        audits: list[AuditReport],
    ) -> list[DecisionRecord]:
        pred_by_id = {record.candidate_id: record for record in predictions}
        tox_by_id = {record.candidate_id: record for record in toxicities}
        audit_by_id = {record.candidate_id: record for record in audits}
        normalized = normalize_predictions(predictions)

        decisions: list[DecisionRecord] = []
        for candidate in candidates:
            prediction = pred_by_id[candidate.candidate_id]
            toxicity = tox_by_id[candidate.candidate_id]
            audit = audit_by_id[candidate.candidate_id]
            components = compute_decision_components(candidate, prediction, toxicity, normalized)
            score = weighted_score(components, self.weights)
            decision_class = self._classify(score, toxicity, audit, components)
            reasons = self._reasons(candidate, prediction, toxicity, audit, components)
            steps = experimental_next_steps(candidate, toxicity, str(candidate.evidence.get("ood_risk", "low")))
            decisions.append(
                DecisionRecord(
                    candidate_id=candidate.candidate_id,
                    decision_class=decision_class,
                    final_score=round(score, 6),
                    component_scores={k: round(v, 6) for k, v in components.items()},
                    reasons=reasons,
                    next_steps=steps,
                )
            )

        return sorted(decisions, key=lambda item: item.final_score, reverse=True)

    def annotate_top_decisions(
        self,
        decisions: list[DecisionRecord],
        candidates: list[CandidateRecord],
        predictions: list[PredictionRecord],
        toxicities: list[ToxicityRecord],
        audits: list[AuditReport],
        top_n: int,
    ) -> list[DecisionRecord]:
        if not (self.llm_client and self.llm_client.enabled):
            return decisions

        candidate_by_id = {item.candidate_id: item for item in candidates}
        prediction_by_id = {item.candidate_id: item for item in predictions}
        toxicity_by_id = {item.candidate_id: item for item in toxicities}
        audit_by_id = {item.candidate_id: item for item in audits}

        for decision in decisions[:top_n]:
            payload = {
                "decision": dataclass_to_dict(decision),
                "candidate": dataclass_to_dict(candidate_by_id[decision.candidate_id]),
                "prediction": dataclass_to_dict(prediction_by_id[decision.candidate_id]),
                "toxicity": dataclass_to_dict(toxicity_by_id[decision.candidate_id]),
                "audit": dataclass_to_dict(audit_by_id[decision.candidate_id]),
                "guardrail": "Do not change candidate_id, decision_class, or final_score.",
            }
            try:
                parsed, _meta = self.llm_client.structured_json(
                    agent_name=self.name,
                    system_prompt=DECISION_SYSTEM,
                    user_payload=payload,
                    schema_name="safe_mof_decision_narrative",
                    schema=DECISION_NARRATIVE_SCHEMA,
                )
                self.llm_calls += 1
                if parsed["candidate_id"] != decision.candidate_id:
                    raise ValueError("LLM changed candidate_id")
                if parsed["decision_class"] != decision.decision_class:
                    raise ValueError("LLM changed decision_class")
                decision.reasons = parsed["reasons"]
                decision.next_steps = parsed["next_steps"]
                decision.llm_summary = parsed["decision_summary"]
                decision.manuscript_claim = parsed["manuscript_claim"]
            except (LLMConfigurationError, LLMRequestError, TypeError, KeyError, ValueError) as exc:
                self.llm_failures.append(f"{decision.candidate_id}: {exc}")
        return decisions

    def _classify(
        self,
        score: float,
        toxicity: ToxicityRecord,
        audit: AuditReport,
        components: dict[str, float],
    ) -> str:
        if toxicity.risk_level == "high" or audit.status == "reject_or_repair":
            return "D_reject_or_repair"
        if components["uncertainty"] < 0.55:
            return "C_active_learning"
        if score >= 0.75:
            return "A_experimental_validation"
        if score >= 0.55:
            return "B_validate_before_scaleup"
        return "C_active_learning"

    def _reasons(
        self,
        candidate: CandidateRecord,
        prediction: PredictionRecord,
        toxicity: ToxicityRecord,
        audit: AuditReport,
        components: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        if components["adsorption_capacity"] >= 0.75:
            reasons.append("high dual-pollutant adsorption ranking")
        elif components["adsorption_capacity"] >= 0.50:
            reasons.append("moderate adsorption ranking")
        else:
            reasons.append("adsorption ranking is not yet compelling")

        if toxicity.risk_level == "low":
            reasons.append("low safety risk based on available evidence")
        elif toxicity.risk_level == "high":
            reasons.append("safety risk blocks direct advancement")
        else:
            reasons.append("toxicity annotation is incomplete")

        if audit.reviewer_risks:
            reasons.append("audit requires additional validation before strong claims")
        if candidate.evidence.get("benzene_vvs_mg_g") is not None:
            reasons.append("contains vapor-phase bridge evidence")
        if prediction.observed:
            reasons.append("contains GCMC or experimental validation evidence")
        return reasons


class SafeMOFAgentPipeline:
    def __init__(
        self,
        repo_root: str | Path,
        weights: dict[str, float] | None = None,
        require_llm: bool = False,
        disable_llm: bool = False,
    ):
        self.paths = build_paths(repo_root)
        self.llm_client = OpenAIResponsesClient(self.paths.repo_root, require_llm=require_llm, force_disable=disable_llm)
        self.planning_agent = PlanningAgent(self.llm_client)
        self.execution_agent = ExecutionAgent(self.paths, self.llm_client)
        self.audit_agent = AuditAgent(self.llm_client)
        self.decision_agent = DecisionAgent(weights, self.llm_client)
        self.trace: list[TraceEvent] = []

    def run(
        self,
        mode: str,
        research_goal: str,
        top_n: int = 10,
        dataset_limit: int | None = None,
    ) -> dict[str, Any]:
        task = self.planning_agent.plan(research_goal)
        self.trace.append(
            TraceEvent(
                agent=self.planning_agent.name,
                action="create_task_card",
                inputs={"research_goal": research_goal},
                outputs={
                    "pollutants": task.pollutants,
                    "workflow_steps": len(task.workflow),
                    "llm": self.planning_agent.last_meta,
                },
            )
        )

        if mode not in {"manuscript", "dataset"}:
            raise ValueError("mode must be 'manuscript' or 'dataset'")
        execution_plan = self.execution_agent.create_execution_plan(mode, dataset_limit, task)
        candidates, predictions, toxicities = self.execution_agent.execute_plan(execution_plan)

        self.trace.append(
            TraceEvent(
                agent=self.execution_agent.name,
                action=f"execute_{execution_plan['selected_tool']}",
                inputs={"execution_plan": execution_plan},
                outputs={
                    "candidates": len(candidates),
                    "predictions": len(predictions),
                    "llm": self.execution_agent.last_meta,
                },
            )
        )

        pred_by_id = {record.candidate_id: record for record in predictions}
        tox_by_id = {record.candidate_id: record for record in toxicities}
        audits = [
            self.audit_agent.audit(candidate, pred_by_id[candidate.candidate_id], tox_by_id[candidate.candidate_id])
            for candidate in candidates
        ]
        self.trace.append(
            TraceEvent(
                agent=self.audit_agent.name,
                action="audit_candidates",
                outputs={
                    "audits": len(audits),
                    "with_actions": sum(a.status != "pass" for a in audits),
                    "llm_enabled": self.llm_client.enabled,
                    "llm_calls": self.audit_agent.llm_calls,
                    "llm_failures": self.audit_agent.llm_failures[:10],
                },
            )
        )

        decisions = self.decision_agent.rank(candidates, predictions, toxicities, audits)
        decisions = self.decision_agent.annotate_top_decisions(
            decisions, candidates, predictions, toxicities, audits, top_n
        )
        self.trace.append(
            TraceEvent(
                agent=self.decision_agent.name,
                action="rank_candidates",
                outputs={
                    "decisions": len(decisions),
                    "top_candidate": decisions[0].candidate_id if decisions else None,
                    "llm_enabled": self.llm_client.enabled,
                    "llm_calls": self.decision_agent.llm_calls,
                    "llm_failures": self.decision_agent.llm_failures[:10],
                },
            )
        )

        output = {
            "run_metadata": {
                "mode": mode,
                "created_at": utc_timestamp(),
                "repo_root": str(self.paths.repo_root),
                "llm_enabled": self.llm_client.enabled,
                "llm_models": {
                    "planner": self.llm_client.settings.planner_model,
                    "execution": self.llm_client.settings.execution_model,
                    "audit": self.llm_client.settings.audit_model,
                    "decision": self.llm_client.settings.decision_model,
                },
            },
            "task_card": dataclass_to_dict(task),
            "candidates": [dataclass_to_dict(item) for item in candidates],
            "predictions": [dataclass_to_dict(item) for item in predictions],
            "toxicities": [dataclass_to_dict(item) for item in toxicities],
            "audits": [dataclass_to_dict(item) for item in audits],
            "all_decisions": [dataclass_to_dict(item) for item in decisions],
            "decisions": [dataclass_to_dict(item) for item in decisions[:top_n]],
            "trace": [dataclass_to_dict(item) for item in self.trace],
        }
        self._write_outputs(mode, output)
        return output

    def _write_outputs(self, mode: str, output: dict[str, Any]) -> None:
        report_dir = self.paths.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        write_json(report_dir / f"{mode}_safe_mof_agent_report.json", output)
        rows = []
        for decision in output.get("all_decisions", output["decisions"]):
            rows.append(
                {
                    "candidate_id": decision["candidate_id"],
                    "decision_class": decision["decision_class"],
                    "final_score": decision["final_score"],
                    "adsorption_capacity": decision["component_scores"]["adsorption_capacity"],
                    "toxicity_safety": decision["component_scores"]["toxicity_safety"],
                    "water_stability": decision["component_scores"]["water_stability"],
                    "reasons": "; ".join(decision["reasons"]),
                    "llm_summary": decision.get("llm_summary") or "",
                    "manuscript_claim": decision.get("manuscript_claim") or "",
                }
            )
        write_decision_csv(report_dir / f"{mode}_safe_mof_agent_decisions.csv", rows)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
