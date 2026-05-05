from __future__ import annotations


TASK_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "research_goal": {"type": "string"},
        "pollutants": {"type": "array", "items": {"type": "string"}},
        "material_class": {"type": "string"},
        "objectives": {"type": "array", "items": {"type": "string"}},
        "constraints": {
            "type": "object",
            "properties": {
                "exclude_highly_toxic_metals": {"type": "array", "items": {"type": "string"}},
                "require_traceable_descriptor_source": {"type": "boolean"},
                "require_human_experimental_validation": {"type": "boolean"},
                "preferred_validation": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "exclude_highly_toxic_metals",
                "require_traceable_descriptor_source",
                "require_human_experimental_validation",
                "preferred_validation",
                "notes",
            ],
            "additionalProperties": False,
        },
        "workflow": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["research_goal", "pollutants", "material_class", "objectives", "constraints", "workflow"],
    "additionalProperties": False,
}


EXECUTION_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_tool": {
            "type": "string",
            "enum": ["load_manuscript_records", "run_dataset_screen"],
        },
        "tool_arguments": {
            "type": "object",
            "properties": {
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
        "expected_outputs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["selected_tool", "tool_arguments", "rationale", "expected_outputs"],
    "additionalProperties": False,
}


AUDIT_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["pass", "pass_with_actions", "reject_or_repair"],
        },
        "scientific_validity": {
            "type": "object",
            "properties": {
                "prediction_available": {"type": "boolean"},
                "descriptor_or_evidence_source": {"type": "string"},
                "ood_risk": {"type": "string"},
                "evidence_assessment": {"type": "string"},
            },
            "required": [
                "prediction_available",
                "descriptor_or_evidence_source",
                "ood_risk",
                "evidence_assessment",
            ],
            "additionalProperties": False,
        },
        "environmental_safety": {
            "type": "object",
            "properties": {
                "metal_toxicity_flag": {"type": "boolean"},
                "ligand_toxicity_risk": {"type": "string"},
                "pmt_vpvm_risk": {"type": "string"},
                "leaching_risk": {"type": "string"},
                "overall_risk": {"type": "string"},
                "safety_assessment": {"type": "string"},
            },
            "required": [
                "metal_toxicity_flag",
                "ligand_toxicity_risk",
                "pmt_vpvm_risk",
                "leaching_risk",
                "overall_risk",
                "safety_assessment",
            ],
            "additionalProperties": False,
        },
        "reviewer_risks": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "candidate_id",
        "status",
        "scientific_validity",
        "environmental_safety",
        "reviewer_risks",
        "recommended_actions",
    ],
    "additionalProperties": False,
}


DECISION_NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "decision_class": {"type": "string"},
        "decision_summary": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "manuscript_claim": {"type": "string"},
    },
    "required": [
        "candidate_id",
        "decision_class",
        "decision_summary",
        "reasons",
        "next_steps",
        "manuscript_claim",
    ],
    "additionalProperties": False,
}
