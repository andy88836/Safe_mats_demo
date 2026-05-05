# Safe-MOF-Agent manuscript insert

## Proposed framing

Safe-MOF-Agent is a human-in-the-loop, simulation-driven, safety-aware
agentic framework for discovering environmentally compatible MOF
adsorbents for water remediation. The agent layer does not replace
GCMC simulations, machine-learning adsorption models, toxicity models,
or experiments. Instead, it coordinates these validated tools, audits
scientific and environmental risks, and generates traceable candidate
decisions for experimental validation.

## Results paragraph draft

To make the simulation-assisted screening workflow executable and
auditable, we implemented Safe-MOF-Agent as a four-module orchestration
framework consisting of Planning, Execution, Audit, and Decision agents.
The Planning Agent converts a water-remediation objective into a
structured task card containing target pollutants, performance
objectives, safety constraints, and validation requirements. The
Execution Agent then loads descriptor matrices, adsorption models, and
candidate evidence from the existing GCMC/ML workflow. The Audit Agent
flags reviewer-facing risks, including incomplete toxicity annotation,
descriptor-domain uncertainty, missing vapor-to-water validation, and
insufficient water-stability evidence. Finally, the Decision Agent ranks
candidates using a multi-objective score that balances adsorption
capacity, dual-pollutant performance, toxicity safety, water stability,
synthesis feasibility, membrane processability, and uncertainty.

In the manuscript-validation mode, Safe-MOF-Agent prioritized Mg-MOF as
an A-class candidate for experimental validation because it combined high
benzene and toluene adsorption capacities with low observed aquatic
toxicity, vapor-phase validation evidence, and strong membrane
processability. In contrast, Ag-MOF retained a high adsorption score but
was classified as D-class reject-or-repair because the safety audit
identified high leaching and developmental-toxicity risks. This outcome
demonstrates that Safe-MOF-Agent does not merely select the highest
adsorption material; it operationalizes a safe-by-design decision rule
that can reject high-performing but environmentally risky candidates.

## Methods paragraph draft

Safe-MOF-Agent was implemented in Python as a modular four-agent
pipeline. Candidate records, prediction records, toxicity records, audit
reports, decision records, and trace events were represented as
structured dataclasses and serialized to JSON/CSV outputs. In dataset
mode, the Execution Agent reads the existing sine Coulomb matrix
descriptor tables for benzene and toluene, loads the trained Random
Forest benzene model and XGBoost toluene model together with their
standard scalers, and predicts adsorption capacities for candidates
shared by both datasets. In manuscript-validation mode, the Execution
Agent reads curated Mg-MOF and Ag-MOF validation records derived from
the experimental manuscript evidence. Candidate ranking used a
multi-objective weighted score with weights of 0.30 for adsorption
capacity, 0.15 for dual-pollutant performance, 0.20 for toxicity safety,
0.15 for water stability, 0.10 for synthesis feasibility, 0.05 for
membrane processability, and 0.05 for uncertainty. All agent actions
were logged as trace events to preserve provenance from task creation to
final decision.

## Figure 1 caption draft

Figure 1. Safe-MOF-Agent framework for simulation-driven safe-by-design
MOF discovery. A natural-language water-remediation objective is first
converted into a structured task card by the Planning Agent. The
Execution Agent invokes domain tools, including MOF descriptor loading,
GCMC/ML adsorption prediction, toxicity evidence, and experimental
validation records. The Audit Agent evaluates scientific validity,
descriptor-domain risk, toxicity and leaching risk, and missing
validation evidence. The Decision Agent integrates these outputs into a
multi-objective score and assigns each candidate to experimental
validation, further validation, active learning, or rejection/repair.

## Current implementation status

- Implemented: LLM-powered four-agent code path using OpenAI-compatible Responses API calls with strict JSON-schema outputs.
- Implemented: deterministic fallback path for reproducibility when `.env` is not configured.
- Implemented: traceable JSON/CSV outputs with LLM metadata, model names, tool-selection plans, audit reports, and decision summaries.
- Implemented: manuscript mode for Mg-MOF versus Ag-MOF safe-by-design logic.
- Implemented: dataset mode for 649 common benzene/toluene sine-matrix candidates.
- Not yet implemented: literature RAG, automatic CIF parsing, automatic RASPA job submission, active-learning retraining.

## Recommended next additions

1. Add ligand SMILES and metal-center annotations for screened database candidates.
2. Add QSAR toxicity and PMT/vPvM inference as callable tools.
3. Add a RASPA input generator and output parser for confirmatory GCMC.
4. Add a small LLM planner/auditor layer with JSON-schema outputs and trace logs.
5. Use the generated trace as evidence for agentic orchestration in the revised manuscript.
