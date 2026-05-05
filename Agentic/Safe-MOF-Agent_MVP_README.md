# Safe-MOF-Agent MVP

This folder contains the runnable outputs for the Safe-MOF-Agent MVP.

## What it implements

- Planning Agent: calls an LLM to convert a water-remediation goal into a structured task card, with deterministic fallback.
- Execution Agent: calls an LLM to select a local tool, then loads either manuscript validation records or existing sine-matrix ML artifacts.
- Audit Agent: combines rule-based safety checks with LLM-generated structured audit reports.
- Decision Agent: ranks candidates with deterministic multi-objective safe-by-design scoring, then uses the LLM for manuscript-ready reasoning on top candidates.

## Run

From `D:\Doc_Projects\All_des_cal`:

```powershell
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode manuscript --top-n 5
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode dataset --dataset-limit 50 --top-n 10
```

The full dataset mode can be run by omitting `--dataset-limit`.

To require real LLM calls, fill `.env` and run:

```powershell
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode manuscript --top-n 2 --require-llm
```

To force local deterministic fallback:

```powershell
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode manuscript --top-n 2 --no-llm
```

## Outputs

Reports are written to:

```text
Agentic/safe_mof_agent_reports/
```

Key files:

- `manuscript_safe_mof_agent_report.json`
- `manuscript_safe_mof_agent_decisions.csv`
- `dataset_safe_mof_agent_report.json`
- `dataset_safe_mof_agent_decisions.csv`

## Manuscript positioning

Use this MVP as the implementation basis for a human-in-the-loop,
simulation-driven, safety-aware MOF discovery framework. The LLM/agent
layer does not replace GCMC, ML, toxicity prediction, or experiments; it
orchestrates these tools, audits risks, and produces traceable candidate
decisions.
