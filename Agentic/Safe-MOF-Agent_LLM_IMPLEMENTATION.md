# Safe-MOF-Agent LLM implementation

This version contains a real LLM-powered agentic layer. The API key and
model names are intentionally left blank in `.env` for you to fill.

## Architecture

- Planning Agent: calls an OpenAI-compatible Responses API endpoint and
  generates a strict JSON task card.
- Execution Agent: calls the LLM to select an execution tool, then local
  Python executes that tool. Available tools are:
  - `load_manuscript_records`
  - `run_dataset_screen`
- Audit Agent: first creates a deterministic rule audit, then calls the
  LLM to produce a structured audit report. Critical rule flags are
  preserved by a guardrail merge.
- Decision Agent: computes deterministic scores/classes, then calls the
  LLM to generate manuscript-ready reasons and next-step experiments for
  the top candidates. The LLM is not allowed to change score or class.

## Configuration

Edit:

```text
D:\Doc_Projects\All_des_cal\.env
```

Minimum configuration:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=your_model_here
```

Optional per-agent models:

```text
OPENAI_PLANNER_MODEL=your_planning_model
OPENAI_EXECUTION_MODEL=your_execution_model
OPENAI_AUDIT_MODEL=your_audit_model
OPENAI_DECISION_MODEL=your_decision_model
```

## Run with LLM required

```powershell
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode manuscript --top-n 2 --require-llm
```

If `.env` is not filled, `--require-llm` fails immediately. Without
`--require-llm`, the workflow falls back to deterministic local agents
for reproducibility.

## Run deterministic fallback

```powershell
.\.venv\Scripts\python.exe .\run_safe_mof_agent.py --mode manuscript --top-n 2 --no-llm
```

## Output evidence

Each report includes:

- `run_metadata.llm_enabled`
- per-agent model names
- trace events for Planning, Execution, Audit, and Decision
- LLM usage metadata when API calls run
- deterministic scores and classifications
- LLM-generated decision summaries when enabled

Reports are written to:

```text
Agentic/safe_mof_agent_reports/
```
