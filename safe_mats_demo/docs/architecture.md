# Architecture

## LangGraph State Machine

Linear pipeline, 7 nodes, no branching or loops.

### Node List

| Node | Function | Tool Called | Output Key |
|------|----------|------------|------------|
| ingest_cif | Parse CIF path, extract MOF ID | — | mof_id |
| compute_descriptor | CIF → SCM eigenvalues (520 + 584 dim) | tools/descriptors.py | scm_meta |
| predict_adsorption | SCM → benzene/toluene uptake | tools/adsorption.py | adsorption |
| extract_linker | CIF → metals + linker SMILES (3-level) | tools/linker_extraction.py | linker |
| predict_toxicity | SMILES → 4 aquatic toxicity endpoints | tools/toxicity.py | toxicity |
| apply_safety_rules | Metal blacklist + PMT pre-filter | tools/safety_rules.py | safety |
| score_and_explain | Deterministic scoring + LLM/rule explanation | — | final_score, recommendation, explanation |

### Data Flow

```
ScreeningState flows through each node sequentially.
Each node reads from state and writes its output key.
Errors are accumulated in state["errors"], never raised.
Warnings are accumulated in state["warnings"].
```

### LLM Usage

LLM is only used in `score_and_explain` for natural language interpretation.
It does NOT compute scores or make decisions.
Fallback: rule-based template explanation when no API key is available.

Priority: OPENAI_API_KEY → DASHSCOPE_API_KEY → rule-based fallback.
