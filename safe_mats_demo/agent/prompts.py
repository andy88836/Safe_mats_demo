"""LLM prompt templates for the score_and_explain node."""


def build_explanation_prompt(state: dict, score: float, recommendation: str) -> str:
    adsorption = state.get("adsorption") or {}
    safety = state.get("safety") or {}
    toxicity = state.get("toxicity") or {}
    linker = state.get("linker") or {}
    mof_id = state.get("mof_id", "unknown")

    return f"""You are an expert materials scientist evaluating a Metal-Organic Framework (MOF)
for environmental safety and adsorption performance.

## MOF: {mof_id}

### Adsorption Performance
- Benzene uptake: {adsorption.get('benzene_uptake_mg_g', 'N/A')} mg/g
- Toluene uptake: {adsorption.get('toluene_uptake_mg_g', 'N/A')} mg/g

### Metal Node Safety
- Metals: {', '.join(linker.get('metals', []))}
- Safety tier: {safety.get('metal_tier', 'unknown')}
- Details: {safety.get('metal_details', 'N/A')}

### Organic Linker
- SMILES: {linker.get('linker_smiles', 'N/A')}
- Name: {linker.get('linker_name', 'N/A')}
- PMT pass: {safety.get('pmt_pass', 'N/A')}

### Aquatic Toxicity (-log scale, higher = more toxic)
- LC50 Pimephales promelas: {toxicity.get('LC50_Pimephales', 'N/A')}
- LC50 Daphnia magna: {toxicity.get('LC50_Daphnia', 'N/A')}
- IGC50 Tetrahymena pyriformis: {toxicity.get('IGC50_Tetrahymena', 'N/A')}
- IBC50 Vibrio fischeri: {toxicity.get('IBC50_Vibrio', 'N/A')}

### Computed Score
- Final score: {score:.2f} / 10.0
- Recommendation: {recommendation}

## Instructions
Write a 3-5 sentence natural language interpretation of these results.
- Reference specific numerical values.
- Identify the biggest risk factor (low adsorption / high toxicity / unsafe metal).
- Do NOT modify the final score or recommendation — those are deterministic.
- Be concise and scientifically accurate.
"""
