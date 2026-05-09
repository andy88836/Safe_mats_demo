# Scoring Formula

Weights reflect the safe-by-design philosophy: toxicity and safety together (0.6) outweigh raw adsorption capacity (0.4).

## Final Score Computation

```
final_score = 0.4 * adsorption_score + 0.3 * safety_score + 0.3 * tox_score
```

Range: [0.0, 10.0]

### Adsorption Score (0-10)

```
adsorption_score = min(benzene_uptake / 1000.0, 1.0) * 10
```

### Safety Score (0-10)

| Metal Tier | Base Score |
|-----------|-----------|
| white | 10.0 |
| grey | 7.0 |
| black | 0.0 |
| unknown | 5.0 |

If PMT pre-filter fails: `safety_score = max(safety_score - 3, 0)`

### Toxicity Score (0-10)

```
mean_tox = mean of available -log toxicity values
tox_score = clamp((5.0 - mean_tox) * 2 + 5, 0, 10)
```

If no toxicity data: tox_score = 5.0 (neutral)

## Decision Thresholds

| Score Range | Recommendation |
|------------|----------------|
| >= 7.0 | recommend |
| 5.0 - 7.0 | borderline |
| < 5.0 | reject |

## Toxicity Model Confidence

| Endpoint | Organism | CV R² | N_train | Confidence |
|----------|----------|-------|---------|------------|
| LC50 | P. promelas (fish) | 0.649 | 659 | high |
| LC50DM | D. magna | 0.597 | 283 | medium |
| IGC50 | T. pyriformis | 0.691 | 1434 | high |
| IBC50 | V. fischeri | 0.480 | 950 | low |

Endpoints with low/medium confidence are marked with ⚠️/⚡ in the UI.
