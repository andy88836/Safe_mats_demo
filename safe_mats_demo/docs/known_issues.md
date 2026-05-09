# Known Issues

## 1. SCM Dimension Mismatch for Large/Small MOFs

**Issue:** MOFs with very few atoms produce SCM eigenvalue vectors much shorter than
the training set (520/584 dimensions). Extensive zero-padding may reduce prediction accuracy.

**Mitigation:** `applicability_warning` is set when padding/truncation occurs and
displayed in the UI with a yellow highlight.

## 2. Linker SMILES Extraction Limitations

**Issue:** Level 1/2 extraction relies on formula matching against a known linker database
of 15 entries. Complex or novel linkers will fall through to Level 3 (unable_to_assess).

**Mitigation:** Level 3 still reports metals and adsorption predictions. Only toxicity
assessment is skipped. Users can manually provide SMILES in future versions.

## 3. AquaTox Model Performance

**Issue:** RF models trained on Morgan fingerprints show moderate test R² (0.45-0.74)
depending on endpoint. LC50DM (Daphnia magna) has the weakest performance (R²=0.45).

**Mitigation:** Results should be interpreted as screening-level estimates, not
definitive toxicity predictions. Flag in the UI when toxicity predictions have
high uncertainty.

## 4. rdkit-pypi vs numpy 2.x Compatibility

**Issue:** `rdkit-pypi` package (2022.x) is incompatible with numpy 2.x.
Must use `rdkit>=2024.3` (installed as `rdkit` not `rdkit-pypi`).

**Mitigation:** requirements.txt specifies `rdkit>=2024.3.0`.

## 5. XGBoost Version Lock

**Issue:** Toluene model was trained with xgboost 3.2.0. Loading with xgboost 2.x
produces incorrect predictions (R²=-2.8).

**Mitigation:** requirements.txt locks `xgboost>=3.2.0`.

## 6. Adsorption Models Only Support Benzene/Toluene

**Issue:** No models exist for other VOCs or gas-phase adsorption.

**Mitigation:** Clearly documented in README. Future work could extend to other adsorbates.

## 7. CIF Parsing Edge Cases

**Issue:** Some CIF files with non-standard formatting or partial occupancy may fail
pymatgen parsing.

**Mitigation:** Errors are caught per-CIF and reported in the results table without
crashing the entire batch.
