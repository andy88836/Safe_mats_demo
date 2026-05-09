# End-to-End Test Log

Date: 2026-05-08
Environment: Python 3.11.7, sklearn 1.8.0, xgboost 3.2.0 (safe_mats_demo/.venv)

## 1. Unit Tests (pytest)

```
20 passed in 28.39s

tests/test_adsorption.py::test_benzene_reasonable_range PASSED
tests/test_adsorption.py::test_toluene_reasonable_range PASSED
tests/test_adsorption.py::test_benzene_wrong_dim PASSED
tests/test_adsorption.py::test_output_schema PASSED
tests/test_descriptors.py::test_shape_520 PASSED
tests/test_descriptors.py::test_shape_584 PASSED
tests/test_descriptors.py::test_deterministic PASSED
tests/test_descriptors.py::test_padding_metadata PASSED
tests/test_descriptors.py::test_file_not_found PASSED
tests/test_linker.py::test_irmof1_extracts_bdc PASSED
tests/test_linker.py::test_zif8_extracts_meim PASSED
tests/test_linker.py::test_nonmof_level3 PASSED
tests/test_pipeline.py::test_full_pipeline PASSED
tests/test_pipeline.py::test_missing_cif PASSED
tests/test_safety.py::test_pb_rejected PASSED
tests/test_safety.py::test_mg_passed PASSED
tests/test_safety.py::test_cu_grey PASSED
tests/test_safety.py::test_pfas_flagged PASSED
tests/test_safety.py::test_bdc_passes PASSED
tests/test_safety.py::test_none_smiles PASSED
```

## 2. Batch Screening (batch_screen.py)

Command: `python examples/batch_screen.py --cif-dir tests/fixtures --out tests/fixtures/test_results.csv`

Input: 3 CIF files (IRMOF-1.cif, ZIF-8.cif, NaCl.cif)

### Results Summary

| MOF ID | Metal | Linker SMILES | Benzene (mg/g) | Toluene (mg/g) | Score | Recommendation |
|--------|-------|---------------|----------------|----------------|-------|----------------|
| IRMOF-1 | Zn | OC(=O)c1ccc(C(=O)O)cc1 (BDC) | 83.44 | 76.88 | 5.74 | borderline |
| ZIF-8 | Zn | Cc1[nH]ccn1 (MeIm) | 139.73 | 780.20 | 6.56 | borderline |
| NaCl | Na | N/A (Level 3) | 249.60 | — | 5.50 | borderline |

Summary: 0 recommend, 3 borderline, 0 reject

### Per-CIF Node Outputs

#### IRMOF-1

- **Linker extraction**: Level 1 match → BDC (1,4-benzenedicarboxylic acid)
- **Metal**: Zn → white tier
- **PMT filter**: Pass
- **Toxicity**: LC50=3.14, LC50DM=4.30, IGC50=2.76, IBC50=3.76
- **Warnings**: SCM dim 424 < 520/584, zero-padded (applicability domain flag)

#### ZIF-8

- **Linker extraction**: Level 1 match → MeIm (2-methylimidazole)
- **Metal**: Zn → white tier
- **PMT filter**: Pass
- **Toxicity**: LC50=2.22, LC50DM=3.34, IGC50=2.11, IBC50=2.15
- **Warnings**: SCM dim 46 < 520/584, heavy padding (applicability domain flag)

### Explanation Examples

IRMOF-1:
> MOF IRMOF-1 achieves 83.4 mg/g benzene uptake. Metal node(s) [Zn] are in the white tier. Predicted aquatic toxicity: mean -log toxicity = 3.49. Final score: 5.74 -> borderline.

ZIF-8:
> MOF ZIF-8 achieves 139.7 mg/g benzene uptake. Metal node(s) [Zn] are in the white tier. Predicted aquatic toxicity: mean -log toxicity = 2.45. Final score: 6.56 -> borderline.

## 3. Model Validation (validate_models.py)

| Model | R² | MAE | Paper | Deviation |
|-------|-----|------|-------|-----------|
| Benzene (tuned RF, rs=8) | 0.7855 | 17.77 | 0.777 | +1.1% |
| Toluene (XGBoost, rs=42) | 0.9819 | 11.43 | 0.9819 | 0.00% |

Both within acceptable tolerance (< 0.1% for toluene, +1.1% for benzene due to tuning improvement).
