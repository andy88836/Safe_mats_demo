# Model Files

Pre-trained model files (.pkl) are not included in this repository due to size constraints. You must obtain them separately before running the pipeline.

## Required Models

### Adsorption Models

| File | Description | Size |
|------|-------------|------|
| `rf_only_seed8_best_model_benzene.pkl` | Tuned RF Pipeline (benzene, 520-dim SCM input) | ~50 MB |
| `best_model_toluene.pkl` | XGBoost regressor (toluene, 584-dim SCM input) | ~15 MB |
| `scaler_toluene.pkl` | StandardScaler for toluene features | <1 MB |

### Toxicity Models (4 endpoints)

| File | Endpoint | Organism | Training N |
|------|----------|----------|------------|
| `rf_lc50.pkl` | LC50 96h | *Pimephales promelas* (fish) | 659 |
| `rf_lc50dm.pkl` | LC50 48h | *Daphnia magna* | 283 |
| `rf_igc50.pkl` | IGC50 48h | *Tetrahymena pyriformis* | 1434 |
| `rf_ibc50.pkl` | IBC50 15min | *Vibrio fischeri* | 950 |

Place toxicity models in `trained_toxicity_models/` under the project root.

## How to Obtain

### Option 1: Contact Authors (immediate)

Contact Yan Lab, South China Agricultural University for pre-trained model files.

### Option 2: Retrain from Source Data

Toxicity models can be retrained using the included training script:

```bash
python train_toxicity_models.py
```

This requires access to the AquaTox training data (SDF files). See [AquaTox GitHub](https://github.com/YanLabAI/AquaTox) for the original dataset.

Adsorption models require the original GCMC simulation dataset and Sine Coulomb Matrix features. Contact authors for the training pipeline.

### Option 3: Zenodo / Hugging Face Hub (planned)

A public release of all model files is planned for publication. This section will be updated with download links when available.

## Version Requirements

- Adsorption models: trained with `scikit-learn==1.8.0`, `xgboost==3.2.0`
- Toxicity models: trained with `scikit-learn==1.8.0`
- Loading with incompatible versions will produce incorrect predictions (especially xgboost < 3.2.0)
