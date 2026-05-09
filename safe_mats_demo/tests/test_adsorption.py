"""Tests for adsorption prediction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from tools.descriptors import compute_scm_eigenvalues
from tools.adsorption import predict_benzene, predict_toluene

SAMPLE_CIF = str(Path(__file__).resolve().parent / "fixtures" / "IRMOF-1.cif")


def test_benzene_reasonable_range():
    scm = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    result = predict_benzene(scm["eigenvalues"])
    assert 0 <= result["uptake_mg_g"] <= 5000


def test_toluene_reasonable_range():
    scm = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=584)
    result = predict_toluene(scm["eigenvalues"])
    assert 0 <= result["uptake_mg_g"] <= 5000


def test_benzene_wrong_dim():
    with pytest.raises(ValueError):
        predict_benzene(np.zeros(100))


def test_output_schema():
    scm = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    result = predict_benzene(scm["eigenvalues"])
    assert "uptake_mg_g" in result
    assert "model_version" in result
    assert "applicability_warning" in result
