"""Tests for SCM descriptor computation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from tools.descriptors import compute_scm_eigenvalues

SAMPLE_CIF = str(Path(__file__).resolve().parent / "fixtures" / "IRMOF-1.cif")


def test_shape_520():
    result = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    assert result["eigenvalues"].shape == (520,)


def test_shape_584():
    result = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=584)
    assert result["eigenvalues"].shape == (584,)


def test_deterministic():
    r1 = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    r2 = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    assert np.allclose(r1["eigenvalues"], r2["eigenvalues"])


def test_padding_metadata():
    result = compute_scm_eigenvalues(SAMPLE_CIF, target_dim=520)
    assert isinstance(result["padded"], bool)
    assert isinstance(result["raw_dim"], int)
    assert result["raw_dim"] > 0


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        compute_scm_eigenvalues("nonexistent.cif", target_dim=520)
