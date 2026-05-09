"""Integration test for full LangGraph pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import screening_app

SAMPLE_CIF = str(Path(__file__).resolve().parent / "fixtures" / "IRMOF-1.cif")


def test_full_pipeline():
    result = screening_app.invoke({
        "cif_path": SAMPLE_CIF,
        "warnings": [],
        "errors": [],
    })
    assert result["mof_id"] == "IRMOF-1"
    assert result["recommendation"] in ("recommend", "borderline", "reject")
    assert 0 <= result["final_score"] <= 10
    assert len(result["errors"]) == 0
    assert result["explanation"] is not None


def test_missing_cif():
    result = screening_app.invoke({
        "cif_path": "nonexistent.cif",
        "warnings": [],
        "errors": [],
    })
    assert len(result["errors"]) > 0
