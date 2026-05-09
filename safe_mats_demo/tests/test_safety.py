"""Tests for safety rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.safety_rules import check_metal_safety, check_pmt_pre_filter


def test_pb_rejected():
    result = check_metal_safety(["Pb"])
    assert result["tier"] == "black"


def test_mg_passed():
    result = check_metal_safety(["Mg"])
    assert result["tier"] == "white"


def test_cu_grey():
    result = check_metal_safety(["Cu"])
    assert result["tier"] == "grey"


def test_pfas_flagged():
    result = check_pmt_pre_filter("FC(F)(F)c1ccc(C(=O)O)cc1")
    assert result["pmt_pass"] is False
    assert any("C-F" in f for f in result["flags"])


def test_bdc_passes():
    result = check_pmt_pre_filter("OC(=O)c1ccc(C(=O)O)cc1")
    assert result["pmt_pass"] is True


def test_none_smiles():
    result = check_pmt_pre_filter(None)
    assert result["pmt_pass"] is True
