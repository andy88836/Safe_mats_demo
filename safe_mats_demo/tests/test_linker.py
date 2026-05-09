"""Tests for linker extraction with real MOF fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.linker_extraction import extract_linker

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_irmof1_extracts_bdc():
    result = extract_linker(str(FIXTURES / "IRMOF-1.cif"))
    assert result["extraction_level"] <= 2
    assert result["linker_smiles"] is not None
    assert "Zn" in result["metals"]
    assert "BDC" in result["extraction_note"] or "benzenedicarboxylic" in (result["linker_name"] or "").lower()


def test_zif8_extracts_meim():
    result = extract_linker(str(FIXTURES / "ZIF-8.cif"))
    assert result["extraction_level"] <= 2
    assert result["linker_smiles"] is not None
    assert "Zn" in result["metals"]
    assert "MeIm" in result["extraction_note"] or "methylimidazole" in (result["linker_name"] or "").lower()


def test_nonmof_level3():
    """Non-MOF structures should gracefully degrade to Level 3."""
    from pymatgen.core import Structure, Lattice
    import tempfile

    lattice = Lattice.cubic(5.0)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        struct.to(filename=f.name)
        result = extract_linker(f.name)
    assert result["extraction_level"] == 3
    assert result["linker_smiles"] is None
