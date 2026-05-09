"""CIF -> metal symbols + linker SMILES extraction with 3-level degradation."""
import math
import re
from pathlib import Path

import yaml
from pymatgen.core import Structure


_METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "U", "Th",
}


def _load_known_linkers() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "known_linkers.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("linkers", {})


def _extract_metals(structure: Structure) -> list[str]:
    metals = set()
    for site in structure:
        symbol = site.specie.symbol
        if symbol in _METAL_ELEMENTS:
            metals.add(symbol)
    return sorted(metals)


def _get_organic_elements(structure: Structure) -> dict[str, int]:
    organic = {}
    for site in structure:
        symbol = site.specie.symbol
        if symbol not in _METAL_ELEMENTS:
            organic[symbol] = organic.get(symbol, 0) + 1
    return organic


def _format_formula(elements: dict[str, int]) -> str:
    parts = []
    for el in ["C", "H", "N", "O", "F", "Cl", "Br", "S", "P"]:
        if el in elements:
            count = elements[el]
            parts.append(f"{el}{count}" if count > 1 else el)
    for el in sorted(set(elements.keys()) - {"C", "H", "N", "O", "F", "Cl", "Br", "S", "P"}):
        count = elements[el]
        parts.append(f"{el}{count}" if count > 1 else el)
    return "".join(parts)


def _parse_formula(formula: str) -> dict[str, int]:
    elements = {}
    for m in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        el, count = m.groups()
        if el:
            elements[el] = int(count) if count else 1
    return elements


def _deprotonate_linker(formula: dict[str, int]) -> list[dict[str, int]]:
    """Generate deprotonated variants by removing 1-6 H atoms (carboxylate deprotonation)."""
    variants = [formula.copy()]
    if "H" in formula:
        for n_remove in range(1, min(7, formula["H"] + 1)):
            v = formula.copy()
            v["H"] = formula["H"] - n_remove
            if v["H"] == 0:
                del v["H"]
            variants.append(v)
    return variants


def _score_linker_match(
    linker_elements: dict[str, int],
    observed: dict[str, int],
    n_metals: int,
) -> tuple[float, int]:
    """Score how well a linker formula matches observed organic composition.

    Uses C (or N if no C) as anchor to determine multiplier.
    Allows O excess (metal-oxide clusters) and H tolerance (protonation states).

    Returns (score, multiplier) where score 0.0=no match, 1.0=perfect.
    """
    anchor = "C" if "C" in linker_elements else ("N" if "N" in linker_elements else None)
    if anchor is None or anchor not in observed:
        return 0.0, 0

    multiplier_raw = observed[anchor] / linker_elements[anchor]
    multiplier = round(multiplier_raw)
    if multiplier < 1 or abs(multiplier_raw - multiplier) > 0.1:
        return 0.0, 0

    score = 1.0
    penalties = 0.0

    for el in linker_elements:
        expected = linker_elements[el] * multiplier
        actual = observed.get(el, 0)

        if el == anchor:
            if actual != expected:
                return 0.0, 0
            continue

        if el == "O":
            if actual < expected:
                penalties += 0.3
            elif actual > expected:
                excess = actual - expected
                max_cluster_o = n_metals * 2
                if excess <= max_cluster_o:
                    penalties += 0.05
                else:
                    penalties += 0.4
            continue

        if el == "H":
            tolerance = max(multiplier * 2, 4)
            if abs(actual - expected) <= tolerance:
                penalties += min(abs(actual - expected) / (expected + 1) * 0.2, 0.2)
            else:
                penalties += 0.5
            continue

        if actual != expected:
            if abs(actual - expected) / max(expected, 1) < 0.15:
                penalties += 0.1
            else:
                return 0.0, 0

    for el in observed:
        if el not in linker_elements and el != "O":
            return 0.0, 0

    final_score = max(0.0, score - penalties)
    return final_score, multiplier


def extract_linker(cif_path: str) -> dict:
    """Extract metal nodes and organic linker information from a CIF file.

    Three-level degradation:
        Level 1: exact formula match (direct or deprotonated variant).
        Level 2: ratio-based match with tolerance for cluster atoms.
        Level 3: unable_to_assess (metals still reported).

    Args:
        cif_path: absolute path to a .cif file.

    Returns:
        {
            "metals": list[str],
            "linker_smiles": str | None,
            "linker_name": str | None,
            "linker_formula": str | None,
            "extraction_level": int,
            "extraction_note": str,
        }

    Raises:
        FileNotFoundError: if cif_path does not exist.
        ValueError: if CIF cannot be parsed.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(str(path))
    metals = _extract_metals(structure)
    observed = _get_organic_elements(structure)
    organic_formula = _format_formula(observed)
    known_linkers = _load_known_linkers()

    if not observed or "C" not in observed:
        return {
            "metals": metals,
            "linker_smiles": None,
            "linker_name": None,
            "linker_formula": organic_formula if organic_formula else None,
            "extraction_level": 3,
            "extraction_note": (
                f"No organic carbon detected. Formula: {organic_formula}. "
                "Adsorption predictions remain valid; toxicity assessment skipped."
            ),
        }

    # Level 1: exact formula match (including deprotonated variants)
    for key, info in known_linkers.items():
        linker_el = _parse_formula(info.get("formula", ""))
        for variant in _deprotonate_linker(linker_el):
            if variant == observed:
                return {
                    "metals": metals,
                    "linker_smiles": info["smiles"],
                    "linker_name": info["name"],
                    "linker_formula": info["formula"],
                    "extraction_level": 1,
                    "extraction_note": f"Exact formula match: '{key}'.",
                }

    # Level 2: ratio-based match with scoring
    n_metals = sum(1 for s in structure if s.specie.symbol in _METAL_ELEMENTS)
    best_score = 0.0
    best_match = None
    best_key = None
    best_multiplier = 0

    for key, info in known_linkers.items():
        linker_el = _parse_formula(info.get("formula", ""))
        for variant in _deprotonate_linker(linker_el):
            score, mult = _score_linker_match(variant, observed, n_metals)
            if score > best_score:
                best_score = score
                best_match = info
                best_key = key
                best_multiplier = mult

    if best_score >= 0.5 and best_match is not None:
        return {
            "metals": metals,
            "linker_smiles": best_match["smiles"],
            "linker_name": best_match["name"],
            "linker_formula": best_match["formula"],
            "extraction_level": 2,
            "extraction_note": (
                f"Ratio match to '{best_key}' (score={best_score:.2f}, "
                f"~{best_multiplier} linkers/cell). "
                f"Organic formula: {organic_formula}."
            ),
        }

    # Level 3: unable to assess
    return {
        "metals": metals,
        "linker_smiles": None,
        "linker_name": None,
        "linker_formula": organic_formula if organic_formula else None,
        "extraction_level": 3,
        "extraction_note": (
            f"Unable to identify linker. Organic formula: {organic_formula}. "
            "Adsorption predictions remain valid; toxicity assessment skipped."
        ),
    }
