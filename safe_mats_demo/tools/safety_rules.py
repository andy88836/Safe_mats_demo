"""Safety rule checks: metal blacklist + PMT pre-filter."""
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors


def _load_metals_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "metals.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_pmt_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "thresholds.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["pmt_criteria"]


def check_metal_safety(metals: list[str]) -> dict:
    """Check metal nodes against black/grey/white list.

    Args:
        metals: list of element symbols (e.g. ["Zn", "Cu"]).

    Returns:
        {
            "tier": "white" | "grey" | "black" | "unknown",
            "flagged_metals": list[str],
            "details": str,
        }
    """
    cfg = _load_metals_config()
    black = set(cfg.get("black_list", []))
    grey = set(cfg.get("grey_list", []))
    white = set(cfg.get("white_list", []))

    flagged = [m for m in metals if m in black]
    if flagged:
        return {
            "tier": "black",
            "flagged_metals": flagged,
            "details": f"Blacklisted metal(s): {', '.join(flagged)}. Hard reject.",
        }

    grey_found = [m for m in metals if m in grey]
    if grey_found:
        return {
            "tier": "grey",
            "flagged_metals": grey_found,
            "details": f"Grey-list metal(s): {', '.join(grey_found)}. Use with caution.",
        }

    white_found = [m for m in metals if m in white]
    unknown = [m for m in metals if m not in white]
    if unknown:
        return {
            "tier": "unknown",
            "flagged_metals": unknown,
            "details": f"Unknown metal(s): {', '.join(unknown)}. Not in any list.",
        }

    return {
        "tier": "white",
        "flagged_metals": [],
        "details": f"All metals ({', '.join(metals)}) are on the white list.",
    }


def check_pmt_pre_filter(linker_smiles: str | None) -> dict:
    """Screen linker SMILES against ECHA PMT criteria.

    Args:
        linker_smiles: SMILES string or None.

    Returns:
        {
            "pmt_pass": bool,
            "flags": list[str],
            "descriptors": dict | None,
        }
    """
    if linker_smiles is None:
        return {
            "pmt_pass": True,
            "flags": ["No linker SMILES available; PMT check skipped."],
            "descriptors": None,
        }

    mol = Chem.MolFromSmiles(linker_smiles)
    if mol is None:
        return {
            "pmt_pass": False,
            "flags": [f"Invalid SMILES: {linker_smiles}"],
            "descriptors": None,
        }

    pmt = _load_pmt_config()

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    descriptors = {
        "logP": round(logp, 2),
        "MW": round(mw, 2),
        "HBD": hbd,
        "HBA": hba,
    }

    flags = []
    if logp > pmt["logP_max"]:
        flags.append(f"logP={logp:.2f} > {pmt['logP_max']} (bioaccumulation risk)")
    if mw > pmt["mw_max"]:
        flags.append(f"MW={mw:.1f} > {pmt['mw_max']} (persistence concern)")
    if hbd > pmt["hbd_max"]:
        flags.append(f"HBD={hbd} > {pmt['hbd_max']}")
    if hba > pmt["hba_max"]:
        flags.append(f"HBA={hba} > {pmt['hba_max']}")

    # PFAS check: presence of C-F bonds
    has_cf = any(
        bond.GetBeginAtom().GetSymbol() == "F" or bond.GetEndAtom().GetSymbol() == "F"
        for bond in mol.GetBonds()
    )
    if has_cf:
        flags.append("Contains C-F bonds (potential PFAS concern)")

    return {
        "pmt_pass": len(flags) == 0,
        "flags": flags if flags else ["All PMT criteria passed."],
        "descriptors": descriptors,
    }
