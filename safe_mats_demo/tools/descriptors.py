"""CIF -> Sine Coulomb Matrix eigenvalues descriptor computation."""
from pathlib import Path

import numpy as np
from matminer.featurizers.structure.matrix import SineCoulombMatrix
from pymatgen.core import Structure


def compute_scm_eigenvalues(
    cif_path: str,
    target_dim: int,
) -> dict:
    """Compute Sine Coulomb Matrix eigenvalues from a CIF file.

    Args:
        cif_path: absolute path to a .cif file.
        target_dim: desired output dimension (520 for benzene, 584 for toluene).

    Returns:
        {
            "eigenvalues": np.ndarray of shape (target_dim,),
            "raw_dim": int,
            "padded": bool,
            "truncated": bool,
            "applicability_warning": str | None,
        }

    Raises:
        FileNotFoundError: if cif_path does not exist.
        ValueError: if CIF cannot be parsed into a valid Structure.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(str(path))
    if structure.num_sites == 0:
        raise ValueError(f"CIF parsed but contains 0 sites: {cif_path}")

    scm = SineCoulombMatrix(flatten=False)
    scm.fit([structure])
    matrix = scm.featurize(structure)[0]

    eigenvalues = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    raw_dim = len(eigenvalues)

    padded = False
    truncated = False
    warning = None

    if raw_dim < target_dim:
        eigenvalues = np.pad(eigenvalues, (0, target_dim - raw_dim), mode="constant")
        padded = True
        warning = (
            f"SCM eigenvalue dimension ({raw_dim}) < target ({target_dim}). "
            f"Zero-padded {target_dim - raw_dim} values. "
            f"Prediction may be outside the model's applicability domain."
        )
    elif raw_dim > target_dim:
        eigenvalues = eigenvalues[:target_dim]
        truncated = True
        warning = (
            f"SCM eigenvalue dimension ({raw_dim}) > target ({target_dim}). "
            f"Truncated {raw_dim - target_dim} smallest eigenvalues. "
            f"Prediction may be outside the model's applicability domain."
        )

    return {
        "eigenvalues": eigenvalues,
        "raw_dim": raw_dim,
        "padded": padded,
        "truncated": truncated,
        "applicability_warning": warning,
    }
