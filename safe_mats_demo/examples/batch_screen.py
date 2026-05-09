"""Command-line batch screening script."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from agent.graph import screening_app


def screen_single(cif_path: str) -> dict:
    result = screening_app.invoke({
        "cif_path": cif_path,
        "warnings": [],
        "errors": [],
    })
    adsorption = result.get("adsorption") or {}
    linker = result.get("linker") or {}
    toxicity = result.get("toxicity") or {}
    safety = result.get("safety") or {}

    return {
        "mof_id": result.get("mof_id", "unknown"),
        "metal": ", ".join(linker.get("metals", [])),
        "linker_smiles": linker.get("linker_smiles"),
        "benzene_mg_g": adsorption.get("benzene_uptake_mg_g"),
        "toluene_mg_g": adsorption.get("toluene_uptake_mg_g"),
        "LC50_Pimephales": toxicity.get("LC50_Pimephales"),
        "LC50_Daphnia": toxicity.get("LC50_Daphnia"),
        "IGC50_Tetrahymena": toxicity.get("IGC50_Tetrahymena"),
        "IBC50_Vibrio": toxicity.get("IBC50_Vibrio"),
        "metal_tier": safety.get("metal_tier"),
        "pmt_pass": safety.get("pmt_pass"),
        "score": result.get("final_score"),
        "recommendation": result.get("recommendation"),
        "explanation": result.get("explanation"),
        "warnings": "; ".join(result.get("warnings", [])),
        "errors": "; ".join(result.get("errors", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch MOF screening")
    parser.add_argument("--cif-dir", required=True, help="Directory containing CIF files")
    parser.add_argument("--out", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    if not cif_dir.exists():
        print(f"Error: directory not found: {cif_dir}")
        sys.exit(1)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        print(f"No .cif files found in {cif_dir}")
        sys.exit(1)

    print(f"Found {len(cif_files)} CIF files in {cif_dir}")
    rows = []
    for i, cif_file in enumerate(cif_files, 1):
        print(f"  [{i}/{len(cif_files)}] {cif_file.name}...", end=" ", flush=True)
        try:
            row = screen_single(str(cif_file))
            rows.append(row)
            print(f"score={row['score']}, {row['recommendation']}")
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({"mof_id": cif_file.stem, "errors": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")

    recommend = sum(1 for r in rows if r.get("recommendation") == "recommend")
    borderline = sum(1 for r in rows if r.get("recommendation") == "borderline")
    reject = sum(1 for r in rows if r.get("recommendation") == "reject")
    print(f"Summary: {recommend} recommend, {borderline} borderline, {reject} reject")


if __name__ == "__main__":
    main()
