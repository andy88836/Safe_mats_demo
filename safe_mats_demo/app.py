"""MOFScreen-Agent: Streamlit web demo for MOF screening."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.graph import screening_app


def load_paths_config():
    config_path = Path(__file__).resolve().parent / "configs" / "paths.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_model_files(cfg):
    checks = {}
    ads = cfg["adsorption_models"]
    checks["Benzene model"] = Path(ads["benzene_model"]).exists()
    checks["Toluene model"] = Path(ads["toluene_model"]).exists()
    checks["Toluene scaler"] = Path(ads["toluene_scaler"]).exists()
    tox_dir = Path(cfg["toxicity_models"]["model_dir"])
    for ep in ["rf_lc50.pkl", "rf_lc50dm.pkl", "rf_igc50.pkl", "rf_ibc50.pkl"]:
        checks[f"Tox: {ep}"] = (tox_dir / ep).exists()
    return checks


def run_screening(cif_path: str) -> dict:
    return screening_app.invoke({
        "cif_path": cif_path,
        "warnings": [],
        "errors": [],
    })


def _fmt_tox(val, confidence):
    if val is None:
        return "N/A"
    icon = ""
    if confidence == "low":
        icon = " ⚠️"
    elif confidence == "medium":
        icon = " ⚡"
    return f"{val:.2f}{icon}"


def result_to_row(result: dict) -> dict:
    adsorption = result.get("adsorption") or {}
    linker = result.get("linker") or {}
    toxicity = result.get("toxicity") or {}
    safety = result.get("safety") or {}

    return {
        "MOF ID": result.get("mof_id", "unknown"),
        "Metal": ", ".join(linker.get("metals", [])),
        "Linker": linker.get("linker_name") or linker.get("linker_formula") or "N/A",
        "Linker SMILES": linker.get("linker_smiles") or "N/A",
        "Benzene (mg/g)": adsorption.get("benzene_uptake_mg_g"),
        "Toluene (mg/g)": adsorption.get("toluene_uptake_mg_g"),
        "LC50 fish (-log mol/L)": _fmt_tox(
            toxicity.get("LC50_Pimephales"),
            toxicity.get("LC50_Pimephales_confidence")),
        "LC50 daphnia (-log mol/L)": _fmt_tox(
            toxicity.get("LC50_Daphnia"),
            toxicity.get("LC50_Daphnia_confidence")),
        "IGC50 tetrahymena (-log mol/L)": _fmt_tox(
            toxicity.get("IGC50_Tetrahymena"),
            toxicity.get("IGC50_Tetrahymena_confidence")),
        "IBC50 vibrio (-log mol/L)": _fmt_tox(
            toxicity.get("IBC50_Vibrio"),
            toxicity.get("IBC50_Vibrio_confidence")),
        "Safety": safety.get("metal_tier", "N/A"),
        "PMT": "Pass" if safety.get("pmt_pass", True) else "Fail",
        "Score": result.get("final_score"),
        "Recommendation": result.get("recommendation", "N/A"),
    }


def _get_worst_tox(toxicity: dict) -> tuple[float | None, str]:
    endpoints = {
        "LC50_Pimephales": "P. promelas (fish)",
        "LC50_Daphnia": "D. magna",
        "IGC50_Tetrahymena": "T. pyriformis",
        "IBC50_Vibrio": "V. fischeri",
    }
    worst_val = None
    worst_species = "N/A"
    for key, species in endpoints.items():
        val = toxicity.get(key)
        if val is not None:
            if worst_val is None or val > worst_val:
                worst_val = val
                worst_species = species
    return worst_val, worst_species


def main():
    st.set_page_config(page_title="MOFScreen-Agent", page_icon="🔬", layout="wide")

    st.title("MOFScreen-Agent")
    st.markdown("Simulation-in-the-loop screening for environmentally compatible MOFs")
    st.caption(
        "Toxicity values are in -log₁₀(mol/L): higher = more toxic. "
        "⚠️ = low model confidence (CV R² < 0.50), ⚡ = medium confidence (CV R² 0.50–0.65)."
    )

    with st.sidebar:
        st.header("Settings")
        uploaded_files = st.file_uploader("Upload CIF files", type=["cif"], accept_multiple_files=True)
        llm_provider = st.selectbox(
            "LLM Provider", ["Rule-based fallback", "OpenAI (gpt-4o-mini)", "Qwen (qwen-turbo)"]
        )
        run_button = st.button("Run Screening", type="primary", disabled=not uploaded_files)

        st.divider()
        st.subheader("Model Status")
        cfg = load_paths_config()
        checks = check_model_files(cfg)
        for name, ok in checks.items():
            st.text(f"{'✅' if ok else '❌'} {name}")

    if not uploaded_files:
        st.info("Upload one or more CIF files in the sidebar to begin screening.")
        return
    if not run_button:
        st.info(f"{len(uploaded_files)} file(s) uploaded. Click **Run Screening** to start.")
        return

    results = []
    full_results = []
    progress_bar = st.progress(0, text="Starting screening...")

    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(i / len(uploaded_files),
                              text=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            result = run_screening(tmp_path)
            result["mof_id"] = Path(uploaded_file.name).stem
            full_results.append(result)
            if result.get("errors"):
                st.warning(f"⚠️ {uploaded_file.name}: {'; '.join(result['errors'])}")
            results.append(result_to_row(result))
        except Exception as e:
            st.error(f"❌ {uploaded_file.name} failed: {e}")
            results.append({"MOF ID": Path(uploaded_file.name).stem, "Recommendation": "error"})
            full_results.append({"mof_id": Path(uploaded_file.name).stem, "errors": [str(e)]})

    progress_bar.progress(1.0, text="Screening complete!")
    if not results:
        st.warning("No results to display.")
        return

    # Results table
    st.subheader("Screening Results")
    df = pd.DataFrame(results)

    def highlight_rec(val):
        colors = {"recommend": "#d4edda", "reject": "#f8d7da", "borderline": "#fff3cd"}
        return f"background-color: {colors.get(val, '')}"

    styled = df.style.applymap(highlight_rec, subset=["Recommendation"])
    st.dataframe(styled, use_container_width=True)

    csv_data = df.to_csv(index=False)
    st.download_button("Download results as CSV", csv_data,
                       file_name="mofscreen_results.csv", mime="text/csv")

    # Pareto plot: worst-case toxicity
    st.subheader("Pareto Plot: Adsorption vs Worst-Case Toxicity")
    plot_rows = []
    for result in full_results:
        ads = result.get("adsorption") or {}
        tox = result.get("toxicity") or {}
        benzene = ads.get("benzene_uptake_mg_g")
        if benzene is None:
            continue
        worst_val, worst_species = _get_worst_tox(tox)
        plot_rows.append({
            "MOF ID": result.get("mof_id", "unknown"),
            "Benzene (mg/g)": benzene,
            "Worst Toxicity (-log mol/L)": worst_val if worst_val is not None else 0,
            "Most Sensitive Species": worst_species,
            "Recommendation": result.get("recommendation", "N/A"),
            "Score": result.get("final_score"),
            "Linker SMILES": (result.get("linker") or {}).get("linker_smiles", "N/A"),
        })

    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        color_map = {"recommend": "#28a745", "borderline": "#ffc107",
                     "reject": "#dc3545", "N/A": "#6c757d"}
        fig = px.scatter(
            plot_df, x="Benzene (mg/g)", y="Worst Toxicity (-log mol/L)",
            color="Recommendation", color_discrete_map=color_map,
            hover_data=["MOF ID", "Most Sensitive Species", "Linker SMILES", "Score"],
            title="Benzene Uptake vs Most Sensitive Species Toxicity",
        )
        fig.update_layout(
            xaxis_title="Benzene Uptake (mg/g)",
            yaxis_title="Most sensitive species toxicity (-log mol/L, lower = safer)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid data for Pareto plot.")

    # Per-MOF detail
    st.subheader("Per-MOF Detail")
    for result in full_results:
        mof_id = result.get("mof_id", "unknown")
        with st.expander(f"📋 {mof_id}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Adsorption**")
                ads = result.get("adsorption") or {}
                st.metric("Benzene", f"{ads.get('benzene_uptake_mg_g', 'N/A')} mg/g")
                st.metric("Toluene", f"{ads.get('toluene_uptake_mg_g', 'N/A')} mg/g")
            with col2:
                st.markdown("**Safety**")
                safety = result.get("safety") or {}
                st.metric("Metal Tier", safety.get("metal_tier", "N/A"))
                st.metric("PMT Filter", "Pass" if safety.get("pmt_pass", True) else "Fail")
            st.markdown("**Explanation**")
            st.write(result.get("explanation", "No explanation available."))
            if result.get("warnings"):
                st.markdown("**Warnings**")
                for w in result["warnings"]:
                    st.warning(w)
            if result.get("errors"):
                st.markdown("**Errors**")
                for e in result["errors"]:
                    st.error(e)


if __name__ == "__main__":
    main()
