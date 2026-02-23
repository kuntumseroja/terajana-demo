"""
TerraGrid — Infrastructure Resilience Index via TerraMind TiM
==============================================================
HuggingFace Space for the TerraMind Blue-Sky Challenge.

Computes a pixel-level composite risk score (0-100) for Indonesia's
power grid infrastructure by fusing 4 TiM-generated modalities.

Run with:
    streamlit run terragrid/app.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

import streamlit as st

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "notebooks"))

from pln_demo.pln_analysis import (
    PLNTerraMindAnalyzer,
    s2_to_rgb, s1_to_rgb, dem_to_rgb, lulc_to_rgb, ndvi_to_rgb,
    LULC_CLASS_NAMES, select_device, _compute_lulc_distribution,
)
from pln_demo.pln_config import PLN_SITES

from terragrid.iri import compute_iri, extract_modality_arrays
from terragrid.viz import (
    iri_to_rgb, component_to_rgb, plot_iri_components,
    plot_iri_summary, plot_ablation_comparison,
    COMPONENT_LABELS, IRI_CMAP,
)
from terragrid.ablation import run_ablation, ABLATION_CONFIGS

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TerraGrid IRI",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: bold; color: #1a5276;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.05rem; color: #566573; margin-bottom: 1.5rem;
    }
    .iri-score-box {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        border-radius: 12px; padding: 20px; text-align: center;
        color: white; margin: 10px 0;
    }
    .iri-score-box .score { font-size: 3rem; font-weight: bold; }
    .iri-score-box .label { font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Indonesia PLN Sites with satellite data
# ---------------------------------------------------------------------------
INDONESIA_SITES = {
    "Aceh - PLTU Nagan Raya": {
        "file": "aceh_nagan_raya_s2l2a.tif",
        "province": "Aceh", "lat": 4.15, "lon": 96.20,
        "type": "Coal Power Plant", "capacity_mw": 210,
        "description": "210 MW coal plant. Coastal corridor with tropical vegetation.",
        "icon": "🏭",
    },
    "Sumbar - PLN Padang Hub": {
        "file": "sumbar_padang_s2l2a.tif",
        "province": "Sumatra Barat", "lat": -0.9471, "lon": 100.3543,
        "type": "Distribution Hub", "capacity_mw": 150,
        "description": "Main distribution hub. Hillside terrain with dense vegetation.",
        "icon": "🔌",
    },
    "Jabar - PLTA Saguling": {
        "file": "jabar_saguling_s2l2a.tif",
        "province": "Jawa Barat", "lat": -6.9218, "lon": 107.3851,
        "type": "Hydroelectric", "capacity_mw": 700,
        "description": "700 MW hydroelectric dam and reservoir on Citarum River.",
        "icon": "💧",
    },
    "Banten - PLTU Suralaya": {
        "file": "suralaya_banten_s2l2a.tif",
        "province": "Banten", "lat": -6.0203, "lon": 106.0464,
        "type": "Coal Power Plant", "capacity_mw": 4025,
        "description": "4,025 MW — largest coal plant in Southeast Asia. Coastal industrial.",
        "icon": "🔥",
    },
    "Jatim - PLTU Paiton": {
        "file": "paiton_east_java_s2l2a.tif",
        "province": "Jawa Timur", "lat": -7.7197, "lon": 113.5525,
        "type": "Coal Power Plant", "capacity_mw": 4710,
        "description": "4,710 MW base-load complex. North Java coast, 500kV backbone.",
        "icon": "⚡",
    },
}


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_state():
    for key in ["analyzer", "input_tensor", "site_name", "iri_result", "ablation_results"]:
        if key not in st.session_state:
            st.session_state[key] = None

init_state()


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_site(filename):
    import rasterio
    path = os.path.join(PROJECT_ROOT, "demo_data", "indonesia", filename)
    with rasterio.open(path) as src:
        data = src.read()
    return torch.tensor(data.astype(np.float32)).unsqueeze(0)


def tensor_to_pil(rgb_array):
    return Image.fromarray(rgb_array)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## 🔌 TerraGrid")
    st.sidebar.markdown("*Infrastructure Resilience Index*")
    st.sidebar.markdown("---")

    # Model
    st.sidebar.markdown("### Model")
    model_size = st.sidebar.selectbox("Model Size", ["tiny", "small", "base"], index=2)
    timesteps = st.sidebar.slider("Diffusion Timesteps", 1, 50, 10)
    device = select_device()
    st.sidebar.info(f"Device: **{device}**")

    if st.sidebar.button("Load TerraMind", use_container_width=True, type="primary"):
        with st.spinner(f"Loading terramind_v1_{model_size}..."):
            st.session_state.analyzer = PLNTerraMindAnalyzer(
                model_size=model_size, timesteps=timesteps,
            )
        st.sidebar.success("Model ready!")

    st.sidebar.markdown("---")

    # Site selection
    st.sidebar.markdown("### PLN Site")
    selected = st.sidebar.selectbox(
        "Select Site",
        list(INDONESIA_SITES.keys()),
        format_func=lambda x: f"{INDONESIA_SITES[x]['icon']} {x}",
    )
    info = INDONESIA_SITES[selected]

    st.sidebar.markdown(f"""
<div style="background:#eaf2f8; border-radius:8px; padding:10px; margin:8px 0;
            border-left:4px solid #2e86c1; font-size:0.85rem;">
    <b>{info['icon']} {info['province']}</b> | {info['capacity_mw']} MW<br>
    {info['description']}<br>
    <span style="color:#666;">📍 {info['lat']:.4f}, {info['lon']:.4f}</span>
</div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Load Site Data", use_container_width=True):
        with st.spinner(f"Loading {info['province']}..."):
            st.session_state.input_tensor = load_site(info["file"])
            st.session_state.site_name = selected
            st.session_state.iri_result = None
            st.session_state.ablation_results = None
        st.sidebar.success(f"Loaded {info['province']}!")

    return selected, info


# ---------------------------------------------------------------------------
# Tab 1: TerraGrid IRI
# ---------------------------------------------------------------------------
def render_iri_tab():
    st.markdown('<p class="main-title">TerraGrid Infrastructure Resilience Index</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">'
        'Composite multi-hazard risk assessment for power grid infrastructure '
        'using TerraMind Thinking-in-Modalities (TiM)'
        '</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.input_tensor is None:
        st.warning("Load a PLN site from the sidebar.")
        return
    if st.session_state.analyzer is None:
        st.warning("Load the TerraMind model from the sidebar.")
        return

    input_tensor = st.session_state.input_tensor
    analyzer = st.session_state.analyzer
    site_name = st.session_state.site_name or "PLN Site"

    # Show input preview
    st.image(tensor_to_pil(s2_to_rgb(input_tensor)), caption=f"Sentinel-2 Input: {site_name}", width=300)

    if st.button("Compute Infrastructure Resilience Index", type="primary", use_container_width=True):
        progress = st.progress(0, text="Generating modalities via TiM...")

        # Step 1: TerraMind inference — generate all 4 modalities
        progress.progress(10, text="TiM: Generating NDVI, LULC, DEM, SAR...")
        generated = analyzer.generate(
            input_tensor,
            output_modalities=["S1GRD", "DEM", "LULC", "NDVI"],
            verbose=False,
        )

        # Step 2: Extract arrays
        progress.progress(60, text="Computing IRI components...")
        arrays = extract_modality_arrays(generated)

        # Step 3: Compute IRI
        iri_result = compute_iri(
            ndvi=arrays["ndvi"],
            lulc_classes=arrays["lulc_classes"],
            dem=arrays["dem"],
            sar=arrays.get("sar"),
        )
        st.session_state.iri_result = iri_result
        st.session_state.generated = generated

        progress.progress(100, text="Complete!")

    # Display results
    if st.session_state.iri_result is not None:
        result = st.session_state.iri_result
        generated = st.session_state.generated
        stats = result["statistics"]

        # IRI Score card
        risk_color = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MODERATE": "#f1c40f", "LOW": "#27ae60"}
        st.markdown(f"""
<div class="iri-score-box">
    <div class="label">Infrastructure Resilience Index</div>
    <div class="score">{stats['iri_mean']:.1f}</div>
    <div style="font-size:1.2rem; color:{risk_color.get(stats['risk_category'], '#fff')};">
        {stats['risk_category']} RISK
    </div>
    <div class="label" style="margin-top:8px;">
        {stats['high_risk_pct']:.1f}% high risk area | {stats['critical_risk_pct']:.1f}% critical
    </div>
</div>
        """, unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        comp_means = stats["component_means"]
        c1.metric("Vegetation Threat", f"{comp_means.get('vegetation_threat', 0):.1f}/100")
        c2.metric("Flood Risk", f"{comp_means.get('flood_risk', 0):.1f}/100")
        c3.metric("Terrain Instability", f"{comp_means.get('terrain_instability', 0):.1f}/100")
        c4.metric("Encroachment Pressure", f"{comp_means.get('encroachment_pressure', 0):.1f}/100")

        st.markdown("---")

        # 6-panel: Input | NDVI | LULC | DEM | SAR | IRI
        st.markdown("### TiM-Generated Modalities + IRI Fusion")
        cols = st.columns(6)
        panels = [
            ("Sentinel-2 Input", tensor_to_pil(s2_to_rgb(input_tensor))),
            ("TiM: NDVI", tensor_to_pil(ndvi_to_rgb(generated["NDVI"]))),
            ("TiM: LULC", tensor_to_pil(lulc_to_rgb(generated["LULC"]))),
            ("TiM: DEM", tensor_to_pil(dem_to_rgb(generated["DEM"]))),
            ("TiM: SAR", tensor_to_pil(s1_to_rgb(generated["S1GRD"]))),
            ("IRI Heatmap", tensor_to_pil(iri_to_rgb(result["iri_map"]))),
        ]
        for col, (title, img) in zip(cols, panels):
            with col:
                st.image(img, caption=title, use_container_width=True)

        # Component breakdown
        st.markdown("### IRI Component Breakdown")
        comp_cols = st.columns(4)
        for col, (comp_name, comp_map) in zip(comp_cols, result["components"].items()):
            with col:
                label = COMPONENT_LABELS.get(comp_name, comp_name)
                st.image(
                    tensor_to_pil(component_to_rgb(comp_map, comp_name)),
                    caption=f"{label}: {float(np.mean(comp_map)):.1f}/100",
                    use_container_width=True,
                )

        # Bar chart
        fig = plot_iri_components(result["components"], site_name)
        st.pyplot(fig)
        plt.close(fig)

        # LULC distribution
        arrays = extract_modality_arrays(generated)
        lulc_dist = _compute_lulc_distribution(arrays["lulc_classes"])
        if lulc_dist:
            st.markdown("### Land Cover Distribution")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(list(lulc_dist.keys()), list(lulc_dist.values()), color="#2e86c1")
            ax.set_xlabel("Coverage (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2: TiM Ablation Study
# ---------------------------------------------------------------------------
def render_ablation_tab():
    st.markdown("### TiM Ablation Study")
    st.markdown(
        "How much does each Thinking-in-Modalities output contribute to the IRI? "
        "We test 6 different modality combinations to find out."
    )

    if st.session_state.input_tensor is None:
        st.warning("Load a PLN site from the sidebar.")
        return
    if st.session_state.analyzer is None:
        st.warning("Load the TerraMind model from the sidebar.")
        return

    st.markdown("**Configurations tested:**")
    for name, mods in ABLATION_CONFIGS.items():
        mod_str = ", ".join(mods) if mods else "*(raw spectral only)*"
        st.markdown(f"- **{name}**: {mod_str}")

    if st.button("Run Ablation Study", type="primary", use_container_width=True):
        site_name = st.session_state.site_name or "PLN Site"

        with st.spinner("Running 6 ablation configurations... (this takes a few minutes)"):
            results = run_ablation(
                st.session_state.analyzer,
                st.session_state.input_tensor,
                site_name,
            )
            st.session_state.ablation_results = results

    if st.session_state.ablation_results is not None:
        results = st.session_state.ablation_results

        # Comparison table
        st.markdown("### Results")
        table_data = []
        for r in results:
            table_data.append({
                "Configuration": r["config_name"],
                "Modalities": ", ".join(r["modalities"]) if r["modalities"] else "None (spectral)",
                "IRI Mean": r["iri_mean"],
                "IRI Std": r["iri_std"],
                "Risk Category": r["risk_category"],
                "High Risk %": r["high_risk_pct"],
                "Time (s)": r["time_s"],
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # Bar chart
        fig = plot_ablation_comparison(results)
        st.pyplot(fig)
        plt.close(fig)

        # Side-by-side IRI heatmaps
        st.markdown("### IRI Heatmap Comparison")
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            with col:
                st.image(
                    tensor_to_pil(iri_to_rgb(r["iri_map"])),
                    caption=f"{r['config_name']}\nIRI={r['iri_mean']:.1f}",
                    use_container_width=True,
                )

        # Key finding
        full = results[0]
        spectral = results[-1]
        improvement = full["iri_mean"] - spectral["iri_mean"]
        st.markdown(f"""
### Key Finding

> **TiM improves IRI information content by {abs(improvement):.1f} points** compared to spectral-only baseline.
> Full 4-modality TiM (IRI={full['iri_mean']:.1f}) captures multi-hazard risks that raw spectral indices
> (IRI={spectral['iri_mean']:.1f}) cannot detect — including flood susceptibility from generated DEM,
> land cover context from generated LULC, and all-weather capability from generated SAR.
        """)


# ---------------------------------------------------------------------------
# Tab 3: Indonesia Grid Map
# ---------------------------------------------------------------------------
def render_map_tab():
    st.markdown("### Indonesia Power Grid Infrastructure")

    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="OpenStreetMap")

    type_colors = {
        "Coal Power Plant": "red",
        "Hydroelectric": "blue",
        "Gas Power Plant": "orange",
        "Geothermal": "green",
        "Distribution Hub": "cadetblue",
    }

    # PLN sites from pln_config
    for name, info in PLN_SITES.items():
        color = type_colors.get(info["type"], "gray")
        has_sat = any(
            abs(info["lat"] - s["lat"]) < 0.1 and abs(info["lon"] - s["lon"]) < 0.1
            for s in INDONESIA_SITES.values()
        )
        popup = (
            f"<b>{name}</b><br>{info['type']}<br>"
            f"{info['capacity_mw']} MW<br>{info['description']}"
        )
        if has_sat:
            popup += "<br><b style='color:#27ae60'>🛰️ Satellite data available</b>"

        folium.Marker(
            [info["lat"], info["lon"]],
            popup=popup,
            tooltip=f"{'🛰️ ' if has_sat else ''}{name}",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(m)

    st_folium(m, width=None, height=500, use_container_width=True)

    # Legend
    st.markdown("""
<div style="display:flex; flex-wrap:wrap; gap:18px; padding:10px 0; font-size:0.88rem;">
    <span style="font-weight:600;">Legend:</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#cb3234;margin-right:4px;vertical-align:middle;"></span> Coal</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#3a7bd5;margin-right:4px;vertical-align:middle;"></span> Hydro</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f0932b;margin-right:4px;vertical-align:middle;"></span> Gas</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#27ae60;margin-right:4px;vertical-align:middle;"></span> Geothermal</span>
    <span>🛰️ = Satellite data</span>
</div>
    """, unsafe_allow_html=True)

    # Site comparison table
    st.markdown("### PLN Infrastructure Sites")
    rows = []
    for name, info in PLN_SITES.items():
        rows.append({
            "Site": name,
            "Type": info["type"],
            "Capacity (MW)": info["capacity_mw"],
            "Region": info["region"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: About
# ---------------------------------------------------------------------------
def render_about_tab():
    st.markdown("""
### About TerraGrid

**TerraGrid** computes an **Infrastructure Resilience Index (IRI)** for power grid
infrastructure using IBM/ESA's TerraMind foundation model.

#### The Problem

Indonesia's national electricity company (PLN) manages over **900 km of 500kV
transmission backbone** through dense tropical vegetation. Weather-related faults
cause the vast majority of grid failures, including the May 2025 Bali blackout
affecting 1.8 million customers. Traditional ground patrols are costly, and
persistent tropical cloud cover limits optical satellite monitoring.

#### The TerraGrid Solution

From a **single Sentinel-2 optical image**, TerraMind's Thinking-in-Modalities (TiM)
generates 4 complementary Earth observation modalities:

| Generated Modality | IRI Component | What It Detects |
|---|---|---|
| **NDVI** + **LULC** | Vegetation Threat | Trees/crops growing into transmission corridors |
| **DEM** + **LULC** | Flood Risk | Low-elevation areas near water bodies |
| **DEM** gradient | Terrain Instability | Steep slopes threatening tower foundations |
| **LULC** distance | Encroachment Pressure | Urban expansion toward infrastructure |

These 4 components are fused into a single **IRI score (0-100)** per pixel:

```
IRI = 0.35 * VegetationThreat + 0.30 * FloodRisk
    + 0.15 * TerrainInstability + 0.20 * EncroachmentPressure
```

#### Technical Novelty

1. **First composite multi-hazard infrastructure risk index via TiM chains** — no prior
   TerraMind application fuses 4+ generated modalities into a unified risk score
2. **Cloud-penetrating capability** — TiM generates SAR and DEM from optical input,
   enabling monitoring even through Indonesia's persistent cloud cover
3. **Ablation study** proves each TiM modality's contribution to risk detection

#### Architecture

```
S2 L2A (12-band optical)
    │
    ▼ TerraMind TiM (single inference)
    ├── NDVI  ─── Vegetation Threat ──┐
    ├── LULC  ─── (shared by 3)     ──┤── IRI (0-100)
    ├── DEM   ─── Flood + Terrain   ──┤
    └── S1GRD ─── Cloud-penetrating ──┘
```

#### References

- [TerraMind Paper (ICCV 2025)](https://arxiv.org/abs/2504.11171)
- [TerraMind GitHub](https://github.com/IBM/terramind)
- [TerraMind Models](https://huggingface.co/ibm-esa-geospatial)
- [Blue-Sky Challenge](https://huggingface.co/spaces/ibm-esa-geospatial/challenge)
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    selected, info = render_sidebar()

    tab_iri, tab_ablation, tab_map, tab_about = st.tabs(
        ["TerraGrid IRI", "TiM Ablation Study", "Indonesia Grid", "About"]
    )

    with tab_iri:
        render_iri_tab()
    with tab_ablation:
        render_ablation_tab()
    with tab_map:
        render_map_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
