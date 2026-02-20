"""
PLN TerraMind Dashboard - Streamlit Web Application
====================================================
Interactive web-based satellite monitoring dashboard for PLN
(Perusahaan Listrik Negara), Indonesia's national electricity company.

Run with:
    source venv/bin/activate
    streamlit run pln_demo/app.py
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io

import streamlit as st

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "notebooks"))

from pln_demo.pln_config import PLN_SITES, PLN_USE_CASES, LULC_CLASSES, LULC_RISK_FOR_TRANSMISSION
from pln_demo.pln_analysis import (
    PLNTerraMindAnalyzer,
    s2_to_rgb,
    s1_to_rgb,
    dem_to_rgb,
    lulc_to_rgb,
    ndvi_to_rgb,
    LULC_CLASS_NAMES,
    select_device,
)

# ---------------------------------------------------------------------------
# API Server Mode Support
# ---------------------------------------------------------------------------
# When TERRAMIND_API_URL is set, the dashboard calls the local API server
# instead of loading the model in-process. This decouples the UI from the
# model and allows multiple dashboards to share one model server.
#
# Set via environment variable or sidebar toggle:
#   export TERRAMIND_API_URL=http://localhost:8786
# ---------------------------------------------------------------------------
TERRAMIND_API_URL = os.environ.get("TERRAMIND_API_URL", "")


class APIAnalyzerWrapper:
    """Wraps TerraMindClient to match PLNTerraMindAnalyzer interface for app.py."""

    def __init__(self, api_url: str):
        from pln_demo.terramind_client import TerraMindClient
        self.client = TerraMindClient(api_url)
        self.api_url = api_url
        info = self.client.health()
        self.device = info.get("device", "remote")
        self.model_size = info.get("model", "api")

    def _tensor_to_geotiff_b64(self, tensor):
        """Save tensor to temp GeoTIFF and return path."""
        import tempfile, rasterio
        from rasterio.transform import from_bounds

        data = tensor.cpu().numpy()
        if data.ndim == 4:
            data = data[0]  # remove batch dim -> (C, H, W)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        with rasterio.open(
            tmp_path, "w", driver="GTiff",
            height=data.shape[1], width=data.shape[2], count=data.shape[0],
            dtype=data.dtype,
        ) as dst:
            dst.write(data)

        return tmp_path

    def generate(self, input_tensor, input_modality="S2L2A", output_modalities=None, verbose=None):
        if output_modalities is None:
            output_modalities = ["S1GRD", "DEM", "LULC"]
        tmp_path = self._tensor_to_geotiff_b64(input_tensor)
        try:
            result = self.client.generate(tmp_path, output_modalities=output_modalities, return_images=False)
        finally:
            os.unlink(tmp_path)
        # The API returns tensor metadata but not raw tensors in non-image mode
        # For full compatibility, we need the dashboard to use analyze endpoints
        return result

    def analyze_vegetation_risk(self, input_tensor):
        return self._call_analyze(input_tensor, "vegetation")

    def analyze_flood_risk(self, input_tensor):
        return self._call_analyze(input_tensor, "flood")

    def analyze_land_use(self, input_tensor):
        return self._call_analyze(input_tensor, "land_use")

    def full_site_survey(self, input_tensor):
        return self._call_analyze(input_tensor, "full_survey")

    def _call_analyze(self, input_tensor, analysis_type):
        tmp_path = self._tensor_to_geotiff_b64(input_tensor)
        try:
            result = self.client.analyze(tmp_path, analysis_type=analysis_type, return_pil=True)
        finally:
            os.unlink(tmp_path)
        return result

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PLN TerraMind Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a5276;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #566573;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2e86c1;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_session():
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "input_tensor" not in st.session_state:
        st.session_state.input_tensor = None
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "site_info" not in st.session_state:
        st.session_state.site_info = None


init_session()


# ---------------------------------------------------------------------------
# Indonesia PLN example data catalog
# ---------------------------------------------------------------------------
INDONESIA_EXAMPLES = {
    "Aceh - PLTU Nagan Raya (Coal Power Plant)": {
        "file": "aceh_nagan_raya_s2l2a.tif",
        "province": "Aceh",
        "lat": 4.1500,
        "lon": 96.2000,
        "description": "PLTU Nagan Raya 210 MW coal-fired power plant area. "
                       "Coastal transmission corridor with tropical vegetation.",
        "pln_context": "Vegetation encroachment & coastal flood risk monitoring",
        "icon": "🏭",
    },
    "Sumatra Barat - PLN Padang (Distribution Hub)": {
        "file": "sumbar_padang_s2l2a.tif",
        "province": "Sumatra Barat",
        "lat": -0.9471,
        "lon": 100.3543,
        "description": "PLN Padang main distribution hub. "
                       "Hillside terrain with dense vegetation and transmission lines.",
        "pln_context": "Terrain routing & vegetation risk on hillside corridors",
        "icon": "🔌",
    },
    "Jawa Barat - PLTA Saguling (Hydroelectric Dam)": {
        "file": "jabar_saguling_s2l2a.tif",
        "province": "Jawa Barat",
        "lat": -6.9218,
        "lon": 107.3851,
        "description": "PLTA Saguling 700 MW hydroelectric dam and reservoir. "
                       "Water level monitoring and flood risk assessment.",
        "pln_context": "Reservoir monitoring, flood risk & land use change",
        "icon": "💧",
    },
    "Banten - PLTU Suralaya (Largest Coal Plant)": {
        "file": "suralaya_banten_s2l2a.tif",
        "province": "Banten",
        "lat": -6.0203,
        "lon": 106.0464,
        "description": "PLTU Suralaya 4,025 MW — largest coal-fired power plant "
                       "in Southeast Asia. Coastal industrial complex.",
        "pln_context": "Coastal erosion, land use change & environmental monitoring",
        "icon": "🔥",
    },
    "Jawa Timur - PLTU Paiton (Base-Load Plant)": {
        "file": "paiton_east_java_s2l2a.tif",
        "province": "Jawa Timur",
        "lat": -7.7197,
        "lon": 113.5525,
        "description": "PLTU Paiton 4,710 MW base-load power plant complex. "
                       "North Java coast with transmission backbone.",
        "pln_context": "500kV backbone monitoring, coastal flood & vegetation risk",
        "icon": "⚡",
    },
}


# ---------------------------------------------------------------------------
# Helper: load example data
# ---------------------------------------------------------------------------
@st.cache_data
def get_example_files():
    examples_dir = os.path.join(PROJECT_ROOT, "examples", "S2L2A")
    if not os.path.isdir(examples_dir):
        return []
    return sorted([f for f in os.listdir(examples_dir) if f.endswith(".tif")])


@st.cache_data
def get_indonesia_examples():
    """Check which Indonesia example files are available."""
    indonesia_dir = os.path.join(PROJECT_ROOT, "demo_data", "indonesia")
    available = {}
    for name, info in INDONESIA_EXAMPLES.items():
        filepath = os.path.join(indonesia_dir, info["file"])
        if os.path.isfile(filepath):
            available[name] = info
    return available


def load_example(filename):
    import rioxarray as rxr
    filepath = os.path.join(PROJECT_ROOT, "examples", "S2L2A", filename)
    data = rxr.open_rasterio(filepath)
    return torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)


def load_indonesia_example(info):
    """Load Indonesia GeoTIFF example using rasterio (no rioxarray needed)."""
    import rasterio
    indonesia_dir = os.path.join(PROJECT_ROOT, "demo_data", "indonesia")
    filepath = os.path.join(indonesia_dir, info["file"])
    with rasterio.open(filepath) as src:
        data = src.read()  # shape: (12, 224, 224)
    return torch.tensor(data.astype(np.float32)).unsqueeze(0)  # (1, 12, 224, 224)


def tensor_to_pil(rgb_array):
    """Convert numpy RGB array to PIL Image for Streamlit display."""
    return Image.fromarray(rgb_array)


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
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
    st.sidebar.markdown("## PLN TerraMind Monitor")
    st.sidebar.markdown("---")

    # Model settings
    st.sidebar.markdown("### Model Settings")
    model_size = st.sidebar.selectbox(
        "Model Size",
        ["tiny", "small", "base"],
        index=2,
        help="Larger models give better results but are slower."
    )
    timesteps = st.sidebar.slider(
        "Diffusion Timesteps",
        min_value=1, max_value=50, value=10,
        help="More steps = better quality, slower inference."
    )
    device = select_device()
    st.sidebar.info(f"Device: **{device}**")

    # Data source
    st.sidebar.markdown("### Data Source")
    indonesia_examples = get_indonesia_examples()
    generic_examples = get_example_files()

    data_sources = ["Indonesia PLN Sites"]
    if generic_examples:
        data_sources.append("Generic Examples")
    data_sources.append("Upload GeoTIFF")

    data_source = st.sidebar.radio("Input Data", data_sources)

    input_tensor = None
    source_name = "N/A"

    if data_source == "Indonesia PLN Sites":
        if indonesia_examples:
            selected_name = st.sidebar.selectbox(
                "Select PLN Site",
                list(indonesia_examples.keys()),
                format_func=lambda x: f"{INDONESIA_EXAMPLES[x]['icon']} {x}",
            )
            info = indonesia_examples[selected_name]

            # Show site info card
            st.sidebar.markdown(f"""
<div style="background:#eaf2f8; border-radius:8px; padding:10px; margin:8px 0;
            border-left:4px solid #2e86c1; font-size:0.85rem;">
    <b>{info['icon']} {info['province']}</b><br>
    {info['description']}<br>
    <span style="color:#666;">📍 {info['lat']:.4f}, {info['lon']:.4f}</span><br>
    <span style="color:#2e86c1;"><b>PLN Focus:</b> {info['pln_context']}</span>
</div>
            """, unsafe_allow_html=True)

            if st.sidebar.button("Load Site Data", use_container_width=True, type="primary"):
                with st.spinner(f"Loading {info['province']} satellite data..."):
                    input_tensor = load_indonesia_example(info)
                    source_name = f"{info['province']} - {info['file']}"
                    st.session_state.input_tensor = input_tensor
                    st.session_state.source_name = source_name
                    st.session_state.site_info = info
                    st.session_state.results = {}
                st.sidebar.success(f"Loaded {info['province']}!")
        else:
            st.sidebar.warning(
                "Indonesia data not found. Run:\n"
                "```\npython pln_demo/download_indonesia_data.py\n```"
            )

    elif data_source == "Generic Examples" and generic_examples:
        selected = st.sidebar.selectbox("Select Example", generic_examples)
        if st.sidebar.button("Load Example", use_container_width=True):
            with st.spinner("Loading example data..."):
                input_tensor = load_example(selected)
                source_name = selected
                st.session_state.input_tensor = input_tensor
                st.session_state.source_name = source_name
                st.session_state.site_info = None
                st.session_state.results = {}

    elif data_source == "Upload GeoTIFF":
        uploaded = st.sidebar.file_uploader(
            "Upload Sentinel-2 L2A GeoTIFF (12-band)",
            type=["tif", "tiff"],
        )
        if uploaded is not None:
            import rasterio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                data = src.read()
            input_tensor = torch.tensor(data.astype(np.float32)).unsqueeze(0)
            source_name = uploaded.name
            st.session_state.input_tensor = input_tensor
            st.session_state.source_name = source_name
            st.session_state.site_info = None
            st.session_state.results = {}
            os.unlink(tmp_path)

    # Model initialization — either local or via API server
    st.sidebar.markdown("### Model Loading")
    api_url = TERRAMIND_API_URL
    use_api = st.sidebar.checkbox(
        "Use API Server",
        value=bool(api_url),
        help="Connect to a running TerraMind server instead of loading model in-process.",
    )
    if use_api:
        api_url = st.sidebar.text_input(
            "Server URL",
            value=api_url or "http://localhost:8786",
        )
        if st.sidebar.button("Connect to Server", use_container_width=True):
            with st.spinner(f"Connecting to {api_url}..."):
                try:
                    wrapper = APIAnalyzerWrapper(api_url)
                    st.session_state.analyzer = wrapper
                    st.session_state.api_mode = True
                    st.sidebar.success(f"Connected! Device: {wrapper.device}")
                except Exception as e:
                    st.sidebar.error(f"Connection failed: {e}")
    else:
        if st.sidebar.button("Initialize Model (Local)", use_container_width=True):
            with st.spinner(f"Loading terramind_v1_{model_size}_generate..."):
                st.session_state.analyzer = PLNTerraMindAnalyzer(
                    model_size=model_size,
                    timesteps=timesteps,
                )
                st.session_state.api_mode = False
            st.sidebar.success("Model loaded!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### PLN Infrastructure")
    site_filter = st.sidebar.multiselect(
        "Filter by Type",
        list(set(s["type"] for s in PLN_SITES.values())),
        default=list(set(s["type"] for s in PLN_SITES.values())),
    )

    return model_size, timesteps, site_filter


# ---------------------------------------------------------------------------
# Main Content Tabs
# ---------------------------------------------------------------------------
def render_overview():
    """Overview / Home tab."""
    st.markdown('<p class="main-header">PLN Satellite Infrastructure Monitor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'AI-powered Earth observation for Indonesia\'s electricity grid using IBM TerraMind'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    total_capacity = sum(s["capacity_mw"] for s in PLN_SITES.values())
    col1.metric("Monitored Sites", len(PLN_SITES))
    col2.metric("Total Capacity", f"{total_capacity:,} MW")
    col3.metric("Modalities", "5 (S2, S1, DEM, LULC, NDVI)")
    col4.metric("Model", "TerraMind v1")

    st.markdown("---")
    st.markdown("### PLN Infrastructure Sites")

    # Build satellite lookup for table
    _sat_sites = set()
    for _ex_name, _ex_info in INDONESIA_EXAMPLES.items():
        for _pln_name, _pln_info in PLN_SITES.items():
            if (abs(_pln_info["lat"] - _ex_info["lat"]) < 0.1
                    and abs(_pln_info["lon"] - _ex_info["lon"]) < 0.1):
                _sat_sites.add(_pln_name)
                break

    _color_dots = {
        "Coal Power Plant": "🔴",
        "Hydroelectric": "🔵",
        "Gas Power Plant": "🟠",
        "Geothermal": "🟢",
    }

    # Site table with color indicator and satellite badge
    rows = []
    for name, info in PLN_SITES.items():
        rows.append({
            "": _color_dots.get(info["type"], "⚪"),
            "Site": name,
            "Type": info["type"],
            "Capacity (MW)": info["capacity_mw"],
            "Region": info["region"],
            "Satellite Data": "🛰️" if name in _sat_sites else "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Map
    st.markdown("### Infrastructure Map")
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="OpenStreetMap")

    type_colors = {
        "Coal Power Plant": "red",
        "Hydroelectric": "blue",
        "Gas Power Plant": "orange",
        "Geothermal": "green",
    }

    # Build lookup: which PLN_SITES have satellite data available
    # Match by coordinate proximity (within ~0.05 degrees)
    satellite_lookup = {}
    for _ex_name, ex_info in INDONESIA_EXAMPLES.items():
        for pln_name, pln_info in PLN_SITES.items():
            if (abs(pln_info["lat"] - ex_info["lat"]) < 0.1
                    and abs(pln_info["lon"] - ex_info["lon"]) < 0.1):
                satellite_lookup[pln_name] = ex_info
                break

    # Single marker per site — infrastructure color, satellite badge in popup
    matched_example_keys = set()
    for name, info in PLN_SITES.items():
        color = type_colors.get(info["type"], "gray")
        has_sat = name in satellite_lookup

        popup_html = (
            f"<b>{name}</b><br>"
            f"{info['type']}<br>"
            f"{info['capacity_mw']} MW<br>"
            f"{info['description']}"
        )
        if has_sat:
            ex = satellite_lookup[name]
            popup_html += (
                f"<br><hr style='margin:4px 0;'>"
                f"<b style='color:#27ae60;'>🛰️ Satellite data available</b><br>"
                f"<i>{ex['pln_context']}</i>"
            )
            matched_example_keys.add(id(ex))

        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=popup_html,
            tooltip=f"{'🛰️ ' if has_sat else ''}{name}",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(m)

    # Add satellite-only sites not already covered by PLN_SITES
    for _ex_name, ex_info in INDONESIA_EXAMPLES.items():
        if id(ex_info) not in matched_example_keys:
            folium.Marker(
                location=[ex_info["lat"], ex_info["lon"]],
                popup=(
                    f"<b>{ex_info['icon']} {ex_info['province']}</b><br>"
                    f"{ex_info['description']}<br>"
                    f"<i>PLN: {ex_info['pln_context']}</i><br>"
                    f"<b style='color:#27ae60;'>🛰️ Satellite data available</b>"
                ),
                tooltip=f"🛰️ {ex_info['province']} (Satellite Data)",
                icon=folium.Icon(color="cadetblue", icon="satellite-dish", prefix="fa"),
            ).add_to(m)

    st_folium(m, width=None, height=500, use_container_width=True)

    # Map legend
    st.markdown("""
<div style="display:flex; flex-wrap:wrap; gap:18px; padding:10px 0 6px 0; font-size:0.88rem;">
    <span style="font-weight:600; color:#444;">Map Legend:</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#cb3234;margin-right:4px;vertical-align:middle;"></span> Coal Power Plant</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#3a7bd5;margin-right:4px;vertical-align:middle;"></span> Hydroelectric</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f0932b;margin-right:4px;vertical-align:middle;"></span> Gas Power Plant</span>
    <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#27ae60;margin-right:4px;vertical-align:middle;"></span> Geothermal</span>
    <span style="margin-left:8px;">🛰️ = Satellite data available</span>
</div>
    """, unsafe_allow_html=True)

    # Indonesia satellite data showcase
    indonesia_data = get_indonesia_examples()
    if indonesia_data:
        st.markdown("### 🛰️ Indonesia Satellite Data (Real Sentinel-2)")
        st.markdown(
            "Real Sentinel-2 L2A satellite imagery downloaded for PLN sites. "
            "Select any site from the sidebar to run AI analysis."
        )
        indo_cols = st.columns(len(indonesia_data))
        for col, (name, info) in zip(indo_cols, indonesia_data.items()):
            with col:
                # Load and show thumbnail
                try:
                    indo_tensor = load_indonesia_example(info)
                    st.image(
                        tensor_to_pil(s2_to_rgb(indo_tensor)),
                        caption=f"{info['icon']} {info['province']}",
                        use_container_width=True,
                    )
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:#566573;'>"
                        f"{info['description']}<br>"
                        f"<b>PLN Focus:</b> {info['pln_context']}<br>"
                        f"📍 {info['lat']:.4f}, {info['lon']:.4f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception:
                    st.info(f"{info['icon']} {info['province']} - data available")

        st.markdown("---")

    # Use cases
    st.markdown("### Monitoring Use Cases")
    for key, uc in PLN_USE_CASES.items():
        with st.expander(f"**{uc['name']}** (Priority: {uc['priority']})"):
            st.write(uc["description"])
            st.write(f"**Input:** {', '.join(uc['modalities_in'])}")
            st.write(f"**Output:** {', '.join(uc['modalities_out'])}")


def render_analysis():
    """Analysis tab - run TerraMind analyses."""
    st.markdown("### Satellite Analysis")

    if st.session_state.input_tensor is None:
        st.warning("Please load input data from the sidebar first.")
        return

    if st.session_state.analyzer is None:
        st.warning("Please initialize the model from the sidebar first.")
        return

    input_tensor = st.session_state.input_tensor
    analyzer = st.session_state.analyzer

    # Show input and site context
    site_info = st.session_state.get("site_info")
    if site_info:
        st.markdown(f"""
<div style="background:linear-gradient(135deg, #eaf2f8, #fef9e7); border-radius:10px;
            padding:15px; margin-bottom:15px; border-left:5px solid #2e86c1;">
    <span style="font-size:1.3rem;">{site_info['icon']}</span>
    <b style="font-size:1.1rem; color:#1a5276;"> {site_info['province']}</b>
    <span style="color:#566573;"> | {site_info['description']}</span><br>
    <span style="color:#2e86c1;">PLN Focus: {site_info['pln_context']}</span>
    <span style="color:#888; margin-left:15px;">
        📍 {site_info['lat']:.4f}, {site_info['lon']:.4f}
    </span>
</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**Input:** {st.session_state.get('source_name', 'N/A')} | "
                    f"Shape: {list(input_tensor.shape)}")

    col_input, col_space = st.columns([1, 2])
    with col_input:
        st.image(tensor_to_pil(s2_to_rgb(input_tensor)), caption="Sentinel-2 Optical Input", use_container_width=True)

    st.markdown("---")

    # Analysis selection
    analysis_type = st.selectbox(
        "Select Analysis",
        [
            "Vegetation Encroachment Detection",
            "Flood Risk Assessment",
            "Land Use Change Monitoring",
            "Full Multi-Modal Site Survey",
        ],
    )

    is_api = st.session_state.get("api_mode", False)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Initializing...")

        if analysis_type == "Vegetation Encroachment Detection":
            progress.progress(10, text="Generating NDVI and LULC...")
            result = analyzer.analyze_vegetation_risk(input_tensor)
            st.session_state.results["vegetation"] = result
            progress.progress(100, text="Complete!")

            if is_api:
                render_api_results(input_tensor, result, "vegetation")
            else:
                render_vegetation_results(input_tensor, result)

        elif analysis_type == "Flood Risk Assessment":
            progress.progress(10, text="Generating SAR and DEM...")
            result = analyzer.analyze_flood_risk(input_tensor)
            st.session_state.results["flood"] = result
            progress.progress(100, text="Complete!")

            if is_api:
                render_api_results(input_tensor, result, "flood")
            else:
                render_flood_results(input_tensor, result)

        elif analysis_type == "Land Use Change Monitoring":
            progress.progress(10, text="Generating LULC...")
            result = analyzer.analyze_land_use(input_tensor)
            st.session_state.results["land_use"] = result
            progress.progress(100, text="Complete!")

            if is_api:
                render_api_results(input_tensor, result, "land_use")
            else:
                render_land_use_results(input_tensor, result)

        elif analysis_type == "Full Multi-Modal Site Survey":
            progress.progress(10, text="Generating SAR, DEM, LULC, NDVI...")
            result = analyzer.full_site_survey(input_tensor)
            st.session_state.results["survey"] = result
            progress.progress(100, text="Complete!")

            if is_api:
                render_api_results(input_tensor, result, "full_survey")
            else:
                render_survey_results(input_tensor, result)


def render_vegetation_results(input_tensor, result):
    """Display vegetation analysis results."""
    st.markdown("#### Vegetation Encroachment Analysis")

    stats = result["statistics"]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI Mean", f"{stats['ndvi_mean']:.3f}")
    c2.metric("High Vegetation %", f"{stats['high_vegetation_pct']:.1f}%")
    c3.metric("High Risk Area", f"{stats['high_risk_pct']:.1f}%")
    c4.metric("Critical Risk Area", f"{stats['critical_risk_pct']:.1f}%")

    # Risk assessment
    crit = stats["critical_risk_pct"]
    if crit > 40:
        st.error("**HIGH RISK**: Significant vegetation encroachment detected. Immediate field inspection recommended.")
    elif crit > 20:
        st.warning("**MODERATE RISK**: Vegetation growth detected. Schedule vegetation management.")
    else:
        st.success("**LOW RISK**: Transmission corridor is relatively clear.")

    # Images
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(tensor_to_pil(s2_to_rgb(input_tensor)), caption="Optical Input", use_container_width=True)
    with col2:
        st.image(tensor_to_pil(ndvi_to_rgb(result["ndvi"])), caption="NDVI (Vegetation Index)", use_container_width=True)
    with col3:
        st.image(tensor_to_pil(lulc_to_rgb(result["lulc"])), caption="Land Cover", use_container_width=True)
    with col4:
        risk_cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "red"], N=4)
        risk_rgb = (risk_cmap(result["risk_map"] / 3)[:, :, :3] * 255).astype(np.uint8)
        st.image(tensor_to_pil(risk_rgb), caption="Risk Map", use_container_width=True)

    # LULC distribution
    st.markdown("#### Land Cover Distribution")
    dist = stats["lulc_distribution"]
    if dist:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(list(dist.keys()), list(dist.values()), color="#2e86c1")
        ax.set_xlabel("Coverage (%)")
        ax.set_title("Land Cover Classes")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_flood_results(input_tensor, result):
    """Display flood risk results."""
    st.markdown("#### Flood Risk Assessment")

    stats = result["statistics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elevation Min", f"{stats['dem_min_m']:.0f} m")
    c2.metric("Elevation Max", f"{stats['dem_max_m']:.0f} m")
    c3.metric("Low Elevation %", f"{stats['low_elevation_pct']:.1f}%")
    c4.metric("High Flood Risk %", f"{stats['high_flood_risk_pct']:.1f}%")

    risk = stats["high_flood_risk_pct"]
    if risk > 50:
        st.error("**HIGH FLOOD RISK**: Significant low-elevation areas detected. Review flood mitigation measures.")
    elif risk > 25:
        st.warning("**MODERATE FLOOD RISK**: Some areas susceptible to flooding.")
    else:
        st.success("**LOW FLOOD RISK**: Area has favorable elevation profile.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(tensor_to_pil(s2_to_rgb(input_tensor)), caption="Optical Input", use_container_width=True)
    with col2:
        st.image(tensor_to_pil(s1_to_rgb(result["sar"])), caption="SAR (S1 GRD)", use_container_width=True)
    with col3:
        st.image(tensor_to_pil(dem_to_rgb(result["dem"])), caption="Elevation (DEM)", use_container_width=True)
    with col4:
        flood_cmap = LinearSegmentedColormap.from_list("flood", ["white", "skyblue", "blue", "darkblue"], N=256)
        flood_rgb = (flood_cmap(result["flood_susceptibility"])[:, :, :3] * 255).astype(np.uint8)
        st.image(tensor_to_pil(flood_rgb), caption="Flood Susceptibility", use_container_width=True)


def render_land_use_results(input_tensor, result):
    """Display land use analysis results."""
    st.markdown("#### Land Use Change Monitoring")

    stats = result["statistics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Built Area", f"{stats['built_area_pct']:.1f}%")
    c2.metric("Forest Coverage", f"{stats['forest_pct']:.1f}%")
    c3.metric("Agriculture", f"{stats['agriculture_pct']:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.image(tensor_to_pil(s2_to_rgb(input_tensor)), caption="Optical Input", use_container_width=True)
    with col2:
        st.image(tensor_to_pil(lulc_to_rgb(result["lulc"])), caption="Land Cover Classification", use_container_width=True)

    # Distribution chart
    st.markdown("#### Land Cover Distribution")
    dist = stats["lulc_distribution"]
    if dist:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#3171AD", "#469C76", "#83CA70", "#EAE159", "#C07CB8", "#C19368", "#6FB2E4", "#F1F1F1", "#C66526"]
        bars = ax.bar(list(dist.keys()), list(dist.values()), color=colors[:len(dist)])
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Land Cover Classes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Risk table
    st.markdown("#### Transmission Line Risk by Land Cover")
    risk_data = []
    for cls_name, pct in dist.items():
        risk_level = LULC_RISK_FOR_TRANSMISSION.get(cls_name, "N/A")
        risk_data.append({"Land Cover": cls_name, "Coverage (%)": pct, "Risk Assessment": risk_level})
    st.dataframe(risk_data, use_container_width=True, hide_index=True)


def render_survey_results(input_tensor, result):
    """Display full site survey results."""
    st.markdown("#### Multi-Modal Site Survey")

    stats = result["statistics"]
    generated = result["generated"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI Mean", f"{stats['ndvi_mean']:.3f}")
    c2.metric("Elevation Range", stats["dem_range_m"] + " m")
    c3.metric("Vegetation %", f"{stats['vegetation_coverage_pct']:.1f}%")
    c4.metric("Built Area %", f"{stats['built_area_pct']:.1f}%")

    # All modalities
    cols = st.columns(5)
    panels = [
        ("Optical (S2 L2A)", s2_to_rgb(input_tensor)),
        ("SAR (S1 GRD)", s1_to_rgb(generated["S1GRD"])),
        ("Elevation (DEM)", dem_to_rgb(generated["DEM"])),
        ("Land Cover (LULC)", lulc_to_rgb(generated["LULC"])),
        ("Vegetation (NDVI)", ndvi_to_rgb(generated["NDVI"])),
    ]
    for col, (title, rgb) in zip(cols, panels):
        with col:
            st.image(tensor_to_pil(rgb), caption=title, use_container_width=True)

    # Land cover distribution
    st.markdown("#### Land Cover Distribution")
    dist = stats["lulc_distribution"]
    if dist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Bar chart
        ax1.barh(list(dist.keys()), list(dist.values()), color="#2e86c1")
        ax1.set_xlabel("Coverage (%)")
        ax1.set_title("Land Cover Classes")

        # Pie chart
        ax2.pie(
            list(dist.values()),
            labels=list(dist.keys()),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Distribution")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_api_results(input_tensor, result, analysis_type):
    """Render results returned from the TerraMind API server.

    API results contain:
      - result["statistics"]: dict of stats (same structure as local)
      - result["images"]: dict of PIL Images keyed by modality name
      - result["inference_time_s"]: float
    """
    stats = result.get("statistics", {})
    images = result.get("images", {})
    inf_time = result.get("inference_time_s", 0)

    st.markdown(f"#### Analysis Results  *(API mode, {inf_time:.1f}s)*")

    # Show statistics as metrics
    if analysis_type == "vegetation":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NDVI Mean", f"{stats.get('ndvi_mean', 0):.3f}")
        c2.metric("High Vegetation %", f"{stats.get('high_vegetation_pct', 0):.1f}%")
        c3.metric("High Risk Area", f"{stats.get('high_risk_pct', 0):.1f}%")
        c4.metric("Critical Risk Area", f"{stats.get('critical_risk_pct', 0):.1f}%")

    elif analysis_type == "flood":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elevation Min", f"{stats.get('dem_min_m', 0):.0f} m")
        c2.metric("Elevation Max", f"{stats.get('dem_max_m', 0):.0f} m")
        c3.metric("Low Elevation %", f"{stats.get('low_elevation_pct', 0):.1f}%")
        c4.metric("High Flood Risk %", f"{stats.get('high_flood_risk_pct', 0):.1f}%")

    elif analysis_type == "land_use":
        c1, c2, c3 = st.columns(3)
        c1.metric("Built Area", f"{stats.get('built_area_pct', 0):.1f}%")
        c2.metric("Forest Coverage", f"{stats.get('forest_pct', 0):.1f}%")
        c3.metric("Agriculture", f"{stats.get('agriculture_pct', 0):.1f}%")

    elif analysis_type == "full_survey":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NDVI Mean", f"{stats.get('ndvi_mean', 0):.3f}")
        c2.metric("Elevation Range", str(stats.get("dem_range_m", "N/A")) + " m")
        c3.metric("Vegetation %", f"{stats.get('vegetation_coverage_pct', 0):.1f}%")
        c4.metric("Built Area %", f"{stats.get('built_area_pct', 0):.1f}%")

    # Show images from API (already PIL Images)
    if images:
        cols = st.columns(len(images))
        for col, (name, pil_img) in zip(cols, images.items()):
            with col:
                caption = name.upper().replace("_", " ")
                st.image(pil_img, caption=caption, use_container_width=True)

    # LULC distribution chart
    lulc_dist = stats.get("lulc_distribution", {})
    if lulc_dist:
        st.markdown("#### Land Cover Distribution")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(list(lulc_dist.keys()), list(lulc_dist.values()), color="#2e86c1")
        ax.set_xlabel("Coverage (%)")
        ax.set_title("Land Cover Classes")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_generation():
    """Free-form generation tab."""
    st.markdown("### Custom Modality Generation")
    st.write("Generate any combination of Earth observation modalities from Sentinel-2 optical input.")

    if st.session_state.input_tensor is None:
        st.warning("Please load input data from the sidebar first.")
        return

    if st.session_state.analyzer is None:
        st.warning("Please initialize the model from the sidebar first.")
        return

    output_mods = st.multiselect(
        "Select Output Modalities",
        ["S1GRD", "DEM", "LULC", "NDVI"],
        default=["S1GRD", "DEM", "LULC"],
    )

    if not output_mods:
        st.info("Select at least one output modality.")
        return

    if st.button("Generate", type="primary"):
        with st.spinner(f"Generating {', '.join(output_mods)}..."):
            generated = st.session_state.analyzer.generate(
                st.session_state.input_tensor,
                output_modalities=output_mods,
            )

        cols = st.columns(len(output_mods) + 1)
        with cols[0]:
            st.image(
                tensor_to_pil(s2_to_rgb(st.session_state.input_tensor)),
                caption="Input (S2 L2A)",
                use_container_width=True,
            )
        for i, mod in enumerate(output_mods):
            renderer = {
                "S1GRD": s1_to_rgb,
                "DEM": dem_to_rgb,
                "LULC": lulc_to_rgb,
                "NDVI": ndvi_to_rgb,
            }[mod]
            with cols[i + 1]:
                st.image(
                    tensor_to_pil(renderer(generated[mod])),
                    caption=f"Generated {mod}",
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    model_size, timesteps, site_filter = render_sidebar()

    tab_overview, tab_analysis, tab_generation, tab_about = st.tabs(
        ["Overview", "Analysis", "Generation", "About"]
    )

    with tab_overview:
        render_overview()

    with tab_analysis:
        render_analysis()

    with tab_generation:
        render_generation()

    with tab_about:
        st.markdown("### About This Demo")
        st.markdown("""
**PLN TerraMind Monitor** is a demonstration of satellite-based infrastructure
monitoring for **Perusahaan Listrik Negara (PLN)**, Indonesia's national
electricity company.

#### Technology Stack
- **TerraMind** by IBM/ESA: Any-to-any generative foundation model for Earth Observation
- **TerraTorch**: Geospatial ML toolkit for model deployment and fine-tuning
- **Streamlit**: Interactive web dashboard framework

#### Supported Modalities
| Modality | Description | PLN Use Case |
|----------|-------------|--------------|
| S2 L2A | Sentinel-2 Optical (12 bands) | Visual inspection, base input |
| S1 GRD | Sentinel-1 SAR (2 bands) | All-weather monitoring, flood detection |
| DEM | Digital Elevation Model | Terrain analysis, flood risk |
| LULC | Land Use/Land Cover (10 classes) | Vegetation, urban expansion monitoring |
| NDVI | Vegetation Index | Vegetation health, encroachment detection |

#### PLN Monitoring Applications
1. **Vegetation Encroachment**: Detect trees growing into transmission corridors
2. **Flood Risk**: Assess flooding vulnerability of substations and power plants
3. **Land Use Change**: Track urbanization and deforestation near infrastructure
4. **Terrain Analysis**: Plan new transmission routes through diverse terrain
5. **Site Surveys**: Comprehensive multi-modal assessment for new projects

#### Data Requirements
- Input: Sentinel-2 Level 2A GeoTIFF (12 spectral bands, ideally 224x224 pixels)
- The model generates other modalities (SAR, DEM, LULC, NDVI) from this optical input

#### References
- [TerraMind Paper (ICCV 2025)](https://arxiv.org/abs/2504.11171)
- [TerraMind GitHub](https://github.com/IBM/terramind)
- [TerraMind Models on HuggingFace](https://huggingface.co/ibm-esa-geospatial)
        """)


if __name__ == "__main__":
    main()
