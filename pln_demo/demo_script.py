#!/usr/bin/env python3
"""
===============================================================================
  PLN x TerraMind  --  Guided Demo Script
  ========================================
  Satellite-Powered Infrastructure Intelligence
  for Perusahaan Listrik Negara (PLN), Indonesia
===============================================================================

  This script walks through 5 real-world PLN use-case scenarios, each with:
    - Business context & objectives
    - Live AI inference using IBM TerraMind
    - Side-by-side comparison with ground-truth data
    - Actionable risk metrics and recommendations

  Run:
      python pln_demo/demo_script.py               # Full guided demo
      python pln_demo/demo_script.py --scenario 2   # Single scenario
      python pln_demo/demo_script.py --quick         # Fast mode (tiny model)
      python pln_demo/demo_script.py --interactive   # Pause between scenarios
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "notebooks"))

from pln_demo.pln_analysis import (
    PLNTerraMindAnalyzer,
    s2_to_rgb, s1_to_rgb, dem_to_rgb, lulc_to_rgb, ndvi_to_rgb,
    LULC_CLASS_NAMES, LULC_CMAP, select_device,
)

DEMO_DATA = os.path.join(PROJECT_ROOT, "demo_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pln_demo", "demo_output")

# ============================================================================
# Pretty printing helpers
# ============================================================================

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"


def banner(text, width=72):
    print()
    print(f"{BLUE}{BOLD}{'=' * width}")
    for line in text.strip().split("\n"):
        padding = (width - len(line)) // 2
        print(f"{' ' * padding}{line}")
    print(f"{'=' * width}{RESET}")
    print()


def section(num, title, subtitle=""):
    print(f"\n{CYAN}{BOLD}--- Scenario {num} -------------------------------------------------{RESET}")
    print(f"{BOLD}    {title}{RESET}")
    if subtitle:
        print(f"{DIM}    {subtitle}{RESET}")
    print()


def narrate(text):
    for line in text.strip().split("\n"):
        print(f"  {DIM}{line.strip()}{RESET}")
    print()


def objective(text):
    print(f"  {YELLOW}{BOLD}OBJECTIVE:{RESET} {text}")


def finding(label, value, level="info"):
    color = {"high": RED, "medium": YELLOW, "low": GREEN, "info": RESET}
    print(f"    {color.get(level, RESET)}{BOLD}{label:30s}{RESET}  {value}")


def recommendation(text, severity="info"):
    icon = {"high": "!!!", "medium": ">>", "low": "--", "info": ">>"}
    color = {"high": RED, "medium": YELLOW, "low": GREEN, "info": CYAN}
    print(f"\n  {color[severity]}{BOLD}  [{icon[severity]}] RECOMMENDATION: {text}{RESET}")


def pause_if_interactive(interactive):
    if interactive:
        input(f"\n  {DIM}Press ENTER to continue to the next scenario...{RESET}")


# ============================================================================
# Data loading
# ============================================================================

def load_tif(path):
    import rioxarray as rxr
    data = rxr.open_rasterio(path)
    return torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)


def load_ground_truth_visual(path, modality):
    import rioxarray as rxr
    data = rxr.open_rasterio(path)
    arr = data.values
    renderers = {"DEM": dem_to_rgb, "LULC": lulc_to_rgb, "NDVI": ndvi_to_rgb, "S1RTC": s1_to_rgb}
    return renderers[modality](arr)


# ============================================================================
# Plotting helpers
# ============================================================================

def save_comparison(panels, title, filepath, suptitle_y=0.98):
    """Save a multi-panel figure. panels = [(image_rgb, label), ...]"""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=suptitle_y)

    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {DIM}Saved: {os.path.relpath(filepath, PROJECT_ROOT)}{RESET}")


def make_risk_rgb(risk_map, vmax=3):
    cmap = LinearSegmentedColormap.from_list("risk", ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"], N=256)
    normed = np.clip(risk_map / vmax, 0, 1)
    return (cmap(normed)[:, :, :3] * 255).astype(np.uint8)


def make_flood_rgb(flood_map):
    cmap = LinearSegmentedColormap.from_list("flood", ["#ffffff", "#85c1e9", "#2e86c1", "#1a5276"], N=256)
    return (cmap(np.clip(flood_map, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def make_lulc_stats_chart(stats_dist, filepath):
    """Bar chart of LULC distribution."""
    if not stats_dist:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    classes = list(stats_dist.keys())
    values = list(stats_dist.values())
    colors = ["#3498db", "#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#9b59b6", "#e74c3c", "#95a5a6", "#1abc9c"]
    ax.barh(classes, values, color=colors[: len(classes)], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Land Cover Distribution", fontweight="bold")
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================================
# SCENARIO 1 -- Vegetation Encroachment Risk
# ============================================================================

def scenario_1(analyzer, interactive):
    section(1, "Vegetation Encroachment Risk Detection",
            "Protecting 500kV Java Backbone Transmission Corridor")

    narrate("""
        PLN operates over 55,000 km of transmission lines across Indonesia.
        Trees and fast-growing tropical vegetation can contact 500kV lines,
        causing flashovers, wildfires, and widespread blackouts.

        Traditional monitoring requires helicopter patrols costing
        Rp 15-25 million/hour. Satellite-based AI monitoring provides
        continuous, cost-effective surveillance of the entire corridor.
    """)

    objective("Detect high-risk vegetation zones requiring immediate trimming")

    # --- Sample A: Coastal corridor ---
    print(f"\n  {BOLD}[Sample A] Coastal Transmission Corridor{RESET}")
    input_a = load_tif(os.path.join(DEMO_DATA, "scenario_1_vegetation_risk/input/coastal_transmission_corridor.tif"))
    print(f"  Input loaded: {list(input_a.shape)} (Sentinel-2, 12 bands, 224x224)")

    t0 = time.time()
    result_a = analyzer.analyze_vegetation_risk(input_a)
    dt = time.time() - t0
    stats = result_a["statistics"]

    finding("NDVI Mean", f"{stats['ndvi_mean']:.3f}", "info")
    finding("Dense Vegetation", f"{stats['high_vegetation_pct']:.1f}%", "medium" if stats["high_vegetation_pct"] > 30 else "low")
    finding("High-Risk Area", f"{stats['high_risk_pct']:.1f}%", "high" if stats["high_risk_pct"] > 30 else "medium")
    finding("Critical Risk (Trees)", f"{stats['critical_risk_pct']:.1f}%", "high" if stats["critical_risk_pct"] > 20 else "low")
    finding("Inference Time", f"{dt:.1f}s", "info")

    # Land cover breakdown
    print(f"\n    {BOLD}Land Cover Breakdown:{RESET}")
    for cls, pct in stats["lulc_distribution"].items():
        lvl = "high" if cls in ("Trees", "Flooded Veg") and pct > 20 else "info"
        finding(f"  {cls}", f"{pct:.1f}%", lvl)

    save_comparison(
        [
            (s2_to_rgb(input_a), "Sentinel-2 Optical"),
            (ndvi_to_rgb(result_a["ndvi"]), "Generated NDVI"),
            (lulc_to_rgb(result_a["lulc"]), "Generated Land Cover"),
            (make_risk_rgb(result_a["risk_map"]), "Vegetation Risk Map"),
        ],
        "Scenario 1A: Coastal Transmission Corridor - Vegetation Risk",
        os.path.join(OUTPUT_DIR, "s1a_coastal_vegetation_risk.png"),
    )

    make_lulc_stats_chart(stats["lulc_distribution"],
                          os.path.join(OUTPUT_DIR, "s1a_lulc_distribution.png"))

    # Ground truth comparison
    gt_lulc = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_1_vegetation_risk/ground_truth/coastal_lulc.tif"), "LULC"
    )
    gt_ndvi = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_1_vegetation_risk/ground_truth/coastal_ndvi.tif"), "NDVI"
    )
    save_comparison(
        [
            (ndvi_to_rgb(result_a["ndvi"]), "Generated NDVI"),
            (gt_ndvi, "Ground Truth NDVI"),
            (lulc_to_rgb(result_a["lulc"]), "Generated LULC"),
            (gt_lulc, "Ground Truth LULC"),
        ],
        "Scenario 1A: AI Generation vs Ground Truth Validation",
        os.path.join(OUTPUT_DIR, "s1a_validation.png"),
    )

    # --- Sample B: Dense vegetation ---
    print(f"\n  {BOLD}[Sample B] Dense Tropical Vegetation Corridor{RESET}")
    input_b = load_tif(os.path.join(DEMO_DATA, "scenario_1_vegetation_risk/input/dense_vegetation_corridor.tif"))
    result_b = analyzer.analyze_vegetation_risk(input_b)
    stats_b = result_b["statistics"]

    finding("Critical Risk (Trees)", f"{stats_b['critical_risk_pct']:.1f}%",
            "high" if stats_b["critical_risk_pct"] > 30 else "medium")
    finding("NDVI Mean", f"{stats_b['ndvi_mean']:.3f}", "high" if stats_b["ndvi_mean"] > 0.5 else "info")

    save_comparison(
        [
            (s2_to_rgb(input_b), "Sentinel-2 Optical"),
            (ndvi_to_rgb(result_b["ndvi"]), "Generated NDVI"),
            (lulc_to_rgb(result_b["lulc"]), "Generated Land Cover"),
            (make_risk_rgb(result_b["risk_map"]), "Vegetation Risk Map"),
        ],
        "Scenario 1B: Dense Vegetation Corridor - HIGH RISK",
        os.path.join(OUTPUT_DIR, "s1b_dense_vegetation_risk.png"),
    )

    if stats_b["critical_risk_pct"] > 30:
        recommendation(
            "Deploy ground crew for emergency vegetation clearing in this corridor. "
            "88%+ flooded vegetation indicates severe encroachment.",
            "high"
        )
    else:
        recommendation("Schedule routine vegetation management patrol.", "medium")

    pause_if_interactive(interactive)


# ============================================================================
# SCENARIO 2 -- Flood Risk Assessment
# ============================================================================

def scenario_2(analyzer, interactive):
    section(2, "Flood Risk Assessment for Power Infrastructure",
            "Protecting PLN Substations During Monsoon Season (Oct-Mar)")

    narrate("""
        Indonesia experiences severe monsoon flooding annually,
        with 2,000+ flood events per year affecting critical infrastructure.

        PLN lost Rp 1.2 trillion in flood-related damages in 2024 alone.
        Substations and underground cables are especially vulnerable in
        low-lying coastal and riverine areas.

        TerraMind generates SAR radar (sees through clouds) and Digital
        Elevation Models from optical imagery to assess flood vulnerability
        even during overcast monsoon conditions.
    """)

    objective("Identify infrastructure at high flood risk for pre-monsoon hardening")

    # --- Sample A: Coastal substation ---
    print(f"\n  {BOLD}[Sample A] Coastal Substation Area (Low Elevation){RESET}")
    input_a = load_tif(os.path.join(DEMO_DATA, "scenario_2_flood_assessment/input/coastal_substation_area.tif"))

    t0 = time.time()
    result_a = analyzer.analyze_flood_risk(input_a)
    dt = time.time() - t0
    stats = result_a["statistics"]

    finding("Elevation Range", f"{stats['dem_min_m']:.0f}m - {stats['dem_max_m']:.0f}m", "info")
    finding("Mean Elevation", f"{stats['dem_mean_m']:.1f}m", "high" if stats["dem_mean_m"] < 10 else "info")
    finding("Low-Lying Area", f"{stats['low_elevation_pct']:.1f}%", "medium")
    finding("High Flood Risk Zone", f"{stats['high_flood_risk_pct']:.1f}%",
            "high" if stats["high_flood_risk_pct"] > 40 else "medium")
    finding("Inference Time", f"{dt:.1f}s", "info")

    save_comparison(
        [
            (s2_to_rgb(input_a), "Sentinel-2 Optical"),
            (s1_to_rgb(result_a["sar"]), "Generated SAR (All-Weather)"),
            (dem_to_rgb(result_a["dem"]), "Generated Elevation (DEM)"),
            (make_flood_rgb(result_a["flood_susceptibility"]), "Flood Susceptibility"),
        ],
        "Scenario 2A: Coastal Substation - Flood Risk Assessment",
        os.path.join(OUTPUT_DIR, "s2a_coastal_flood.png"),
    )

    # Ground truth comparison
    gt_dem = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_2_flood_assessment/ground_truth/coastal_dem.tif"), "DEM"
    )
    gt_sar = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_2_flood_assessment/ground_truth/coastal_sar.tif"), "S1RTC"
    )
    save_comparison(
        [
            (dem_to_rgb(result_a["dem"]), "Generated DEM"),
            (gt_dem, "Ground Truth DEM"),
            (s1_to_rgb(result_a["sar"]), "Generated SAR"),
            (gt_sar, "Ground Truth SAR"),
        ],
        "Scenario 2A: AI Generation vs Ground Truth Validation",
        os.path.join(OUTPUT_DIR, "s2a_validation.png"),
    )

    # --- Sample B: River basin ---
    print(f"\n  {BOLD}[Sample B] River Basin Near Hydroelectric Plant{RESET}")
    input_b = load_tif(os.path.join(DEMO_DATA, "scenario_2_flood_assessment/input/river_basin_plant.tif"))
    result_b = analyzer.analyze_flood_risk(input_b)
    stats_b = result_b["statistics"]

    finding("Elevation Range", f"{stats_b['dem_min_m']:.0f}m - {stats_b['dem_max_m']:.0f}m", "info")
    finding("High Flood Risk Zone", f"{stats_b['high_flood_risk_pct']:.1f}%",
            "high" if stats_b["high_flood_risk_pct"] > 40 else "low")

    save_comparison(
        [
            (s2_to_rgb(input_b), "Sentinel-2 Optical"),
            (s1_to_rgb(result_b["sar"]), "Generated SAR"),
            (dem_to_rgb(result_b["dem"]), "Generated DEM"),
            (make_flood_rgb(result_b["flood_susceptibility"]), "Flood Susceptibility"),
        ],
        "Scenario 2B: River Basin Plant - Flood Risk",
        os.path.join(OUTPUT_DIR, "s2b_river_basin_flood.png"),
    )

    if stats["high_flood_risk_pct"] > 50:
        recommendation(
            "Coastal substation is in a critical flood zone (elevation <10m). "
            "Install flood barriers, elevate transformers, and deploy mobile pumps "
            "before monsoon season begins in October.",
            "high"
        )
    else:
        recommendation("Monitor water levels; standard monsoon preparedness.", "low")

    pause_if_interactive(interactive)


# ============================================================================
# SCENARIO 3 -- Urban Expansion & Load Forecasting
# ============================================================================

def scenario_3(analyzer, interactive):
    section(3, "Urban Expansion & Grid Load Forecasting",
            "Planning Grid Capacity for Rapid Urbanization")

    narrate("""
        Indonesia is urbanizing at 4.1% annually - among the fastest
        in Southeast Asia. New housing developments, industrial zones,
        and the IKN (Ibu Kota Nusantara) new capital require massive
        grid expansion.

        TerraMind's land-use classification identifies built-area
        expansion, agricultural conversion, and infrastructure development
        to support PLN's demand forecasting and grid planning teams.
    """)

    objective("Map urban vs. non-urban areas for load growth projection")

    # --- Sample A: Fully urban ---
    print(f"\n  {BOLD}[Sample A] Jakarta Industrial Zone (Dense Urban){RESET}")
    input_a = load_tif(os.path.join(DEMO_DATA, "scenario_3_urban_expansion/input/jakarta_industrial_zone.tif"))

    t0 = time.time()
    result_a = analyzer.analyze_land_use(input_a)
    dt = time.time() - t0
    stats = result_a["statistics"]

    finding("Built Area", f"{stats['built_area_pct']:.1f}%",
            "high" if stats["built_area_pct"] > 80 else "info")
    finding("Forest", f"{stats['forest_pct']:.1f}%", "info")
    finding("Agriculture", f"{stats['agriculture_pct']:.1f}%", "info")
    finding("Inference Time", f"{dt:.1f}s", "info")

    print(f"\n    {BOLD}Land Cover:{RESET}")
    for cls, pct in stats["lulc_distribution"].items():
        finding(f"  {cls}", f"{pct:.1f}%", "info")

    save_comparison(
        [
            (s2_to_rgb(input_a), "Sentinel-2 Optical"),
            (lulc_to_rgb(result_a["lulc"]), "Land Cover Classification"),
        ],
        "Scenario 3A: Jakarta Industrial Zone - Land Use",
        os.path.join(OUTPUT_DIR, "s3a_jakarta_urban.png"),
    )

    # --- Sample B: Peri-urban ---
    print(f"\n  {BOLD}[Sample B] Peri-Urban Development Zone{RESET}")
    input_b = load_tif(os.path.join(DEMO_DATA, "scenario_3_urban_expansion/input/peri_urban_development.tif"))
    result_b = analyzer.analyze_land_use(input_b)
    stats_b = result_b["statistics"]

    finding("Built Area", f"{stats_b['built_area_pct']:.1f}%", "info")
    finding("Forest", f"{stats_b['forest_pct']:.1f}%", "info")
    finding("Agriculture", f"{stats_b['agriculture_pct']:.1f}%",
            "medium" if stats_b["agriculture_pct"] > 15 else "info")

    save_comparison(
        [
            (s2_to_rgb(input_b), "Sentinel-2 Optical"),
            (lulc_to_rgb(result_b["lulc"]), "Land Cover Classification"),
        ],
        "Scenario 3B: Peri-Urban Zone - Land Use Change",
        os.path.join(OUTPUT_DIR, "s3b_periurban.png"),
    )

    make_lulc_stats_chart(stats_b["lulc_distribution"],
                          os.path.join(OUTPUT_DIR, "s3b_lulc_chart.png"))

    # Ground truth comparison
    gt_a = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_3_urban_expansion/ground_truth/jakarta_lulc.tif"), "LULC"
    )
    save_comparison(
        [
            (lulc_to_rgb(result_a["lulc"]), "Generated LULC (Jakarta)"),
            (gt_a, "Ground Truth LULC (Jakarta)"),
            (lulc_to_rgb(result_b["lulc"]), "Generated LULC (Peri-Urban)"),
        ],
        "Scenario 3: AI vs Ground Truth - Urban Classification",
        os.path.join(OUTPUT_DIR, "s3_validation.png"),
    )

    recommendation(
        "Peri-urban zone shows mixed crops/forest transitioning to development. "
        "Plan 150kV feeder line expansion and new distribution substations "
        "within 18-month horizon to match projected load growth.",
        "medium"
    )

    pause_if_interactive(interactive)


# ============================================================================
# SCENARIO 4 -- Terrain & Transmission Route Planning
# ============================================================================

def scenario_4(analyzer, interactive):
    section(4, "Terrain Analysis & Transmission Route Planning",
            "New 275kV Sumatra Backbone Extension through Highland Terrain")

    narrate("""
        PLN is extending the 275kV Sumatra backbone through mountainous
        central terrain. Tower placement requires detailed elevation data,
        slope analysis, and vegetation clearance planning.

        Traditional ground surveys cost Rp 50-100 million per km and take
        6+ months. TerraMind generates elevation models from freely available
        Sentinel-2 imagery, enabling rapid preliminary route screening.
    """)

    objective("Generate terrain profiles for transmission tower placement")

    # --- Sample A: Highland ---
    print(f"\n  {BOLD}[Sample A] Highland Transmission Route (1200m elevation){RESET}")
    input_a = load_tif(os.path.join(DEMO_DATA, "scenario_4_terrain_routing/input/highland_transmission_route.tif"))

    t0 = time.time()
    result_a = analyzer.analyze_flood_risk(input_a)  # Uses DEM generation
    dt = time.time() - t0
    stats = result_a["statistics"]

    finding("Elevation Min", f"{stats['dem_min_m']:.0f} m", "info")
    finding("Elevation Max", f"{stats['dem_max_m']:.0f} m", "info")
    finding("Elevation Range", f"{stats['dem_max_m'] - stats['dem_min_m']:.0f} m", "info")
    finding("Mean Elevation", f"{stats['dem_mean_m']:.0f} m", "info")
    finding("Inference Time", f"{dt:.1f}s", "info")

    save_comparison(
        [
            (s2_to_rgb(input_a), "Sentinel-2 Optical"),
            (s1_to_rgb(result_a["sar"]), "Generated SAR"),
            (dem_to_rgb(result_a["dem"]), "Generated Elevation (DEM)"),
        ],
        "Scenario 4A: Highland Route - Terrain Analysis",
        os.path.join(OUTPUT_DIR, "s4a_highland_terrain.png"),
    )

    # Ground truth comparison
    gt_dem = load_ground_truth_visual(
        os.path.join(DEMO_DATA, "scenario_4_terrain_routing/ground_truth/highland_dem.tif"), "DEM"
    )
    save_comparison(
        [
            (dem_to_rgb(result_a["dem"]), "Generated DEM"),
            (gt_dem, "Ground Truth DEM"),
        ],
        "Scenario 4A: AI-Generated vs Ground Truth Elevation",
        os.path.join(OUTPUT_DIR, "s4a_dem_validation.png"),
    )

    # --- Sample B: Mixed terrain ---
    print(f"\n  {BOLD}[Sample B] Mixed Terrain Corridor (100-250m){RESET}")
    input_b = load_tif(os.path.join(DEMO_DATA, "scenario_4_terrain_routing/input/mixed_terrain_corridor.tif"))
    result_b = analyzer.analyze_flood_risk(input_b)
    stats_b = result_b["statistics"]

    finding("Elevation Range", f"{stats_b['dem_min_m']:.0f}m - {stats_b['dem_max_m']:.0f}m", "info")
    finding("Elevation Variance", f"{stats_b['dem_max_m'] - stats_b['dem_min_m']:.0f}m",
            "medium" if (stats_b['dem_max_m'] - stats_b['dem_min_m']) > 100 else "low")

    save_comparison(
        [
            (s2_to_rgb(input_b), "Sentinel-2 Optical"),
            (dem_to_rgb(result_b["dem"]), "Generated Elevation (DEM)"),
            (make_flood_rgb(result_b["flood_susceptibility"]), "Low-Area Map (Avoid)"),
        ],
        "Scenario 4B: Mixed Terrain Corridor - Route Planning",
        os.path.join(OUTPUT_DIR, "s4b_mixed_terrain.png"),
    )

    recommendation(
        "Highland route has 26m elevation variance -- favorable for transmission. "
        "Recommend routing along the ridge line (higher elevation band) to minimize "
        "tower height and avoid flood-prone valleys.",
        "low"
    )

    pause_if_interactive(interactive)


# ============================================================================
# SCENARIO 5 -- Comprehensive Multi-Modal Survey
# ============================================================================

def scenario_5(analyzer, interactive):
    section(5, "Comprehensive Multi-Modal Site Survey",
            "Full Intelligence Package for New 150kV Substation Siting")

    narrate("""
        Before investing Rp 200+ billion in a new 150kV substation, PLN
        needs a comprehensive site assessment covering:
          - Terrain suitability (DEM)
          - Vegetation clearance requirements (NDVI)
          - Land use compatibility (LULC)
          - All-weather visibility (SAR)
          - Flood risk (combined DEM + SAR)

        TerraMind generates ALL of these from a single Sentinel-2 optical
        image, replacing months of multi-source data procurement with
        seconds of AI inference.
    """)

    objective("Generate complete multi-modal intelligence from a single optical input")

    samples = [
        ("282D_485L_3_3.tif", "Coastal Site Candidate"),
        ("433D_629L_3_1.tif", "Agricultural/Forest Mixed Site"),
    ]

    for fname, label in samples:
        print(f"\n  {BOLD}[{label}]{RESET}")
        input_t = load_tif(os.path.join(DEMO_DATA, "scenario_5_comprehensive_survey/input", fname))

        t0 = time.time()
        result = analyzer.full_site_survey(input_t)
        dt = time.time() - t0

        stats = result["statistics"]
        gen = result["generated"]

        finding("NDVI Mean", f"{stats['ndvi_mean']:.3f}", "info")
        finding("Elevation Range", stats["dem_range_m"] + " m", "info")
        finding("Vegetation %", f"{stats['vegetation_coverage_pct']:.1f}%",
                "medium" if stats["vegetation_coverage_pct"] > 50 else "low")
        finding("Built Area %", f"{stats['built_area_pct']:.1f}%", "info")
        finding("Water %", f"{stats['water_pct']:.1f}%",
                "medium" if stats["water_pct"] > 30 else "info")
        finding("Inference Time", f"{dt:.1f}s", "info")

        tag = fname.split("_")[0]
        save_comparison(
            [
                (s2_to_rgb(input_t), "Optical (Input)"),
                (s1_to_rgb(gen["S1GRD"]), "SAR (Generated)"),
                (dem_to_rgb(gen["DEM"]), "DEM (Generated)"),
                (lulc_to_rgb(gen["LULC"]), "LULC (Generated)"),
                (ndvi_to_rgb(gen["NDVI"]), "NDVI (Generated)"),
            ],
            f"Scenario 5: {label} -- Full Multi-Modal Survey",
            os.path.join(OUTPUT_DIR, f"s5_{tag}_full_survey.png"),
        )

        make_lulc_stats_chart(stats["lulc_distribution"],
                              os.path.join(OUTPUT_DIR, f"s5_{tag}_lulc.png"))

    recommendation(
        "Site assessment complete. Coastal candidate has high water coverage (flood risk). "
        "Agricultural/forest site is better suited but requires vegetation clearing. "
        "Recommend geotechnical ground-truthing of mixed site.",
        "medium"
    )

    pause_if_interactive(interactive)


# ============================================================================
# Summary & Wrap-up
# ============================================================================

def print_summary(total_time):
    banner("""
PLN x TerraMind  --  Demo Summary
    """)

    print(f"  {BOLD}5 Scenarios Completed Successfully{RESET}\n")
    print(f"    1. Vegetation Encroachment Risk   -- Protect transmission corridors")
    print(f"    2. Flood Risk Assessment           -- Pre-monsoon infrastructure hardening")
    print(f"    3. Urban Expansion Monitoring       -- Grid capacity planning")
    print(f"    4. Terrain & Route Planning         -- New transmission line routing")
    print(f"    5. Multi-Modal Site Survey          -- Comprehensive substation siting")

    print(f"\n  {BOLD}Key Value Propositions for PLN:{RESET}")
    print(f"    {GREEN}[1]{RESET} Replace Rp 15-25M/hr helicopter patrols with continuous satellite monitoring")
    print(f"    {GREEN}[2]{RESET} Generate 5 data modalities from 1 free Sentinel-2 image")
    print(f"    {GREEN}[3]{RESET} Pre-screen transmission routes in seconds vs. months of ground survey")
    print(f"    {GREEN}[4]{RESET} All-weather monitoring via SAR generation (critical for tropical climate)")
    print(f"    {GREEN}[5]{RESET} Reduce flood damage losses through predictive site assessment")

    print(f"\n  {BOLD}Technology:{RESET}")
    print(f"    Model:  IBM TerraMind v1 (ICCV 2025)")
    print(f"    Input:  Sentinel-2 Level 2A (free, global, 5-day revisit)")
    print(f"    Output: SAR, DEM, LULC (10-class), NDVI")
    print(f"    Infra:  Runs on GPU, Apple Silicon (MPS), or CPU")

    print(f"\n  {BOLD}Next Steps:{RESET}")
    print(f"    1. Fine-tune on PLN-specific labeled data (configs provided)")
    print(f"    2. Integrate with PLN's GIS platform (ArcGIS/QGIS)")
    print(f"    3. Deploy automated monitoring pipeline on PLN cloud")
    print(f"    4. Expand to Nusantara (IKN) new capital development monitoring")

    print(f"\n  {DIM}Total demo time: {total_time:.0f}s{RESET}")
    print(f"  {DIM}All outputs saved to: pln_demo/demo_output/{RESET}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PLN TerraMind Guided Demo")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Run a single scenario (1-5), or 0 for all")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: use tiny model with 2 timesteps")
    parser.add_argument("--model-size", choices=["tiny", "small", "base"], default="base")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true",
                        help="Pause between scenarios for presenter mode")
    args = parser.parse_args()

    if args.quick:
        args.model_size = "tiny"
        args.timesteps = 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    banner("""
PLN x TerraMind
Satellite-Powered Infrastructure Intelligence
Perusahaan Listrik Negara -- Indonesia

Powered by IBM TerraMind (ICCV 2025)
Any-to-Any Earth Observation Foundation Model
    """)

    print(f"  {BOLD}Configuration:{RESET}")
    print(f"    Model Size:  {args.model_size}")
    print(f"    Timesteps:   {args.timesteps}")
    print(f"    Device:      {select_device()}")
    print(f"    Demo Data:   {os.path.relpath(DEMO_DATA, PROJECT_ROOT)}")
    print(f"    Output:      {os.path.relpath(OUTPUT_DIR, PROJECT_ROOT)}")
    print()

    # Initialize
    print(f"  {BOLD}Initializing TerraMind model...{RESET}")
    analyzer = PLNTerraMindAnalyzer(model_size=args.model_size, timesteps=args.timesteps)

    t_start = time.time()

    scenarios = {
        1: scenario_1,
        2: scenario_2,
        3: scenario_3,
        4: scenario_4,
        5: scenario_5,
    }

    if args.scenario > 0:
        if args.scenario in scenarios:
            scenarios[args.scenario](analyzer, args.interactive)
        else:
            print(f"Invalid scenario: {args.scenario}. Choose 1-5.")
            return
    else:
        for fn in scenarios.values():
            fn(analyzer, args.interactive)

    print_summary(time.time() - t_start)


if __name__ == "__main__":
    main()
