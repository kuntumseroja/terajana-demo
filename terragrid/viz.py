"""
TerraGrid Visualization Helpers
================================
IRI-specific rendering and plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# IRI heatmap: Green (safe) -> Yellow (moderate) -> Orange (high) -> Red (critical)
IRI_COLORS = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8b0000"]
IRI_CMAP = LinearSegmentedColormap.from_list("iri", IRI_COLORS, N=256)

# Component colormaps
COMPONENT_CMAPS = {
    "vegetation_threat": LinearSegmentedColormap.from_list("veg", ["#eafaf1", "#27ae60", "#145a32"], N=256),
    "flood_risk": LinearSegmentedColormap.from_list("flood", ["#eaf2f8", "#3498db", "#1a5276"], N=256),
    "terrain_instability": LinearSegmentedColormap.from_list("terrain", ["#fef9e7", "#f39c12", "#7d6608"], N=256),
    "encroachment_pressure": LinearSegmentedColormap.from_list("encr", ["#f5eef8", "#8e44ad", "#4a235a"], N=256),
}

COMPONENT_LABELS = {
    "vegetation_threat": "Vegetation Threat",
    "flood_risk": "Flood Risk",
    "terrain_instability": "Terrain Instability",
    "encroachment_pressure": "Encroachment Pressure",
}


def iri_to_rgb(iri_map: np.ndarray) -> np.ndarray:
    """Convert IRI score map (0-100) to RGB image using green-to-red gradient."""
    normalized = np.clip(iri_map / 100.0, 0, 1)
    rgb = IRI_CMAP(normalized)[:, :, :3]
    return (rgb * 255).round().clip(0, 255).astype(np.uint8)


def component_to_rgb(component_map: np.ndarray, component_name: str) -> np.ndarray:
    """Convert a single IRI component map to its themed RGB image."""
    cmap = COMPONENT_CMAPS.get(component_name, IRI_CMAP)
    normalized = np.clip(component_map / 100.0, 0, 1)
    rgb = cmap(normalized)[:, :, :3]
    return (rgb * 255).round().clip(0, 255).astype(np.uint8)


def plot_iri_components(components: dict, site_name: str = "PLN Site") -> plt.Figure:
    """
    Create a stacked horizontal bar chart showing the contribution of each
    IRI component to the overall score.
    """
    names = []
    means = []
    colors = ["#27ae60", "#3498db", "#f39c12", "#8e44ad"]

    for comp_name in ["vegetation_threat", "flood_risk", "terrain_instability", "encroachment_pressure"]:
        if comp_name in components:
            names.append(COMPONENT_LABELS[comp_name])
            means.append(float(np.mean(components[comp_name])))

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(names, means, color=colors[:len(names)], edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean Score (0-100)")
    ax.set_title(f"IRI Component Breakdown: {site_name}", fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def plot_iri_summary(iri_result: dict, site_name: str = "PLN Site") -> plt.Figure:
    """
    Create a multi-panel IRI summary figure with heatmap + components.
    """
    components = iri_result["components"]
    iri_map = iri_result["iri_map"]
    stats = iri_result["statistics"]

    n_panels = 1 + len(components)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"TerraGrid IRI: {site_name}  |  Score: {stats['iri_mean']:.1f}/100  [{stats['risk_category']}]",
        fontsize=14, fontweight="bold",
    )

    # IRI composite
    im = axes[0].imshow(iri_map, cmap=IRI_CMAP, vmin=0, vmax=100)
    axes[0].set_title(f"IRI (mean={stats['iri_mean']:.1f})")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], label="Risk Score", shrink=0.8)

    # Individual components
    for i, (comp_name, comp_map) in enumerate(components.items(), start=1):
        cmap = COMPONENT_CMAPS.get(comp_name, IRI_CMAP)
        im = axes[i].imshow(comp_map, cmap=cmap, vmin=0, vmax=100)
        label = COMPONENT_LABELS.get(comp_name, comp_name)
        mean_val = float(np.mean(comp_map))
        axes[i].set_title(f"{label}\n(mean={mean_val:.1f})")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    plt.tight_layout()
    return fig


def plot_ablation_comparison(ablation_results: list) -> plt.Figure:
    """
    Create a comparison chart of IRI scores across different modality ablation configs.

    Args:
        ablation_results: list of dicts with keys 'config_name', 'iri_mean', 'modalities'
    """
    names = [r["config_name"] for r in ablation_results]
    scores = [r["iri_mean"] for r in ablation_results]
    n_mods = [len(r.get("modalities", [])) for r in ablation_results]

    # Color by number of modalities (more = darker)
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(names)))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean IRI Score (0-100)")
    ax.set_title("TiM Ablation Study: IRI with Different Modality Combinations", fontweight="bold")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")

    # Labels on bars
    for bar, score, nm in zip(bars, scores, n_mods):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score:.1f}\n({nm} mod.)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
