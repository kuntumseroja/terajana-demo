"""
PLN TerraMind Analysis Engine
Core analysis functions for satellite-based electricity infrastructure monitoring.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, LinearSegmentedColormap

# Add parent dir for plotting utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from terratorch.registry import FULL_MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Visualization helpers (inline, avoids import issues)
# ---------------------------------------------------------------------------
COLORBLIND_HEX = [
    "#000000", "#3171AD", "#469C76", "#83CA70", "#EAE159",
    "#C07CB8", "#C19368", "#6FB2E4", "#F1F1F1", "#C66526",
]
COLORBLIND_RGB = [hex2color(h) for h in COLORBLIND_HEX]
LULC_CMAP = LinearSegmentedColormap.from_list("lulc", COLORBLIND_RGB, N=10)

LULC_CLASS_NAMES = [
    "No Data", "Water", "Trees", "Grass", "Flooded Veg",
    "Crops", "Scrub", "Built Area", "Bare Ground", "Snow/Ice",
]


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Visualization converters
# ---------------------------------------------------------------------------

def s2_to_rgb(data, gamma=0.7):
    """Convert Sentinel-2 L2A tensor to displayable RGB."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if data.ndim == 4:
        data = data[0]
    rgb = data[[3, 2, 1]].transpose((1, 2, 0))
    lo, hi = np.quantile(rgb, [0.0, 0.99])
    lo = max(lo, 0)
    hi = max(hi, 2000)
    rgb = (rgb - lo) / (hi - lo + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    if gamma:
        rgb = np.power(rgb, gamma)
    return (rgb * 255).round().astype(np.uint8)


def s1_to_rgb(data):
    """Convert Sentinel-1 GRD tensor to displayable RGB."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if data.ndim == 4:
        data = data[0]
    vv, vh = data[0], data[1]
    r = (vv + 30) / 40
    g = (vh + 40) / 40
    b = vv / np.clip(vh, -40, -1) / 1.5
    rgb = np.dstack([r, g, b])
    return np.clip(rgb * 255, 0, 255).round().astype(np.uint8)


def dem_to_rgb(data, cmap="BrBG_r"):
    """Convert DEM tensor to colorized RGB."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    while data.ndim > 2:
        data = data[0]
    lo, hi = data.min() - 5, data.max() + 5
    data = (data - lo) / (hi - lo + 1e-6)
    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    return (rgb * 255).round().clip(0, 255).astype(np.uint8)


def ndvi_to_rgb(data, cmap="RdYlGn"):
    """Convert NDVI tensor to colorized RGB."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    while data.ndim > 2:
        data = data[0]
    data = (data + 1) / 2
    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    return (rgb * 255).round().clip(0, 255).astype(np.uint8)


def lulc_to_rgb(data, num_classes=10):
    """Convert LULC tensor to colorized RGB."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    while data.ndim > 2:
        if data.shape[0] == num_classes:
            data = data.argmax(axis=0)
        else:
            data = data[0]
    rgb = LULC_CMAP(data / (num_classes - 1))[:, :, :3]
    return (rgb * 255).round().clip(0, 255).astype(np.uint8)


MODALITY_RENDERERS = {
    "S2L2A": s2_to_rgb,
    "S1GRD": s1_to_rgb,
    "DEM": dem_to_rgb,
    "LULC": lulc_to_rgb,
    "NDVI": ndvi_to_rgb,
}


# ---------------------------------------------------------------------------
# TerraMind model wrapper for PLN use cases
# ---------------------------------------------------------------------------

# Classes that are climatically impossible in tropical Indonesia.
# ESA WorldCover class 9 = Snow/Ice, class 0 = No Data (often model noise).
# These are excluded from LULC distribution charts and statistics.
TROPICAL_EXCLUDE_CLASSES = {0, 9}  # No Data, Snow/Ice


def _compute_lulc_distribution(lulc_classes, exclude_classes=TROPICAL_EXCLUDE_CLASSES):
    """Compute LULC % distribution, excluding climatically impossible classes.

    Pixels belonging to excluded classes are removed from the total count so
    that the remaining percentages still sum to ~100%.
    """
    total = lulc_classes.size
    excluded_pixels = sum(int(np.sum(lulc_classes == c)) for c in exclude_classes)
    effective_total = total - excluded_pixels
    if effective_total <= 0:
        effective_total = total  # fallback

    dist = {}
    for cls_id in range(10):
        if cls_id in exclude_classes:
            continue
        count = int(np.sum(lulc_classes == cls_id))
        if count > 0:
            dist[LULC_CLASS_NAMES[cls_id]] = round(count / effective_total * 100, 1)
    return dist


class PLNTerraMindAnalyzer:
    """Wraps TerraMind generation models for PLN infrastructure analysis."""

    def __init__(self, model_size="base", device=None, timesteps=10):
        self.device = device or select_device()
        self.model_size = model_size
        self.timesteps = timesteps
        self._models = {}
        print(f"[PLN Analyzer] Device: {self.device} | Model: terramind_v1_{model_size}_generate")

    def _get_model(self, input_modalities, output_modalities):
        """Build or retrieve a cached model for the given modality combination."""
        key = (tuple(input_modalities), tuple(output_modalities))
        if key not in self._models:
            print(f"[PLN Analyzer] Building model: {input_modalities} -> {output_modalities}")
            model = FULL_MODEL_REGISTRY.build(
                f"terramind_v1_{self.model_size}_generate",
                modalities=input_modalities,
                output_modalities=output_modalities,
                pretrained=True,
                standardize=True,
            )
            model = model.to(self.device)
            model.eval()
            self._models[key] = model
        return self._models[key]

    def generate(self, input_tensor, input_modality="S2L2A", output_modalities=None,
                 verbose=None):
        """
        Run generation from input modality to output modalities.

        Args:
            input_tensor: torch.Tensor of shape [B, C, H, W]
            input_modality: name of input modality (default "S2L2A")
            output_modalities: list of output modality names
            verbose: show tqdm progress bar. Defaults to True for CLI,
                     False when running inside Streamlit (avoids BrokenPipeError
                     caused by tqdm flushing a closed stderr pipe).

        Returns:
            dict mapping modality name -> generated tensor
        """
        if output_modalities is None:
            output_modalities = ["S1GRD", "DEM", "LULC"]

        # Auto-detect Streamlit: disable tqdm by default to prevent
        # BrokenPipeError when Streamlit's stderr pipe is closed.
        if verbose is None:
            try:
                import streamlit.runtime.scriptrunner  # noqa: F401
                verbose = False
            except (ImportError, ModuleNotFoundError):
                verbose = True

        model = self._get_model([input_modality], output_modalities)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # Guard against BrokenPipeError from tqdm writing to a closed
            # stderr pipe (common in Streamlit subprocess environments).
            import io
            _orig_stderr = sys.stderr
            try:
                if not verbose:
                    # Redirect stderr to suppress tqdm output entirely
                    sys.stderr = io.StringIO()
                generated = model(input_tensor, verbose=verbose, timesteps=self.timesteps)
            except BrokenPipeError:
                # Retry with verbose=False and stderr silenced
                sys.stderr = io.StringIO()
                generated = model(input_tensor, verbose=False, timesteps=self.timesteps)
            finally:
                sys.stderr = _orig_stderr

        return generated

    # ------------------------------------------------------------------
    # PLN-Specific Analysis Methods
    # ------------------------------------------------------------------

    def analyze_vegetation_risk(self, input_tensor):
        """
        Analyze vegetation encroachment risk near transmission corridors.
        Generates NDVI and LULC from optical imagery.

        Returns:
            dict with 'ndvi', 'lulc', 'risk_map', 'statistics'
        """
        generated = self.generate(input_tensor, output_modalities=["NDVI", "LULC"])

        ndvi_data = generated["NDVI"].cpu().numpy()
        lulc_data = generated["LULC"].cpu().numpy()

        # Classify LULC
        if lulc_data.ndim == 4:
            lulc_data = lulc_data[0]
        if lulc_data.shape[0] == 10:
            lulc_classes = lulc_data.argmax(axis=0)
        else:
            lulc_classes = lulc_data[0] if lulc_data.ndim == 3 else lulc_data

        # Risk scoring: trees=3, flooded_veg=3, crops=2, scrub=2, else=0
        risk_weights = {2: 3, 4: 3, 5: 2, 6: 2}
        risk_map = np.zeros_like(lulc_classes, dtype=np.float32)
        for cls, weight in risk_weights.items():
            risk_map[lulc_classes == cls] = weight

        # Compute NDVI stats
        ndvi_flat = ndvi_data.flatten()
        total_pixels = lulc_classes.size
        stats = {
            "ndvi_mean": float(np.mean(ndvi_flat)),
            "ndvi_std": float(np.std(ndvi_flat)),
            "high_vegetation_pct": float(np.sum(ndvi_flat > 0.5) / len(ndvi_flat) * 100),
            "lulc_distribution": _compute_lulc_distribution(lulc_classes),
            "high_risk_pct": float(np.sum(risk_map >= 2) / total_pixels * 100),
            "critical_risk_pct": float(np.sum(risk_map >= 3) / total_pixels * 100),
        }

        return {
            "ndvi": generated["NDVI"],
            "lulc": generated["LULC"],
            "risk_map": risk_map,
            "lulc_classes": lulc_classes,
            "statistics": stats,
        }

    def analyze_flood_risk(self, input_tensor):
        """
        Assess flood risk around infrastructure using SAR and DEM generation.

        Returns:
            dict with 'sar', 'dem', 'flood_susceptibility', 'statistics'
        """
        generated = self.generate(input_tensor, output_modalities=["S1GRD", "DEM"])

        dem_data = generated["DEM"].cpu().numpy()
        if dem_data.ndim == 4:
            dem_data = dem_data[0]
        while dem_data.ndim > 2:
            dem_data = dem_data[0]

        sar_data = generated["S1GRD"].cpu().numpy()
        if sar_data.ndim == 4:
            sar_data = sar_data[0]

        # Flood susceptibility: low elevation + high backscatter change
        elevation_norm = (dem_data - dem_data.min()) / (dem_data.max() - dem_data.min() + 1e-6)
        # Low elevation = high flood risk
        flood_susceptibility = 1.0 - elevation_norm
        # Areas below 25th percentile elevation are highest risk
        low_threshold = np.percentile(dem_data, 25)
        flood_susceptibility[dem_data < low_threshold] *= 1.5
        flood_susceptibility = np.clip(flood_susceptibility, 0, 1)

        stats = {
            "dem_min_m": float(dem_data.min()),
            "dem_max_m": float(dem_data.max()),
            "dem_mean_m": float(dem_data.mean()),
            "low_elevation_pct": float(np.sum(dem_data < low_threshold) / dem_data.size * 100),
            "high_flood_risk_pct": float(np.sum(flood_susceptibility > 0.7) / flood_susceptibility.size * 100),
        }

        return {
            "sar": generated["S1GRD"],
            "dem": generated["DEM"],
            "flood_susceptibility": flood_susceptibility,
            "statistics": stats,
        }

    def analyze_land_use(self, input_tensor):
        """
        Monitor land use changes around PLN infrastructure.

        Returns:
            dict with 'lulc', 'lulc_classes', 'statistics'
        """
        generated = self.generate(input_tensor, output_modalities=["LULC"])

        lulc_data = generated["LULC"].cpu().numpy()
        if lulc_data.ndim == 4:
            lulc_data = lulc_data[0]
        if lulc_data.shape[0] == 10:
            lulc_classes = lulc_data.argmax(axis=0)
        else:
            lulc_classes = lulc_data[0] if lulc_data.ndim == 3 else lulc_data

        stats = {"lulc_distribution": _compute_lulc_distribution(lulc_classes)}

        # Urban expansion indicator
        stats["built_area_pct"] = stats["lulc_distribution"].get("Built Area", 0.0)
        stats["forest_pct"] = stats["lulc_distribution"].get("Trees", 0.0)
        stats["agriculture_pct"] = stats["lulc_distribution"].get("Crops", 0.0)

        return {
            "lulc": generated["LULC"],
            "lulc_classes": lulc_classes,
            "statistics": stats,
        }

    def full_site_survey(self, input_tensor):
        """
        Comprehensive multi-modal site survey combining all outputs.

        Returns:
            dict with all generated modalities and combined analysis
        """
        generated = self.generate(
            input_tensor, output_modalities=["S1GRD", "DEM", "LULC", "NDVI"]
        )

        # Extract LULC classes
        lulc_data = generated["LULC"].cpu().numpy()
        if lulc_data.ndim == 4:
            lulc_data = lulc_data[0]
        if lulc_data.shape[0] == 10:
            lulc_classes = lulc_data.argmax(axis=0)
        else:
            lulc_classes = lulc_data[0] if lulc_data.ndim == 3 else lulc_data

        ndvi_data = generated["NDVI"].cpu().numpy().flatten()
        dem_data = generated["DEM"].cpu().numpy()
        while dem_data.ndim > 2:
            dem_data = dem_data[0]

        lulc_dist = _compute_lulc_distribution(lulc_classes)

        stats = {
            "ndvi_mean": float(np.mean(ndvi_data)),
            "dem_range_m": f"{float(dem_data.min()):.0f} - {float(dem_data.max()):.0f}",
            "lulc_distribution": lulc_dist,
            "vegetation_coverage_pct": round(
                lulc_dist.get("Trees", 0) + lulc_dist.get("Crops", 0) + lulc_dist.get("Grass", 0), 1
            ),
            "built_area_pct": lulc_dist.get("Built Area", 0.0),
            "water_pct": lulc_dist.get("Water", 0.0),
        }

        return {
            "generated": generated,
            "lulc_classes": lulc_classes,
            "statistics": stats,
        }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_pln_vegetation_analysis(input_tensor, result, site_name="PLN Site", save_path=None):
    """Create a 4-panel vegetation risk analysis figure."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"PLN Vegetation Encroachment Analysis: {site_name}", fontsize=16, fontweight="bold")

    # Panel 1: Input optical
    axes[0].imshow(s2_to_rgb(input_tensor))
    axes[0].set_title("Sentinel-2 Optical")
    axes[0].axis("off")

    # Panel 2: NDVI
    axes[1].imshow(ndvi_to_rgb(result["ndvi"]))
    axes[1].set_title(f"NDVI (mean={result['statistics']['ndvi_mean']:.2f})")
    axes[1].axis("off")

    # Panel 3: LULC
    axes[2].imshow(lulc_to_rgb(result["lulc"]))
    axes[2].set_title("Land Use / Land Cover")
    axes[2].axis("off")

    # Panel 4: Risk map
    risk_cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "red"], N=4)
    im = axes[3].imshow(result["risk_map"], cmap=risk_cmap, vmin=0, vmax=3)
    axes[3].set_title(f"Risk Map ({result['statistics']['critical_risk_pct']:.1f}% critical)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], label="Risk Level", shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_pln_flood_analysis(input_tensor, result, site_name="PLN Site", save_path=None):
    """Create a 4-panel flood risk analysis figure."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"PLN Flood Risk Assessment: {site_name}", fontsize=16, fontweight="bold")

    axes[0].imshow(s2_to_rgb(input_tensor))
    axes[0].set_title("Sentinel-2 Optical")
    axes[0].axis("off")

    axes[1].imshow(s1_to_rgb(result["sar"]))
    axes[1].set_title("Generated SAR (S1 GRD)")
    axes[1].axis("off")

    axes[2].imshow(dem_to_rgb(result["dem"]))
    axes[2].set_title(f"Elevation (DEM)\n{result['statistics']['dem_min_m']:.0f}m - {result['statistics']['dem_max_m']:.0f}m")
    axes[2].axis("off")

    flood_cmap = LinearSegmentedColormap.from_list("flood", ["white", "skyblue", "blue", "darkblue"], N=256)
    im = axes[3].imshow(result["flood_susceptibility"], cmap=flood_cmap, vmin=0, vmax=1)
    axes[3].set_title(f"Flood Susceptibility ({result['statistics']['high_flood_risk_pct']:.1f}% high risk)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], label="Susceptibility", shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_pln_site_survey(input_tensor, result, site_name="PLN Site", save_path=None):
    """Create a comprehensive 5-panel site survey figure."""
    generated = result["generated"]
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(f"PLN Multi-Modal Site Survey: {site_name}", fontsize=16, fontweight="bold")

    panels = [
        ("Sentinel-2 Optical", lambda: s2_to_rgb(input_tensor)),
        ("SAR (S1 GRD)", lambda: s1_to_rgb(generated["S1GRD"])),
        ("Elevation (DEM)", lambda: dem_to_rgb(generated["DEM"])),
        ("Land Cover (LULC)", lambda: lulc_to_rgb(generated["LULC"])),
        ("Vegetation (NDVI)", lambda: ndvi_to_rgb(generated["NDVI"])),
    ]

    for ax, (title, render_fn) in zip(axes, panels):
        ax.imshow(render_fn())
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig
