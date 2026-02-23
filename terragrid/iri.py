"""
TerraGrid Infrastructure Resilience Index (IRI)
================================================
Computes a pixel-level composite risk score (0-100) for power grid
infrastructure by fusing 4 TerraMind TiM-generated modalities:

  IRI = w1·VegetationThreat + w2·FloodRisk + w3·TerrainInstability + w4·EncroachmentPressure

Each component uses one or more TiM-generated modalities (NDVI, LULC, DEM, S1GRD)
to assess a specific hazard dimension relevant to electricity infrastructure.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# LULC class IDs (ESA WorldCover)
# ---------------------------------------------------------------------------
CLS_NO_DATA = 0
CLS_WATER = 1
CLS_TREES = 2
CLS_GRASS = 3
CLS_FLOODED_VEG = 4
CLS_CROPS = 5
CLS_SCRUB = 6
CLS_BUILT = 7
CLS_BARE = 8
CLS_SNOW = 9

# Vegetation risk weights for transmission infrastructure
VEGETATION_RISK = {
    CLS_TREES: 1.0,        # Tall trees: highest risk of falling into lines
    CLS_FLOODED_VEG: 0.9,  # Flooded vegetation: structural weakness + height
    CLS_CROPS: 0.5,        # Crops: moderate risk (seasonal growth)
    CLS_SCRUB: 0.4,        # Scrub: moderate risk (dense undergrowth)
    CLS_GRASS: 0.1,        # Grass: minimal risk
}

# Default IRI component weights
DEFAULT_WEIGHTS = {
    "vegetation": 0.35,
    "flood": 0.30,
    "terrain": 0.15,
    "encroachment": 0.20,
}


# ---------------------------------------------------------------------------
# IRI Components
# ---------------------------------------------------------------------------

def compute_vegetation_threat(ndvi: np.ndarray, lulc_classes: np.ndarray) -> np.ndarray:
    """
    Vegetation encroachment threat score (0-100).

    High threat = dense vegetation (high NDVI) of tall/risky LULC classes
    near transmission corridors.

    Args:
        ndvi: 2D array of NDVI values (typically -1 to 1)
        lulc_classes: 2D array of LULC class IDs (0-9)

    Returns:
        2D array of vegetation threat scores (0-100)
    """
    # Normalize NDVI to 0-1 range
    ndvi_norm = np.clip((ndvi + 1) / 2, 0, 1)

    # LULC-based risk weight per pixel
    lulc_risk = np.zeros_like(lulc_classes, dtype=np.float32)
    for cls_id, weight in VEGETATION_RISK.items():
        lulc_risk[lulc_classes == cls_id] = weight

    # Combine: high NDVI × risky land cover = high vegetation threat
    threat = ndvi_norm * lulc_risk * 100.0
    return np.clip(threat, 0, 100).astype(np.float32)


def compute_flood_risk(dem: np.ndarray, lulc_classes: np.ndarray) -> np.ndarray:
    """
    Flood risk score (0-100).

    High risk = low elevation + proximity to water bodies.
    Uses DEM for elevation-based susceptibility and LULC for water proximity.

    Args:
        dem: 2D array of elevation values (meters)
        lulc_classes: 2D array of LULC class IDs

    Returns:
        2D array of flood risk scores (0-100)
    """
    # Elevation-based susceptibility (low elevation = high risk)
    dem_range = dem.max() - dem.min()
    if dem_range < 1e-6:
        elev_risk = np.full_like(dem, 50.0, dtype=np.float32)
    else:
        elev_norm = (dem - dem.min()) / dem_range
        elev_risk = (1.0 - elev_norm)  # Invert: low elevation = high score

    # Boost risk for areas below 25th percentile
    low_thresh = np.percentile(dem, 25)
    elev_risk[dem < low_thresh] *= 1.3

    # Water proximity component
    water_mask = (lulc_classes == CLS_WATER) | (lulc_classes == CLS_FLOODED_VEG)
    if water_mask.any():
        water_dist = distance_transform_edt(~water_mask)
        max_dist = max(water_dist.max(), 1.0)
        water_proximity = 1.0 - np.clip(water_dist / max_dist, 0, 1)
    else:
        water_proximity = np.zeros_like(dem, dtype=np.float32)

    # Combine elevation risk (70%) and water proximity (30%)
    flood = (0.7 * elev_risk + 0.3 * water_proximity) * 100.0
    return np.clip(flood, 0, 100).astype(np.float32)


def compute_terrain_instability(dem: np.ndarray) -> np.ndarray:
    """
    Terrain instability score (0-100).

    High instability = steep slopes derived from DEM gradient.
    Steep terrain poses risks to transmission towers and access roads.

    Args:
        dem: 2D array of elevation values (meters)

    Returns:
        2D array of terrain instability scores (0-100)
    """
    # Compute slope magnitude from DEM gradient
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx**2 + dy**2)

    # Normalize to 0-100 using robust percentile scaling
    p99 = np.percentile(slope, 99)
    if p99 < 1e-6:
        return np.zeros_like(dem, dtype=np.float32)

    instability = np.clip(slope / p99, 0, 1) * 100.0
    return instability.astype(np.float32)


def compute_encroachment_pressure(lulc_classes: np.ndarray) -> np.ndarray:
    """
    Encroachment pressure score (0-100).

    High pressure = natural/agricultural areas close to expanding built zones.
    Measures the "urban pressure front" that may encroach on infrastructure corridors.

    Args:
        lulc_classes: 2D array of LULC class IDs

    Returns:
        2D array of encroachment pressure scores (0-100)
    """
    built_mask = lulc_classes == CLS_BUILT
    if not built_mask.any():
        return np.zeros_like(lulc_classes, dtype=np.float32)

    # Distance from built area edges
    built_dist = distance_transform_edt(~built_mask)
    max_dist = max(built_dist.max(), 1.0)

    # Inverse distance = pressure (high near built edges, decays outward)
    pressure = 1.0 - np.clip(built_dist / max_dist, 0, 1)

    # Only count pressure on non-built pixels (the encroachment frontier)
    pressure[built_mask] = 0.5  # Built areas themselves: moderate baseline

    return np.clip(pressure * 100.0, 0, 100).astype(np.float32)


# ---------------------------------------------------------------------------
# Composite IRI
# ---------------------------------------------------------------------------

def compute_iri(
    ndvi: np.ndarray,
    lulc_classes: np.ndarray,
    dem: np.ndarray,
    sar: np.ndarray = None,
    weights: dict = None,
    modality_subset: list = None,
) -> dict:
    """
    Compute the composite Infrastructure Resilience Index.

    Fuses 4 TiM-generated modality risk components into a single 0-100 score.
    Supports ablation via modality_subset to test contribution of each modality.

    Args:
        ndvi: 2D NDVI array
        lulc_classes: 2D LULC class array (0-9)
        dem: 2D DEM array (meters)
        sar: 2D SAR array (optional, for future SAR-specific anomaly detection)
        weights: Component weights dict, defaults to DEFAULT_WEIGHTS
        modality_subset: If set, only use these modalities for ablation.
                         E.g., ["NDVI", "LULC"] to skip DEM/SAR components.

    Returns:
        dict with:
            - iri_map: 2D array (0-100)
            - components: dict of individual component maps
            - statistics: summary statistics
            - risk_category: overall site risk level
    """
    w = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)

    # Determine which components are active based on modality_subset
    has_ndvi = modality_subset is None or "NDVI" in modality_subset
    has_lulc = modality_subset is None or "LULC" in modality_subset
    has_dem = modality_subset is None or "DEM" in modality_subset
    has_sar = modality_subset is None or "S1GRD" in modality_subset

    components = {}
    active_weight_sum = 0.0

    # Vegetation threat (requires NDVI + LULC)
    if has_ndvi and has_lulc:
        components["vegetation_threat"] = compute_vegetation_threat(ndvi, lulc_classes)
        active_weight_sum += w["vegetation"]
    elif has_lulc:
        # Fallback: LULC-only vegetation risk (no NDVI intensity)
        lulc_risk = np.zeros_like(lulc_classes, dtype=np.float32)
        for cls_id, weight in VEGETATION_RISK.items():
            lulc_risk[lulc_classes == cls_id] = weight
        components["vegetation_threat"] = np.clip(lulc_risk * 100.0, 0, 100)
        active_weight_sum += w["vegetation"]

    # Flood risk (requires DEM, optionally LULC for water proximity)
    if has_dem:
        if has_lulc:
            components["flood_risk"] = compute_flood_risk(dem, lulc_classes)
        else:
            # DEM-only flood risk (no water proximity)
            dem_range = dem.max() - dem.min()
            if dem_range < 1e-6:
                components["flood_risk"] = np.full_like(dem, 50.0, dtype=np.float32)
            else:
                elev_norm = (dem - dem.min()) / dem_range
                components["flood_risk"] = np.clip((1.0 - elev_norm) * 100.0, 0, 100)
        active_weight_sum += w["flood"]

    # Terrain instability (requires DEM)
    if has_dem:
        components["terrain_instability"] = compute_terrain_instability(dem)
        active_weight_sum += w["terrain"]

    # Encroachment pressure (requires LULC)
    if has_lulc:
        components["encroachment_pressure"] = compute_encroachment_pressure(lulc_classes)
        active_weight_sum += w["encroachment"]

    # Compose IRI
    h, w_px = ndvi.shape if ndvi is not None else (dem.shape if dem is not None else lulc_classes.shape)
    iri_map = np.zeros((h, w_px), dtype=np.float32)

    if active_weight_sum > 0:
        # Re-normalize weights for active components
        weight_map = {
            "vegetation_threat": w["vegetation"],
            "flood_risk": w["flood"],
            "terrain_instability": w["terrain"],
            "encroachment_pressure": w["encroachment"],
        }
        for comp_name, comp_map in components.items():
            comp_weight = weight_map.get(comp_name, 0) / active_weight_sum
            iri_map += comp_weight * comp_map

    iri_map = np.clip(iri_map, 0, 100)

    # Statistics
    mean_iri = float(np.mean(iri_map))
    if mean_iri >= 65:
        risk_cat = "CRITICAL"
    elif mean_iri >= 45:
        risk_cat = "HIGH"
    elif mean_iri >= 25:
        risk_cat = "MODERATE"
    else:
        risk_cat = "LOW"

    stats = {
        "iri_mean": round(mean_iri, 1),
        "iri_std": round(float(np.std(iri_map)), 1),
        "iri_max": round(float(np.max(iri_map)), 1),
        "iri_min": round(float(np.min(iri_map)), 1),
        "high_risk_pct": round(float(np.sum(iri_map >= 60) / iri_map.size * 100), 1),
        "critical_risk_pct": round(float(np.sum(iri_map >= 80) / iri_map.size * 100), 1),
        "risk_category": risk_cat,
        "active_modalities": [m for m, active in
                               [("NDVI", has_ndvi), ("LULC", has_lulc),
                                ("DEM", has_dem), ("S1GRD", has_sar)]
                               if active],
        "component_means": {
            name: round(float(np.mean(comp)), 1)
            for name, comp in components.items()
        },
    }

    return {
        "iri_map": iri_map,
        "components": components,
        "statistics": stats,
        "risk_category": risk_cat,
    }


# ---------------------------------------------------------------------------
# Helper: extract modality arrays from TerraMind output
# ---------------------------------------------------------------------------

def extract_modality_arrays(generated: dict) -> dict:
    """
    Extract numpy arrays from TerraMind generated output tensors.

    Args:
        generated: dict of modality_name -> torch.Tensor

    Returns:
        dict with ndvi, lulc_classes, dem, sar as numpy arrays
    """
    import torch

    result = {}

    # NDVI
    if "NDVI" in generated:
        ndvi = generated["NDVI"]
        if isinstance(ndvi, torch.Tensor):
            ndvi = ndvi.cpu().numpy()
        while ndvi.ndim > 2:
            ndvi = ndvi[0]
        result["ndvi"] = ndvi

    # LULC
    if "LULC" in generated:
        lulc = generated["LULC"]
        if isinstance(lulc, torch.Tensor):
            lulc = lulc.cpu().numpy()
        if lulc.ndim == 4:
            lulc = lulc[0]
        if lulc.shape[0] == 10:
            lulc = lulc.argmax(axis=0)
        elif lulc.ndim == 3:
            lulc = lulc[0]
        result["lulc_classes"] = lulc

    # DEM
    if "DEM" in generated:
        dem = generated["DEM"]
        if isinstance(dem, torch.Tensor):
            dem = dem.cpu().numpy()
        while dem.ndim > 2:
            dem = dem[0]
        result["dem"] = dem

    # SAR (S1GRD)
    if "S1GRD" in generated:
        sar = generated["S1GRD"]
        if isinstance(sar, torch.Tensor):
            sar = sar.cpu().numpy()
        if sar.ndim == 4:
            sar = sar[0]
        result["sar"] = sar

    return result
