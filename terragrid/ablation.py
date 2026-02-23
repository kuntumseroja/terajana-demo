"""
TerraGrid TiM Ablation Study
==============================
Systematically tests which TiM-generated modalities contribute most
to the Infrastructure Resilience Index by running IRI computation
with different modality subsets.
"""

import time
import numpy as np
import torch

# Ablation configurations: name -> list of TiM modalities to generate
ABLATION_CONFIGS = {
    "Full TiM (4 modalities)": ["S1GRD", "DEM", "LULC", "NDVI"],
    "Without SAR":             ["DEM", "LULC", "NDVI"],
    "Without DEM":             ["S1GRD", "LULC", "NDVI"],
    "Without NDVI":            ["S1GRD", "DEM", "LULC"],
    "LULC only":               ["LULC"],
    "No TiM (spectral only)":  [],
}


def run_ablation(analyzer, input_tensor: torch.Tensor, site_name: str = "Site") -> list:
    """
    Run IRI computation across all ablation configs for a single site.

    Args:
        analyzer: PLNTerraMindAnalyzer instance
        input_tensor: (1, 12, 224, 224) Sentinel-2 tensor
        site_name: Name for display

    Returns:
        list of dicts with config_name, iri_mean, modalities, iri_map, components, time_s
    """
    from terragrid.iri import compute_iri, extract_modality_arrays

    results = []

    for config_name, modalities in ABLATION_CONFIGS.items():
        t0 = time.time()

        if not modalities:
            # No TiM: compute a basic spectral-only proxy
            iri_result = _spectral_only_iri(input_tensor)
        else:
            # Generate requested modalities via TerraMind
            generated = analyzer.generate(
                input_tensor,
                output_modalities=modalities,
                verbose=False,
            )

            # Extract numpy arrays
            arrays = extract_modality_arrays(generated)

            # Compute IRI with available modalities
            iri_result = compute_iri(
                ndvi=arrays.get("ndvi"),
                lulc_classes=arrays.get("lulc_classes"),
                dem=arrays.get("dem"),
                sar=arrays.get("sar"),
                modality_subset=modalities,
            )

        elapsed = time.time() - t0

        results.append({
            "config_name": config_name,
            "modalities": modalities,
            "iri_mean": iri_result["statistics"]["iri_mean"],
            "iri_std": iri_result["statistics"]["iri_std"],
            "risk_category": iri_result["risk_category"],
            "high_risk_pct": iri_result["statistics"]["high_risk_pct"],
            "iri_map": iri_result["iri_map"],
            "components": iri_result["components"],
            "component_means": iri_result["statistics"]["component_means"],
            "time_s": round(elapsed, 1),
        })

    return results


def _spectral_only_iri(input_tensor: torch.Tensor) -> dict:
    """
    Baseline IRI using only spectral indices from the raw Sentinel-2 input.
    No TerraMind generation — just NDVI computed from B4 (Red) and B8 (NIR).
    """
    data = input_tensor.cpu().numpy()
    if data.ndim == 4:
        data = data[0]

    # Sentinel-2 L2A bands: B4 (Red) = index 3, B8 (NIR) = index 7
    red = data[3].astype(np.float32)
    nir = data[7].astype(np.float32)

    # Compute raw NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)

    # Use NDVI as sole risk proxy (high vegetation = some risk)
    ndvi_norm = np.clip((ndvi + 1) / 2, 0, 1)
    iri_map = np.clip(ndvi_norm * 50, 0, 100).astype(np.float32)  # Max 50 without TiM

    mean_iri = float(np.mean(iri_map))
    return {
        "iri_map": iri_map,
        "components": {"spectral_ndvi": iri_map},
        "statistics": {
            "iri_mean": round(mean_iri, 1),
            "iri_std": round(float(np.std(iri_map)), 1),
            "iri_max": round(float(np.max(iri_map)), 1),
            "iri_min": round(float(np.min(iri_map)), 1),
            "high_risk_pct": round(float(np.sum(iri_map >= 60) / iri_map.size * 100), 1),
            "critical_risk_pct": round(float(np.sum(iri_map >= 80) / iri_map.size * 100), 1),
            "risk_category": "HIGH" if mean_iri >= 45 else "MODERATE" if mean_iri >= 25 else "LOW",
            "active_modalities": [],
            "component_means": {"spectral_ndvi": round(mean_iri, 1)},
        },
        "risk_category": "HIGH" if mean_iri >= 45 else "MODERATE" if mean_iri >= 25 else "LOW",
    }
