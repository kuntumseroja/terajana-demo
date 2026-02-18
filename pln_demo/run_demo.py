#!/usr/bin/env python3
"""
PLN TerraMind Demo - Command Line Interface
============================================
Demonstrates satellite-based monitoring capabilities for PLN
(Perusahaan Listrik Negara), Indonesia's national electricity company.

Uses TerraMind's any-to-any generation to produce SAR, DEM, LULC, and NDVI
from Sentinel-2 optical imagery for infrastructure monitoring use cases.

Usage:
    python pln_demo/run_demo.py                     # Run all analyses on example data
    python pln_demo/run_demo.py --analysis vegetation # Run specific analysis
    python pln_demo/run_demo.py --input path/to/image.tif  # Use custom input
    python pln_demo/run_demo.py --model-size small   # Use smaller model (faster)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "notebooks"))

from pln_demo.pln_analysis import (
    PLNTerraMindAnalyzer,
    plot_pln_vegetation_analysis,
    plot_pln_flood_analysis,
    plot_pln_site_survey,
    s2_to_rgb,
    select_device,
)
from pln_demo.pln_config import PLN_SITES, PLN_USE_CASES


def load_example_data(example_id=0):
    """Load one of the bundled TerraMesh example images."""
    import rioxarray as rxr

    examples_dir = os.path.join(PROJECT_ROOT, "examples", "S2L2A")
    files = sorted(
        [f for f in os.listdir(examples_dir) if f.endswith(".tif")]
    )
    if not files:
        raise FileNotFoundError(f"No .tif files found in {examples_dir}")

    example_id = min(example_id, len(files) - 1)
    filepath = os.path.join(examples_dir, files[example_id])
    print(f"[Data] Loading: {filepath}")

    data = rxr.open_rasterio(filepath)
    tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
    print(f"[Data] Shape: {tensor.shape} (B, C, H, W)")
    return tensor, files[example_id]


def load_custom_data(filepath):
    """Load a custom GeoTIFF file."""
    import rioxarray as rxr

    print(f"[Data] Loading custom: {filepath}")
    data = rxr.open_rasterio(filepath)
    tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
    print(f"[Data] Shape: {tensor.shape} (B, C, H, W)")
    return tensor, os.path.basename(filepath)


def print_banner():
    print("=" * 70)
    print("  PLN TerraMind Demo")
    print("  Satellite-based Infrastructure Monitoring")
    print("  Perusahaan Listrik Negara - Indonesia")
    print("=" * 70)
    print()
    print("  Powered by: IBM TerraMind (any-to-any Earth Observation model)")
    print("  Modalities: Sentinel-2, Sentinel-1 SAR, DEM, LULC, NDVI")
    print()


def print_site_info():
    print("PLN Infrastructure Sites:")
    print("-" * 50)
    for name, info in PLN_SITES.items():
        print(f"  {name:30s} | {info['type']:20s} | {info['capacity_mw']:>5d} MW | {info['region']}")
    print()


def print_stats(title, stats):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k:25s}: {v}")
        elif isinstance(value, float):
            print(f"  {key:30s}: {value:.2f}")
        else:
            print(f"  {key:30s}: {value}")
    print()


def run_vegetation_analysis(analyzer, input_tensor, filename, output_dir):
    """Run vegetation encroachment analysis."""
    print("\n[1/3] Running Vegetation Encroachment Analysis...")
    print("     Use case: Detect vegetation growth near transmission lines")
    print("     Generating: NDVI + LULC from Sentinel-2 optical")

    t0 = time.time()
    result = analyzer.analyze_vegetation_risk(input_tensor)
    elapsed = time.time() - t0

    print_stats("Vegetation Risk Analysis Results", result["statistics"])
    print(f"  Inference time: {elapsed:.1f}s")

    # Determine risk level
    crit = result["statistics"]["critical_risk_pct"]
    if crit > 40:
        print("  ** ALERT: HIGH vegetation encroachment risk detected! **")
        print("  ** Recommend: Immediate field inspection of corridor **")
    elif crit > 20:
        print("  ** WARNING: Moderate vegetation risk detected **")
        print("  ** Recommend: Schedule vegetation management **")
    else:
        print("  Status: Low vegetation risk - corridor clear")

    save_path = os.path.join(output_dir, f"pln_vegetation_{filename}.png")
    plot_pln_vegetation_analysis(
        input_tensor, result, site_name=f"Demo ({filename})", save_path=save_path
    )
    return result


def run_flood_analysis(analyzer, input_tensor, filename, output_dir):
    """Run flood risk assessment."""
    print("\n[2/3] Running Flood Risk Assessment...")
    print("     Use case: Assess flooding risk near substations & plants")
    print("     Generating: SAR (S1 GRD) + DEM from Sentinel-2 optical")

    t0 = time.time()
    result = analyzer.analyze_flood_risk(input_tensor)
    elapsed = time.time() - t0

    print_stats("Flood Risk Assessment Results", result["statistics"])
    print(f"  Inference time: {elapsed:.1f}s")

    risk = result["statistics"]["high_flood_risk_pct"]
    if risk > 50:
        print("  ** ALERT: Significant flood susceptibility detected! **")
        print("  ** Recommend: Review drainage and flood barriers **")
    elif risk > 25:
        print("  ** WARNING: Moderate flood risk in area **")
    else:
        print("  Status: Low flood risk")

    save_path = os.path.join(output_dir, f"pln_flood_{filename}.png")
    plot_pln_flood_analysis(
        input_tensor, result, site_name=f"Demo ({filename})", save_path=save_path
    )
    return result


def run_site_survey(analyzer, input_tensor, filename, output_dir):
    """Run comprehensive multi-modal site survey."""
    print("\n[3/3] Running Multi-Modal Site Survey...")
    print("     Use case: Comprehensive site assessment for infrastructure planning")
    print("     Generating: SAR + DEM + LULC + NDVI from Sentinel-2 optical")

    t0 = time.time()
    result = analyzer.full_site_survey(input_tensor)
    elapsed = time.time() - t0

    print_stats("Site Survey Results", result["statistics"])
    print(f"  Inference time: {elapsed:.1f}s")

    save_path = os.path.join(output_dir, f"pln_survey_{filename}.png")
    plot_pln_site_survey(
        input_tensor, result, site_name=f"Demo ({filename})", save_path=save_path
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="PLN TerraMind Demo")
    parser.add_argument(
        "--analysis",
        choices=["vegetation", "flood", "survey", "all"],
        default="all",
        help="Which analysis to run (default: all)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a custom Sentinel-2 L2A GeoTIFF (12-band, 224x224)",
    )
    parser.add_argument(
        "--example-id", type=int, default=0,
        help="Index of bundled example to use (0-4)",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base"],
        default="base",
        help="TerraMind model size (default: base)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10,
        help="Number of diffusion timesteps (default: 10, lower=faster)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save output plots (default: pln_demo/output)",
    )
    args = parser.parse_args()

    print_banner()
    print_site_info()

    # Output directory
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "pln_demo", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if args.input:
        input_tensor, filename = load_custom_data(args.input)
    else:
        input_tensor, filename = load_example_data(args.example_id)

    # Initialize analyzer
    analyzer = PLNTerraMindAnalyzer(
        model_size=args.model_size,
        timesteps=args.timesteps,
    )

    # Run analyses
    if args.analysis in ("vegetation", "all"):
        run_vegetation_analysis(analyzer, input_tensor, filename, output_dir)

    if args.analysis in ("flood", "all"):
        run_flood_analysis(analyzer, input_tensor, filename, output_dir)

    if args.analysis in ("survey", "all"):
        run_site_survey(analyzer, input_tensor, filename, output_dir)

    print("\n" + "=" * 70)
    print(f"  Demo complete! Output saved to: {output_dir}")
    print("=" * 70)
    print("\nTo launch the interactive web dashboard:")
    print(f"  source venv/bin/activate")
    print(f"  streamlit run pln_demo/app.py")
    print()


if __name__ == "__main__":
    main()
