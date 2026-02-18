#!/usr/bin/env python3
"""
Download Sentinel-2 L2A 12-band GeoTIFF tiles for PLN sites in Indonesia.

Produces 224x224 pixel, 12-band int16 GeoTIFFs at 10m resolution in UTM,
matching the TerraMind / terratorch expected input format.

Sites:
  - Aceh (Banda Aceh) - PLTU Nagan Raya power plant area
  - Sumatra Barat (Padang) - PLN transmission corridor
  - Jawa Barat (Bandung/Saguling) - PLTA Saguling hydroelectric area

Requirements:
    pip install pystac-client rasterio numpy pyproj
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from pystac_client import Client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "demo_data", "indonesia")

# Sentinel-2 L2A STAC band asset keys (12 bands used by TerraMind)
BAND_KEYS = [
    "coastal",   # B01 - Coastal aerosol (60m)
    "blue",      # B02 - Blue (10m)
    "green",     # B03 - Green (10m)
    "red",       # B04 - Red (10m)
    "rededge1",  # B05 - Vegetation Red Edge 1 (20m)
    "rededge2",  # B06 - Vegetation Red Edge 2 (20m)
    "rededge3",  # B07 - Vegetation Red Edge 3 (20m)
    "nir",       # B08 - NIR (10m)
    "nir08",     # B8A - Narrow NIR (20m)
    "nir09",     # B09 - Water vapour (60m)
    "swir16",    # B11 - SWIR 1.6um (20m)
    "swir22",    # B12 - SWIR 2.2um (20m)
]

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06",
              "B07", "B08", "B8A", "B09", "B11", "B12"]

TILE_SIZE = 224
PIXEL_RES = 10
TILE_EXTENT = TILE_SIZE * PIXEL_RES  # 2240m

MAX_CLOUD_COVER = 15
DATE_RANGE = "2023-06-01/2024-06-30"

# PLN Sites in Indonesia with realistic power infrastructure locations
PLN_SITES = {
    "aceh_nagan_raya": {
        "lat": 4.1500,
        "lon": 96.2000,
        "province": "Aceh",
        "description": "PLTU Nagan Raya Coal Power Plant Area",
        "pln_context": "210 MW coal-fired plant, coastal transmission corridor",
    },
    "sumbar_padang": {
        "lat": -0.9471,
        "lon": 100.3543,
        "province": "Sumatra Barat",
        "description": "PLN Padang Distribution Hub & Transmission Lines",
        "pln_context": "Main distribution hub, hillside terrain with vegetation risk",
    },
    "jabar_saguling": {
        "lat": -6.9218,
        "lon": 107.3851,
        "province": "Jawa Barat",
        "description": "PLTA Saguling Hydroelectric Dam & Reservoir",
        "pln_context": "700 MW hydroelectric, reservoir monitoring, flood risk",
    },
}


def latlon_to_utm(lat, lon):
    """Determine UTM EPSG and project lat/lon to UTM coordinates."""
    zone_number = int((lon + 180) / 6) + 1
    epsg = (32600 + zone_number) if lat >= 0 else (32700 + zone_number)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return epsg, utm_x, utm_y


def search_sentinel2(client, lat, lon, date_range, max_cloud):
    """Search for least-cloudy Sentinel-2 L2A scene covering a point."""
    point = {"type": "Point", "coordinates": [lon, lat]}

    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=point,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=20,
    )

    items = list(search.items())
    if not items:
        print(f"  WARNING: No scenes found with <{max_cloud}% cloud. Trying 30%...")
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=point,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=20,
        )
        items = list(search.items())

    if not items:
        print(f"  ERROR: No scenes found at all. Check coordinates.")
        return None

    best = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    cc = best.properties.get("eo:cloud_cover", "?")
    print(f"  Best scene: {best.id}")
    print(f"  Cloud cover: {cc}%  |  Date: {best.datetime}")
    return best


def download_tile(item, lat, lon, output_path):
    """Download 224x224 12-band GeoTIFF centered on (lat, lon)."""
    utm_epsg, center_x, center_y = latlon_to_utm(lat, lon)
    half = TILE_EXTENT / 2.0
    xmin, xmax = center_x - half, center_x + half
    ymin, ymax = center_y - half, center_y + half

    all_bands = np.zeros((len(BAND_KEYS), TILE_SIZE, TILE_SIZE), dtype=np.int16)
    success_count = 0

    for i, band_key in enumerate(BAND_KEYS):
        asset = item.assets.get(band_key)
        if asset is None:
            print(f"    WARN: Band '{band_key}' not found, trying alternate keys...")
            # Try alternate naming conventions
            alt_keys = {
                "coastal": ["B01", "b01"],
                "blue": ["B02", "b02"],
                "green": ["B03", "b03"],
                "red": ["B04", "b04"],
                "rededge1": ["B05", "b05"],
                "rededge2": ["B06", "b06"],
                "rededge3": ["B07", "b07"],
                "nir": ["B08", "b08"],
                "nir08": ["B8A", "b8a"],
                "nir09": ["B09", "b09"],
                "swir16": ["B11", "b11"],
                "swir22": ["B12", "b12"],
            }
            for alt in alt_keys.get(band_key, []):
                if alt in item.assets:
                    asset = item.assets[alt]
                    break
            if asset is None:
                print(f"    SKIP: Band '{band_key}' ({BAND_NAMES[i]}) unavailable")
                continue

        href = asset.href
        try:
            with rasterio.open(href) as src:
                src_crs = src.crs
                src_epsg = src_crs.to_epsg() if src_crs else None

                if src_epsg and src_epsg != utm_epsg:
                    t = Transformer.from_crs(
                        f"EPSG:{utm_epsg}", f"EPSG:{src_epsg}", always_xy=True
                    )
                    bxmin, bymin = t.transform(xmin, ymin)
                    bxmax, bymax = t.transform(xmax, ymax)
                else:
                    bxmin, bymin, bxmax, bymax = xmin, ymin, xmax, ymax

                window = from_bounds(bxmin, bymin, bxmax, bymax, src.transform)

                data = src.read(
                    1,
                    window=window,
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                )

                all_bands[i] = data.astype(np.int16)
                success_count += 1
                print(f"    Band {i+1:2d}/{len(BAND_KEYS)} ({BAND_NAMES[i]:>3s} "
                      f"{band_key:>10s}): range=[{data.min():6d}, {data.max():6d}]")

        except Exception as e:
            print(f"    ERROR band '{band_key}': {e}")
            continue

    if success_count == 0:
        print("  FAILED: No bands downloaded successfully.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_crs = CRS.from_epsg(utm_epsg)
    output_transform = transform_from_bounds(xmin, ymin, xmax, ymax, TILE_SIZE, TILE_SIZE)

    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": TILE_SIZE,
        "height": TILE_SIZE,
        "count": len(BAND_KEYS),
        "crs": output_crs,
        "transform": output_transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(all_bands)
        for idx, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(idx, name)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"    SAVED: {output_path} ({size_kb:.0f} KB)")
    print(f"    Shape: {all_bands.shape}, CRS: EPSG:{utm_epsg}, "
          f"Bands: {success_count}/{len(BAND_KEYS)}")
    return True


def main():
    print("=" * 70)
    print("  PLN Indonesia - Sentinel-2 L2A Tile Downloader")
    print("  TerraMind-compatible: 12-band, 224x224, int16, UTM")
    print("=" * 70)
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Date range : {DATE_RANGE}")
    print(f"  Cloud limit: <{MAX_CLOUD_COVER}%")
    print(f"  Tile size  : {TILE_SIZE}x{TILE_SIZE} @ {PIXEL_RES}m")
    print("=" * 70)

    client = Client.open("https://earth-search.aws.element84.com/v1")
    results = {}

    for site_name, info in PLN_SITES.items():
        lat, lon = info["lat"], info["lon"]
        print(f"\n{'─' * 60}")
        print(f"  Site: {info['province']} - {info['description']}")
        print(f"  PLN Context: {info['pln_context']}")
        print(f"  Coordinates: {lat:.4f}N, {lon:.4f}E")
        print(f"  Searching Sentinel-2 L2A catalog...")

        item = search_sentinel2(client, lat, lon, DATE_RANGE, MAX_CLOUD_COVER)
        if item is None:
            results[site_name] = False
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{site_name}_s2l2a.tif")
        print(f"  Downloading 12-band tile...")
        ok = download_tile(item, lat, lon, output_path)
        results[site_name] = ok

    print(f"\n{'=' * 70}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    for site_name, ok in results.items():
        status = "OK" if ok else "FAILED"
        info = PLN_SITES[site_name]
        print(f"  [{status:>6s}] {info['province']:>15s} - {info['description']}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
