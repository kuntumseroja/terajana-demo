"""
TerraMind Client
================
Python client for the TerraMind Local Server API.

Usage:
    from pln_demo.terramind_client import TerraMindClient

    client = TerraMindClient("http://localhost:8786")

    # Check server status
    status = client.status()

    # Run analysis from GeoTIFF file
    result = client.analyze_file("path/to/satellite.tif", analysis_type="vegetation")
    print(result["statistics"])

    # Get rendered images as PIL
    images = client.analyze_file("path/to/satellite.tif", return_pil=True)
    images["ndvi"].show()
"""

import base64
import io
import requests
from pathlib import Path
from typing import Optional
from PIL import Image


class TerraMindClient:
    """Client for the TerraMind Local Server."""

    def __init__(self, base_url: str = "http://localhost:8786", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Health & Status
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Check server health."""
        r = requests.get(f"{self.base_url}/", timeout=10)
        r.raise_for_status()
        return r.json()

    def is_ready(self) -> bool:
        """Check if server is ready with model loaded."""
        try:
            info = self.health()
            return info.get("status") == "ready"
        except Exception:
            return False

    def status(self) -> dict:
        """Get detailed model status."""
        r = requests.get(f"{self.base_url}/v1/status", timeout=10)
        r.raise_for_status()
        return r.json()

    def models(self) -> dict:
        """List available models."""
        r = requests.get(f"{self.base_url}/v1/models", timeout=10)
        r.raise_for_status()
        return r.json()

    def load_model(self, model_size: str = "base", timesteps: int = 10) -> dict:
        """Load or reload a model."""
        r = requests.post(
            f"{self.base_url}/v1/load",
            json={"model_size": model_size, "timesteps": timesteps},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        geotiff_path: str,
        output_modalities: list[str] = None,
        return_images: bool = True,
        return_pil: bool = False,
    ) -> dict:
        """
        Generate output modalities from a GeoTIFF file.

        Args:
            geotiff_path: Path to input GeoTIFF file
            output_modalities: List of modalities to generate (default: S1GRD, DEM, LULC)
            return_images: Include rendered images in response
            return_pil: Convert base64 images to PIL Image objects

        Returns:
            dict with inference results and optional images
        """
        if output_modalities is None:
            output_modalities = ["S1GRD", "DEM", "LULC"]

        geotiff_bytes = Path(geotiff_path).read_bytes()
        geotiff_b64 = base64.b64encode(geotiff_bytes).decode("utf-8")

        r = requests.post(
            f"{self.base_url}/v1/generate",
            json={
                "geotiff_b64": geotiff_b64,
                "output_modalities": output_modalities,
                "return_images": return_images,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        result = r.json()

        if return_pil and return_images:
            result = self._convert_images_to_pil(result, key="outputs", nested=True)
            if "input_preview_b64" in result:
                result["input_preview"] = self._b64_to_pil(result.pop("input_preview_b64"))

        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze(
        self,
        geotiff_path: str,
        analysis_type: str = "full_survey",
        return_images: bool = True,
        return_pil: bool = False,
    ) -> dict:
        """
        Run PLN-specific analysis on a GeoTIFF file.

        Args:
            geotiff_path: Path to input GeoTIFF file
            analysis_type: One of 'vegetation', 'flood', 'land_use', 'full_survey'
            return_images: Include rendered images in response
            return_pil: Convert base64 images to PIL Image objects

        Returns:
            dict with statistics and optional images
        """
        geotiff_bytes = Path(geotiff_path).read_bytes()
        geotiff_b64 = base64.b64encode(geotiff_bytes).decode("utf-8")

        r = requests.post(
            f"{self.base_url}/v1/analyze",
            json={
                "geotiff_b64": geotiff_b64,
                "analysis_type": analysis_type,
                "return_images": return_images,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        result = r.json()

        if return_pil and return_images and "images" in result:
            result["images"] = {
                name: self._b64_to_pil(b64)
                for name, b64 in result["images"].items()
            }

        return result

    # Convenience methods
    def analyze_vegetation(self, geotiff_path: str, **kwargs) -> dict:
        return self.analyze(geotiff_path, analysis_type="vegetation", **kwargs)

    def analyze_flood(self, geotiff_path: str, **kwargs) -> dict:
        return self.analyze(geotiff_path, analysis_type="flood", **kwargs)

    def analyze_land_use(self, geotiff_path: str, **kwargs) -> dict:
        return self.analyze(geotiff_path, analysis_type="land_use", **kwargs)

    def full_survey(self, geotiff_path: str, **kwargs) -> dict:
        return self.analyze(geotiff_path, analysis_type="full_survey", **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _b64_to_pil(b64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        img_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(img_bytes))

    def _convert_images_to_pil(self, result: dict, key: str, nested: bool = False) -> dict:
        """Convert base64 image fields in result to PIL Images."""
        if key in result:
            for name, data in result[key].items():
                if nested and isinstance(data, dict) and "image_b64" in data:
                    data["image"] = self._b64_to_pil(data.pop("image_b64"))
                elif isinstance(data, str):
                    result[key][name] = self._b64_to_pil(data)
        return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8786"
    client = TerraMindClient(url)

    print(f"Connecting to {url}...")
    try:
        info = client.health()
        print(f"  Status: {info['status']}")
        print(f"  Model:  {info.get('model', 'none')}")
        print(f"  Device: {info.get('device', 'N/A')}")
        print(f"  Served: {info.get('requests_served', 0)} requests")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    # Test with local satellite data
    test_files = list(Path("demo_data/indonesia").glob("*.tif"))
    if test_files:
        print(f"\nTesting with: {test_files[0].name}")
        result = client.analyze(str(test_files[0]), analysis_type="land_use", return_images=False)
        print(f"  Analysis type: {result['analysis_type']}")
        print(f"  Inference time: {result['inference_time_s']}s")
        print(f"  Statistics: {result['statistics']}")
    else:
        print("\nNo test files found in demo_data/indonesia/")
