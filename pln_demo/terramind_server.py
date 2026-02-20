"""
TerraMind Local API Server
===========================
A local REST API that serves IBM TerraMind model inference,
similar to how Ollama serves LLMs but for geospatial AI.

Start server:
    python pln_demo/terramind_server.py

Or with uvicorn:
    uvicorn pln_demo.terramind_server:app --host 0.0.0.0 --port 8786

API Endpoints:
    GET  /              - Server info & health check
    GET  /v1/models     - List available models
    POST /v1/generate   - Run TerraMind inference
    POST /v1/analyze    - Run PLN-specific analysis
    GET  /v1/status     - Model loading status
"""

import os
import sys
import io
import time
import base64
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

from pln_demo.pln_analysis import (
    PLNTerraMindAnalyzer,
    s2_to_rgb, s1_to_rgb, dem_to_rgb, lulc_to_rgb, ndvi_to_rgb,
    LULC_CLASS_NAMES, MODALITY_RENDERERS, select_device,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("terramind-server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
SERVER_STATE = {
    "analyzer": None,
    "model_size": None,
    "device": None,
    "load_time_s": None,
    "requests_served": 0,
    "started_at": None,
}

# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------
MODEL_SIZE = os.environ.get("TERRAMIND_MODEL_SIZE", "base")
TIMESTEPS = int(os.environ.get("TERRAMIND_TIMESTEPS", "10"))
AUTO_LOAD = os.environ.get("TERRAMIND_AUTO_LOAD", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TerraMind model on server startup."""
    SERVER_STATE["started_at"] = time.time()

    if AUTO_LOAD:
        logger.info(f"Loading TerraMind model: size={MODEL_SIZE}, timesteps={TIMESTEPS}")
        t0 = time.time()
        try:
            analyzer = PLNTerraMindAnalyzer(
                model_size=MODEL_SIZE,
                timesteps=TIMESTEPS,
            )
            # Warm up — build the most common model combination
            logger.info("Warming up model (building S2L2A -> S1GRD, DEM, LULC pipeline)...")
            dummy = torch.randn(1, 12, 224, 224)
            _ = analyzer.generate(dummy, output_modalities=["S1GRD", "DEM", "LULC"], verbose=False)

            load_time = time.time() - t0
            SERVER_STATE["analyzer"] = analyzer
            SERVER_STATE["model_size"] = MODEL_SIZE
            SERVER_STATE["device"] = str(analyzer.device)
            SERVER_STATE["load_time_s"] = round(load_time, 1)
            logger.info(f"TerraMind ready! Device: {analyzer.device} | Load time: {load_time:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Server will start without model. Use POST /v1/load to load manually.")
    else:
        logger.info("Auto-load disabled. Use POST /v1/load to load the model.")

    yield

    logger.info("Shutting down TerraMind server...")
    SERVER_STATE["analyzer"] = None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TerraMind Local Server",
    description="Local REST API for IBM TerraMind geospatial AI inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Request body for /v1/generate when sending base64-encoded GeoTIFF."""
    geotiff_b64: str
    input_modality: str = "S2L2A"
    output_modalities: list[str] = ["S1GRD", "DEM", "LULC"]
    timesteps: Optional[int] = None
    return_images: bool = True  # Return rendered RGB images as base64 PNG


class AnalyzeRequest(BaseModel):
    """Request body for /v1/analyze when sending base64-encoded GeoTIFF."""
    geotiff_b64: str
    analysis_type: str = "full_survey"  # vegetation, flood, land_use, full_survey
    return_images: bool = True


class LoadModelRequest(BaseModel):
    """Request body for /v1/load."""
    model_size: str = "base"
    timesteps: int = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _require_model():
    """Check that model is loaded, raise 503 if not."""
    if SERVER_STATE["analyzer"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Send POST /v1/load to load the model first.",
        )
    return SERVER_STATE["analyzer"]


def _load_geotiff_tensor(geotiff_bytes: bytes) -> torch.Tensor:
    """Load GeoTIFF bytes into a (1, C, H, W) float32 tensor."""
    import rasterio

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(geotiff_bytes)
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path) as src:
            data = src.read()  # (C, H, W)
    finally:
        os.unlink(tmp_path)

    return torch.tensor(data.astype(np.float32)).unsqueeze(0)  # (1, C, H, W)


def _tensor_to_b64_png(rgb_array: np.ndarray) -> str:
    """Convert RGB numpy array to base64-encoded PNG string."""
    from PIL import Image

    img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_modality(name: str, tensor: torch.Tensor) -> np.ndarray:
    """Render a modality tensor to RGB numpy array."""
    renderer = MODALITY_RENDERERS.get(name)
    if renderer:
        return renderer(tensor)
    # Fallback: normalize single band
    data = tensor.cpu().numpy()
    while data.ndim > 2:
        data = data[0]
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    return (np.stack([data] * 3, axis=-1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Server info and health check."""
    uptime = round(time.time() - SERVER_STATE["started_at"], 1) if SERVER_STATE["started_at"] else 0
    return {
        "service": "TerraMind Local Server",
        "version": "1.0.0",
        "status": "ready" if SERVER_STATE["analyzer"] is not None else "no_model",
        "model": f"terramind_v1_{SERVER_STATE['model_size']}_generate" if SERVER_STATE["model_size"] else None,
        "device": SERVER_STATE["device"],
        "load_time_s": SERVER_STATE["load_time_s"],
        "requests_served": SERVER_STATE["requests_served"],
        "uptime_s": uptime,
    }


@app.get("/v1/models")
async def list_models():
    """List available TerraMind models."""
    return {
        "models": [
            {
                "id": "terramind_v1_tiny_generate",
                "name": "TerraMind v1 Tiny",
                "size": "tiny",
                "parameters": "~50M",
                "description": "Fastest, lowest quality. Good for testing.",
            },
            {
                "id": "terramind_v1_small_generate",
                "name": "TerraMind v1 Small",
                "size": "small",
                "parameters": "~200M",
                "description": "Balanced speed and quality.",
            },
            {
                "id": "terramind_v1_base_generate",
                "name": "TerraMind v1 Base",
                "size": "base",
                "parameters": "~600M",
                "description": "Best quality. Recommended for production.",
            },
        ],
        "loaded": f"terramind_v1_{SERVER_STATE['model_size']}_generate" if SERVER_STATE["model_size"] else None,
        "supported_input_modalities": ["S2L2A"],
        "supported_output_modalities": ["S1GRD", "DEM", "LULC", "NDVI"],
        "analysis_types": ["vegetation", "flood", "land_use", "full_survey"],
    }


@app.get("/v1/status")
async def model_status():
    """Get detailed model status."""
    analyzer = SERVER_STATE["analyzer"]
    if analyzer is None:
        return {"loaded": False, "message": "No model loaded"}

    return {
        "loaded": True,
        "model": f"terramind_v1_{SERVER_STATE['model_size']}_generate",
        "device": SERVER_STATE["device"],
        "load_time_s": SERVER_STATE["load_time_s"],
        "cached_pipelines": len(analyzer._models),
        "timesteps": analyzer.timesteps,
        "requests_served": SERVER_STATE["requests_served"],
    }


@app.post("/v1/load")
async def load_model(request: LoadModelRequest):
    """Load or reload the TerraMind model."""
    logger.info(f"Loading model: size={request.model_size}, timesteps={request.timesteps}")
    t0 = time.time()

    try:
        analyzer = PLNTerraMindAnalyzer(
            model_size=request.model_size,
            timesteps=request.timesteps,
        )
        load_time = time.time() - t0
        SERVER_STATE["analyzer"] = analyzer
        SERVER_STATE["model_size"] = request.model_size
        SERVER_STATE["device"] = str(analyzer.device)
        SERVER_STATE["load_time_s"] = round(load_time, 1)

        return {
            "status": "loaded",
            "model": f"terramind_v1_{request.model_size}_generate",
            "device": str(analyzer.device),
            "load_time_s": round(load_time, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    """
    Generate output modalities from input GeoTIFF.

    Send a base64-encoded GeoTIFF and receive generated modalities.
    """
    analyzer = _require_model()

    # Override timesteps if specified
    original_timesteps = analyzer.timesteps
    if request.timesteps:
        analyzer.timesteps = request.timesteps

    try:
        # Decode GeoTIFF
        geotiff_bytes = base64.b64decode(request.geotiff_b64)
        input_tensor = _load_geotiff_tensor(geotiff_bytes)
        logger.info(f"Input shape: {list(input_tensor.shape)} | "
                    f"Output: {request.output_modalities}")

        # Run inference
        t0 = time.time()
        generated = analyzer.generate(
            input_tensor,
            input_modality=request.input_modality,
            output_modalities=request.output_modalities,
            verbose=False,
        )
        inference_time = time.time() - t0
        SERVER_STATE["requests_served"] += 1

        # Build response
        result = {
            "inference_time_s": round(inference_time, 2),
            "input_shape": list(input_tensor.shape),
            "outputs": {},
        }

        for mod_name, mod_tensor in generated.items():
            mod_result = {
                "shape": list(mod_tensor.shape),
                "dtype": str(mod_tensor.dtype),
            }
            if request.return_images:
                rgb = _render_modality(mod_name, mod_tensor)
                mod_result["image_b64"] = _tensor_to_b64_png(rgb)
            result["outputs"][mod_name] = mod_result

        # Also include input preview
        if request.return_images:
            result["input_preview_b64"] = _tensor_to_b64_png(s2_to_rgb(input_tensor))

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        analyzer.timesteps = original_timesteps


@app.post("/v1/generate/upload")
async def generate_upload(
    file: UploadFile = File(...),
    output_modalities: str = Form("S1GRD,DEM,LULC"),
    return_images: bool = Form(True),
):
    """
    Generate from uploaded GeoTIFF file (multipart form).

    Alternative to /v1/generate for easier testing with curl/Postman.

    Example:
        curl -X POST http://localhost:8786/v1/generate/upload \\
             -F "file=@my_satellite.tif" \\
             -F "output_modalities=S1GRD,DEM,LULC,NDVI"
    """
    analyzer = _require_model()

    geotiff_bytes = await file.read()
    input_tensor = _load_geotiff_tensor(geotiff_bytes)
    mods = [m.strip() for m in output_modalities.split(",")]

    logger.info(f"Upload generate: {file.filename} shape={list(input_tensor.shape)} -> {mods}")

    t0 = time.time()
    generated = analyzer.generate(input_tensor, output_modalities=mods, verbose=False)
    inference_time = time.time() - t0
    SERVER_STATE["requests_served"] += 1

    result = {
        "inference_time_s": round(inference_time, 2),
        "filename": file.filename,
        "input_shape": list(input_tensor.shape),
        "outputs": {},
    }

    for mod_name, mod_tensor in generated.items():
        mod_result = {"shape": list(mod_tensor.shape)}
        if return_images:
            rgb = _render_modality(mod_name, mod_tensor)
            mod_result["image_b64"] = _tensor_to_b64_png(rgb)
        result["outputs"][mod_name] = mod_result

    if return_images:
        result["input_preview_b64"] = _tensor_to_b64_png(s2_to_rgb(input_tensor))

    return result


@app.post("/v1/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Run PLN-specific analysis on a GeoTIFF.

    Analysis types:
    - vegetation: Vegetation encroachment risk analysis (NDVI + LULC)
    - flood: Flood risk assessment (SAR + DEM)
    - land_use: Land use change monitoring (LULC)
    - full_survey: Complete multi-modal site survey (all outputs)
    """
    analyzer = _require_model()

    try:
        geotiff_bytes = base64.b64decode(request.geotiff_b64)
        input_tensor = _load_geotiff_tensor(geotiff_bytes)
        logger.info(f"Analyze: type={request.analysis_type} shape={list(input_tensor.shape)}")

        t0 = time.time()

        if request.analysis_type == "vegetation":
            result = analyzer.analyze_vegetation_risk(input_tensor)
            stats = result["statistics"]
            images = {
                "input": s2_to_rgb(input_tensor),
                "ndvi": ndvi_to_rgb(result["ndvi"]),
                "lulc": lulc_to_rgb(result["lulc"]),
            }

        elif request.analysis_type == "flood":
            result = analyzer.analyze_flood_risk(input_tensor)
            stats = result["statistics"]
            images = {
                "input": s2_to_rgb(input_tensor),
                "sar": s1_to_rgb(result["sar"]),
                "dem": dem_to_rgb(result["dem"]),
            }

        elif request.analysis_type == "land_use":
            result = analyzer.analyze_land_use(input_tensor)
            stats = result["statistics"]
            images = {
                "input": s2_to_rgb(input_tensor),
                "lulc": lulc_to_rgb(result["lulc"]),
            }

        elif request.analysis_type == "full_survey":
            result = analyzer.full_site_survey(input_tensor)
            stats = result["statistics"]
            gen = result["generated"]
            images = {
                "input": s2_to_rgb(input_tensor),
                "sar": s1_to_rgb(gen["S1GRD"]),
                "dem": dem_to_rgb(gen["DEM"]),
                "lulc": lulc_to_rgb(gen["LULC"]),
                "ndvi": ndvi_to_rgb(gen["NDVI"]),
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown analysis type: {request.analysis_type}. "
                       f"Use: vegetation, flood, land_use, full_survey",
            )

        inference_time = time.time() - t0
        SERVER_STATE["requests_served"] += 1

        response = {
            "analysis_type": request.analysis_type,
            "inference_time_s": round(inference_time, 2),
            "statistics": stats,
        }

        if request.return_images:
            response["images"] = {
                name: _tensor_to_b64_png(rgb) for name, rgb in images.items()
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    analysis_type: str = Form("full_survey"),
    return_images: bool = Form(True),
):
    """
    Run analysis on uploaded GeoTIFF file (multipart form).

    Example:
        curl -X POST http://localhost:8786/v1/analyze/upload \\
             -F "file=@aceh_nagan_raya_s2l2a.tif" \\
             -F "analysis_type=vegetation"
    """
    geotiff_bytes = await file.read()
    geotiff_b64 = base64.b64encode(geotiff_bytes).decode("utf-8")
    req = AnalyzeRequest(
        geotiff_b64=geotiff_b64,
        analysis_type=analysis_type,
        return_images=return_images,
    )
    return await analyze(req)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("TERRAMIND_PORT", "8786"))
    host = os.environ.get("TERRAMIND_HOST", "0.0.0.0")

    print(f"""
╔══════════════════════════════════════════════════╗
║          TerraMind Local Server v1.0             ║
║                                                  ║
║  Like Ollama, but for geospatial AI              ║
╠══════════════════════════════════════════════════╣
║  Model:   terramind_v1_{MODEL_SIZE}_generate{' ' * (16 - len(MODEL_SIZE))}║
║  Device:  {str(select_device()):40s}  ║
║  Port:    {str(port):40s}  ║
║                                                  ║
║  API Docs: http://{host}:{port}/docs{' ' * (22 - len(str(port)))}║
╚══════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "pln_demo.terramind_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
