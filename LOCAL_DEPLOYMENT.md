# TerraMind Local Deployment Guide

## Architecture Overview

The PLN TerraMind demo supports two deployment modes:

```
Mode 1: Direct (In-Process)                Mode 2: API Server (Recommended)
========================                    ================================

┌──────────────────────┐                   ┌──────────────────────┐
│  Streamlit Dashboard │                   │  Streamlit Dashboard │
│     (app.py)         │                   │     (app.py)         │
│                      │                   │                      │
│  ┌────────────────┐  │                   │  Uses HTTP REST API  │
│  │ TerraMind Model│  │                   │  to call server      │
│  │  loaded in     │  │                   └──────────┬───────────┘
│  │  same process  │  │                              │
│  │  (~5.5 GB RAM) │  │                   ┌──────────▼───────────┐
│  └────────────────┘  │                   │  TerraMind Server    │
└──────────────────────┘                   │  (terramind_server)  │
                                           │                      │
                                           │  ┌────────────────┐  │
                                           │  │ TerraMind Model│  │
                                           │  │  loaded once   │  │
                                           │  │  shared by all │  │
                                           │  │  clients       │  │
                                           │  └────────────────┘  │
                                           └──────────────────────┘
```

**Mode 2 (API Server)** is recommended because:
- Model loads once, shared across multiple dashboards/scripts
- Dashboard stays lightweight and responsive
- Other applications (Python scripts, notebooks, curl) can use the same model
- Similar to how Ollama serves LLMs locally

---

## Quick Start

### Prerequisites

```bash
cd /path/to/terramind

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install terratorch==1.1 diffusers==0.30.0 streamlit rasterio
pip install folium streamlit-folium matplotlib Pillow
pip install fastapi uvicorn python-multipart requests
```

### Mode 1: Direct (Simple, No Server)

Just run the Streamlit dashboard directly. The model loads inside the Streamlit process.

```bash
source venv/bin/activate
streamlit run pln_demo/app.py
```

- Open http://localhost:8501
- Click "Initialize Model (Local)" in the sidebar
- First run downloads ~5.5 GB of model weights from HuggingFace Hub
- Weights are cached in `~/.cache/huggingface/hub/`

### Mode 2: API Server (Recommended)

**Step 1: Start the TerraMind server**

```bash
source venv/bin/activate
python pln_demo/terramind_server.py
```

The server will:
1. Load the TerraMind base model (~5.5 GB)
2. Warm up the inference pipeline
3. Listen on http://localhost:8786

You should see:
```
╔══════════════════════════════════════════════════╗
║          TerraMind Local Server v1.0             ║
║                                                  ║
║  Like Ollama, but for geospatial AI              ║
╠══════════════════════════════════════════════════╣
║  Model:   terramind_v1_base_generate             ║
║  Device:  mps / cuda / cpu                       ║
║  Port:    8786                                   ║
║                                                  ║
║  API Docs: http://0.0.0.0:8786/docs              ║
╚══════════════════════════════════════════════════╝
```

**Step 2: Start the dashboard (connects to server)**

```bash
# In a second terminal:
source venv/bin/activate
TERRAMIND_API_URL=http://localhost:8786 streamlit run pln_demo/app.py
```

Or set the API URL in the dashboard sidebar:
1. Check "Use API Server"
2. Enter `http://localhost:8786`
3. Click "Connect to Server"

---

## API Server Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check and server info |
| GET | `/v1/models` | List available models |
| GET | `/v1/status` | Detailed model status |
| POST | `/v1/load` | Load or reload a model |
| POST | `/v1/generate` | Generate modalities (JSON body) |
| POST | `/v1/generate/upload` | Generate modalities (file upload) |
| POST | `/v1/analyze` | Run PLN analysis (JSON body) |
| POST | `/v1/analyze/upload` | Run PLN analysis (file upload) |

Interactive API docs available at: **http://localhost:8786/docs**

### Example Usage

#### Health Check

```bash
curl http://localhost:8786/
```

Response:
```json
{
  "service": "TerraMind Local Server",
  "status": "ready",
  "model": "terramind_v1_base_generate",
  "device": "mps",
  "requests_served": 5,
  "uptime_s": 1234.5
}
```

#### Analyze a GeoTIFF (file upload)

```bash
curl -X POST http://localhost:8786/v1/analyze/upload \
     -F "file=@demo_data/indonesia/aceh_nagan_raya_s2l2a.tif" \
     -F "analysis_type=vegetation"
```

Response:
```json
{
  "analysis_type": "vegetation",
  "inference_time_s": 12.5,
  "statistics": {
    "ndvi_mean": 0.342,
    "high_vegetation_pct": 45.2,
    "lulc_distribution": {
      "Trees": 38.5,
      "Crops": 22.1,
      "Built Area": 15.3,
      "Water": 12.8
    },
    "high_risk_pct": 60.6,
    "critical_risk_pct": 38.5
  },
  "images": {
    "input": "<base64 PNG>",
    "ndvi": "<base64 PNG>",
    "lulc": "<base64 PNG>"
  }
}
```

#### Python Client

```python
from pln_demo.terramind_client import TerraMindClient

client = TerraMindClient("http://localhost:8786")

# Check server
print(client.health())

# Run vegetation analysis
result = client.analyze_vegetation(
    "demo_data/indonesia/aceh_nagan_raya_s2l2a.tif",
    return_pil=True,
)

# View statistics
print(result["statistics"])

# Display images (PIL format)
result["images"]["ndvi"].show()
result["images"]["lulc"].show()

# Full site survey
survey = client.full_survey(
    "demo_data/indonesia/suralaya_banten_s2l2a.tif",
    return_pil=True,
)
```

#### Generate Specific Modalities

```python
result = client.generate(
    "demo_data/indonesia/paiton_east_java_s2l2a.tif",
    output_modalities=["S1GRD", "DEM", "LULC", "NDVI"],
    return_pil=True,
)

# Access generated images
for mod_name, mod_data in result["outputs"].items():
    mod_data["image"].show()
```

---

## Server Configuration

The server can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TERRAMIND_MODEL_SIZE` | `base` | Model size: `tiny`, `small`, `base` |
| `TERRAMIND_TIMESTEPS` | `10` | Diffusion timesteps (more = better quality, slower) |
| `TERRAMIND_PORT` | `8786` | Server port |
| `TERRAMIND_HOST` | `0.0.0.0` | Server host |
| `TERRAMIND_AUTO_LOAD` | `true` | Auto-load model on startup |

Example with custom settings:

```bash
TERRAMIND_MODEL_SIZE=tiny \
TERRAMIND_TIMESTEPS=5 \
TERRAMIND_PORT=9000 \
python pln_demo/terramind_server.py
```

---

## Model Weights & Cache

### Where Models Are Stored

TerraMind model weights are **not** included in this repository. They are downloaded from HuggingFace Hub on first use and cached locally:

```
~/.cache/huggingface/hub/
├── models--ibm-esa-geospatial--TerraMind-1.0-base            (1.4 GB)
├── models--ibm-esa-geospatial--TerraMind-1.0-Tokenizer-DEM   (1.1 GB)
├── models--ibm-esa-geospatial--TerraMind-1.0-Tokenizer-LULC  (702 MB)
├── models--ibm-esa-geospatial--TerraMind-1.0-Tokenizer-NDVI  (1.1 GB)
├── models--ibm-esa-geospatial--TerraMind-1.0-Tokenizer-S1GRD (1.1 GB)
└── models--ibm-esa-geospatial--TerraMind-1.0-tiny            (202 MB)
                                                         Total: ~5.5 GB
```

### First-Time Setup

The first time you run the server or dashboard, it will download models automatically. This requires:
- Internet connection
- ~6 GB free disk space

Subsequent runs use the cached weights (no internet needed).

### Offline Usage

After the first download, the server works completely offline. The model weights in `~/.cache/huggingface/hub/` are all that's needed.

To pre-download models without running the server:

```python
from huggingface_hub import snapshot_download

# Download base model + all tokenizers
models = [
    "ibm-esa-geospatial/TerraMind-1.0-base",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI",
]
for model in models:
    snapshot_download(model)
```

---

## Hardware Requirements

### Minimum (CPU only)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 10 GB free (models + dependencies)
- **CPU**: Any modern x86_64 or ARM64
- **Inference time**: ~30-60 seconds per analysis

### Recommended (GPU)
- **RAM**: 16 GB
- **GPU**: NVIDIA with 4+ GB VRAM (CUDA) or Apple Silicon (MPS)
- **Inference time**: ~5-15 seconds per analysis

### Device Selection

The server automatically selects the best device:
1. **CUDA** (NVIDIA GPU) - fastest
2. **MPS** (Apple Silicon) - fast on M1/M2/M3
3. **CPU** - works everywhere, slowest

---

## Running as a System Service

### macOS (launchd)

Create `~/Library/LaunchAgents/com.terramind.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.terramind.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/terramind/venv/bin/python</string>
        <string>/path/to/terramind/pln_demo/terramind_server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/terramind</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/terramind-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/terramind-server.err</string>
</dict>
</plist>
```

Load with:
```bash
launchctl load ~/Library/LaunchAgents/com.terramind.server.plist
```

### Linux (systemd)

Create `/etc/systemd/system/terramind.service`:

```ini
[Unit]
Description=TerraMind Local Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/terramind
ExecStart=/path/to/terramind/venv/bin/python pln_demo/terramind_server.py
Restart=on-failure
RestartSec=10
Environment=TERRAMIND_MODEL_SIZE=base
Environment=TERRAMIND_PORT=8786

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl enable terramind
sudo systemctl start terramind
sudo systemctl status terramind
```

---

## File Structure

```
terramind/
├── pln_demo/
│   ├── app.py                  # Streamlit dashboard (UI)
│   ├── terramind_server.py     # FastAPI local server (like Ollama)
│   ├── terramind_client.py     # Python client for the server
│   ├── pln_analysis.py         # TerraMind model wrapper & analysis logic
│   ├── pln_config.py           # PLN site configurations
│   ├── download_indonesia_data.py  # Sentinel-2 data downloader
│   ├── demo_script.py          # CLI demo script
│   └── run_demo.py             # Batch analysis runner
├── demo_data/
│   └── indonesia/              # Real Sentinel-2 GeoTIFF data
│       ├── aceh_nagan_raya_s2l2a.tif
│       ├── sumbar_padang_s2l2a.tif
│       ├── jabar_saguling_s2l2a.tif
│       ├── suralaya_banten_s2l2a.tif
│       └── paiton_east_java_s2l2a.tif
├── configs/                    # YAML configs for analysis pipelines
├── examples/                   # Generic TerraMind example data
├── deploy/
│   ├── hf-spaces/              # Hugging Face Spaces deployment
│   └── deploy_to_hf.sh        # HF deployment script
├── .github/
│   └── workflows/
│       └── deploy-hf-spaces.yml  # Auto-deploy to HF on push
├── LOCAL_DEPLOYMENT.md         # This file
└── venv/                       # Python virtual environment
```

---

## Comparison: TerraMind Server vs Ollama

| Feature | TerraMind Server | Ollama |
|---------|-----------------|--------|
| **Model Type** | Geospatial AI (image-to-image) | Text LLM (text-to-text) |
| **Input** | GeoTIFF satellite imagery | Text prompts |
| **Output** | LULC, DEM, NDVI, SAR maps | Text completions |
| **API Style** | REST (FastAPI) | REST (Go) |
| **Port** | 8786 | 11434 |
| **Model Format** | PyTorch safetensors | GGUF quantized |
| **GPU Support** | CUDA, MPS | CUDA, Metal |
| **Auto-download** | Yes (HuggingFace Hub) | Yes (Ollama Hub) |
| **Client Library** | terramind_client.py | ollama-python |

Both follow the same pattern: **download model once, serve locally, call via REST API**.

---

## Troubleshooting

### Server won't start

```
CUDA out of memory
```
- Switch to CPU: `TERRAMIND_MODEL_SIZE=tiny python pln_demo/terramind_server.py`
- Or use the `tiny` model which needs ~500 MB

### Connection refused

```
Connection refused: http://localhost:8786
```
- Make sure the server is running in another terminal
- Check the port isn't blocked by firewall
- Try `curl http://localhost:8786/` to test

### Slow inference on CPU

- Expected: 30-60 seconds per analysis on CPU
- Use fewer timesteps: `TERRAMIND_TIMESTEPS=5`
- Use tiny model: `TERRAMIND_MODEL_SIZE=tiny`
- Best: use GPU (NVIDIA CUDA or Apple MPS)

### Model download fails

```
ConnectionError: HTTPSConnectionPool
```
- Check internet connection
- Set HF mirror: `HF_ENDPOINT=https://hf-mirror.com python pln_demo/terramind_server.py`
- Or manually download with `huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-base`
