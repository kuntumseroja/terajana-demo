# PLN TerraMind Demo Data

Organized demo data for 5 PLN infrastructure monitoring scenarios.
Each scenario directory contains `input/` (Sentinel-2 L2A optical) and
`ground_truth/` (reference modalities for validation).

## Scenarios

| # | Scenario | PLN Relevance | Input Samples |
|---|----------|---------------|---------------|
| 1 | Vegetation Encroachment Risk | Transmission line corridor safety | 2 |
| 2 | Flood Risk Assessment | Substation & power plant protection | 2 |
| 3 | Urban Expansion Monitoring | Grid planning & load forecasting | 2 |
| 4 | Terrain & Route Planning | New transmission line routing | 2 |
| 5 | Comprehensive Multi-Modal Survey | Full site assessment | 5 |

## Data Format

- **Input**: Sentinel-2 Level 2A GeoTIFF (12 spectral bands, 224x224 pixels)
- **Ground Truth**: DEM, LULC, NDVI, SAR reference data from TerraMesh dataset

## Usage

```bash
source venv/bin/activate
python pln_demo/demo_script.py              # Run full guided demo
python pln_demo/demo_script.py --scenario 1 # Run single scenario
python pln_demo/demo_script.py --quick       # Quick mode (tiny model)
```
