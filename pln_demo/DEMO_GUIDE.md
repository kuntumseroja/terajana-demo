# PLN x TerraMind -- Demo Guide

> **For presenters and non-technical stakeholders.**
> This guide walks you through every step of the demo, explains what is
> happening on screen in plain language, and includes a glossary so anyone
> can follow along.

---

## Table of Contents

1. [What This Demo Does (Plain Language)](#1-what-this-demo-does)
2. [How to Start](#2-how-to-start)
3. [Dashboard Walkthrough (Step-by-Step)](#3-dashboard-walkthrough)
4. [Command-Line Demo Walkthrough](#4-command-line-demo-walkthrough)
5. [The 5 Demo Scenarios & Objectives](#5-the-5-demo-scenarios)
6. [Glossary -- Every Term Explained Simply](#6-glossary)
7. [Frequently Asked Questions](#7-faq)
8. [Technical Reference (for IT team)](#8-technical-reference)

---

## 1. What This Demo Does

Imagine you have a **single photograph** taken by a satellite flying over
a PLN power plant or transmission line. From that one photograph, the AI
can **generate** four additional "views" that normally require completely
different sensors, field surveys, or separate data purchases:

| What the AI creates | What it tells PLN | Normally requires |
|---|---|---|
| **Radar image (SAR)** | What the area looks like through clouds and at night | A different satellite (Sentinel-1) |
| **Elevation map (DEM)** | How high or low the ground is (flood risk) | LiDAR survey or aerial mapping |
| **Land cover map (LULC)** | What is on the ground: trees, water, buildings, farms | Manual classification or separate dataset |
| **Vegetation health (NDVI)** | How thick and healthy the vegetation is | Specialized analysis of satellite bands |

**In short**: 1 satellite photo in, 4 intelligence products out -- in seconds.

---

## 2. How to Start

### Option A: Web Dashboard (recommended for presentations)

Open a terminal and run:

```bash
cd /Users/priyo/Downloads/terramind
source venv/bin/activate
streamlit run pln_demo/app.py
```

A browser window will open at **http://localhost:8501**.

### Option B: Guided Script (automated slide-show style)

```bash
cd /Users/priyo/Downloads/terramind
source venv/bin/activate

# Quick run (~1 minute, lower quality -- good for testing)
python pln_demo/demo_script.py --quick

# Full quality (~10 minutes, best for presentations)
python pln_demo/demo_script.py

# Presenter mode -- pauses between each scenario for discussion
python pln_demo/demo_script.py --interactive
```

---

## 3. Dashboard Walkthrough (Step-by-Step)

### Step 1: Open the Dashboard

After running `streamlit run pln_demo/app.py`, the browser shows the
dashboard with **4 tabs** at the top:

| Tab | What it does |
|-----|-------------|
| **Overview** | Shows PLN infrastructure map, site list, and use cases |
| **Analysis** | Run AI analysis on satellite images (the main event) |
| **Generation** | Free-form: pick any combination of outputs |
| **About** | Background information on the technology |

### Step 2: Look at the Sidebar (left panel)

The sidebar has the controls. Here is what each one means:

#### Model Settings

| Control | What it means | What to choose |
|---------|---------------|----------------|
| **Model Size** | How big and accurate the AI brain is. Bigger = better results but slower. | `tiny` for quick test, `base` for best results |
| **Diffusion Timesteps** | How many "refinement passes" the AI makes when creating an image. More passes = cleaner result, like an artist doing more drafts. | `2` for fast preview, `10` for good quality, `30` for best |
| **Device** | What hardware runs the AI. Shown automatically. | `cuda` = NVIDIA GPU (fastest), `mps` = Apple chip (good), `cpu` = any computer (slowest) |

#### Data Source

| Control | What it means |
|---------|---------------|
| **Example Data** | Pre-loaded satellite images that come with the demo |
| **Upload GeoTIFF** | Use your own satellite image file (must be Sentinel-2 format) |
| **Load Example** button | Click to load the selected sample image into the system |

#### Initialize Model button

Click this to **load the AI model into memory**. This downloads the model
weights the first time (~500 MB) and takes 10-30 seconds. You only need
to do this once per session.

### Step 3: Load Data + Initialize Model

1. In the sidebar, keep **"Example Data"** selected
2. Pick any file from the dropdown (e.g., `282D_485L_3_3.tif`)
3. Click **"Load Example"**
4. Click **"Initialize Model"**
5. Wait for the green "Model loaded!" message

### Step 4: Explore the Overview Tab

This tab shows:

- **Top metrics**: Number of monitored sites, total power capacity
- **Site table**: List of 9 PLN power plants with type, capacity, region
- **Interactive map**: Click markers to see plant details. Color-coded:
  - Red = Coal, Blue = Hydro, Orange = Gas, Green = Geothermal
- **Use cases**: Expandable cards explaining each monitoring application

**Presenter tip**: Use the map to show the geographic spread across
Java-Bali, Sumatra, Kalimantan, and Sulawesi.

### Step 5: Run an Analysis

Switch to the **"Analysis"** tab. You will see:

1. A preview of the satellite image you loaded
2. A dropdown to pick which analysis to run
3. A big blue **"Run Analysis"** button

#### Available Analyses

| Analysis | What it answers | What the AI generates |
|----------|----------------|----------------------|
| **Vegetation Encroachment** | "Are trees growing dangerously close to our power lines?" | Vegetation index + Land cover + Risk map |
| **Flood Risk Assessment** | "Could this substation flood during monsoon?" | Radar image + Elevation map + Flood susceptibility |
| **Land Use Change** | "Is urbanization expanding near our infrastructure?" | Land cover classification map |
| **Full Multi-Modal Survey** | "Give me everything about this site" | All 4 outputs at once |

#### Reading the Results

After clicking "Run Analysis", you will see:

- **Metric cards** at the top: Key numbers summarizing the finding
- **Alert banner**: Color-coded assessment
  - Red = HIGH RISK, action needed
  - Yellow = MODERATE, schedule attention
  - Green = LOW RISK, area is fine
- **Side-by-side images**: Original photo + AI-generated maps
- **Charts**: Bar/pie charts showing land cover breakdown

### Step 6: Try the Generation Tab

This is the "freestyle" mode. You can:

1. Pick any combination of output types (SAR, DEM, LULC, NDVI)
2. Click **"Generate"**
3. See the original image alongside all generated outputs

**Presenter tip**: This is the "wow moment" -- show how 1 image
produces multiple completely different data products.

### Step 7: Show the About Tab

Quick reference for stakeholders who want to understand the technology
and see links to the research paper and model repository.

---

## 4. Command-Line Demo Walkthrough

The command-line demo (`demo_script.py`) runs all 5 scenarios
automatically and saves images to `pln_demo/demo_output/`.

### What Happens When You Run It

```
1. Banner appears: "PLN x TerraMind"
2. Shows configuration (model size, device)
3. Loads the AI model (one-time download)
4. Runs Scenario 1 -> Scenario 5 sequentially
5. Each scenario:
   - Explains the PLN problem (narration text)
   - States the objective
   - Loads satellite image
   - Runs AI inference
   - Prints metrics with color-coded risk levels
   - Saves comparison images to demo_output/
   - Gives a recommendation
6. Final summary with value propositions and next steps
```

### Command Options Explained

| Command | What it does | When to use |
|---------|-------------|-------------|
| `--quick` | Uses smallest model, fewest refinement steps | Testing, time-limited demos |
| `--model-size base` | Uses the best model | Formal presentations |
| `--timesteps 10` | 10 refinement passes (good balance) | Default, recommended |
| `--scenario 2` | Run only scenario 2 (flood risk) | When audience cares about specific topic |
| `--interactive` | Pauses after each scenario, press Enter to continue | Live presentations with discussion |

---

## 5. The 5 Demo Scenarios

### Scenario 1: Vegetation Encroachment Risk

> **"Are trees threatening our 500kV transmission lines?"**

| | |
|---|---|
| **The problem** | Tropical trees grow fast. When branches touch 500kV lines, it causes electrical flashovers (like lightning), wildfires, or blackouts affecting millions. |
| **Today's solution** | Helicopter patrol at Rp 15-25 million/hour. Expensive, limited coverage, weather-dependent. |
| **AI solution** | Satellite scans the entire corridor. AI creates a vegetation health map (NDVI) and land cover map (LULC), then calculates a risk score for each area. |
| **What you see** | 4 panels: satellite photo, vegetation map (green=dense), land type map, and a red/yellow/green risk map. |
| **Key number** | "Critical Risk %" -- the percentage of the corridor where tall trees or dense vegetation could reach power lines. |
| **Action** | If >30% critical: dispatch ground crew immediately. If 10-30%: schedule trimming. If <10%: corridor is clear. |

### Scenario 2: Flood Risk Assessment

> **"Could our coastal substation flood during monsoon season?"**

| | |
|---|---|
| **The problem** | Indonesia has 2,000+ flood events/year. PLN lost Rp 1.2 trillion to flood damage in 2024. Substations with expensive transformers sit in low-lying areas. |
| **Today's solution** | React after flooding occurs. Emergency repairs take weeks. |
| **AI solution** | From a satellite photo, AI generates a radar view (works through clouds) and an elevation map. Low-lying areas near water = high flood risk. |
| **What you see** | 4 panels: satellite photo, radar image, elevation map (brown=high, green=low), flood susceptibility (blue=danger). |
| **Key number** | "Mean Elevation" -- if below 10 meters, the area is critically vulnerable. "High Flood Risk %" shows how much area could flood. |
| **Action** | If >50% high risk: install flood barriers, elevate transformers, stage mobile pumps BEFORE October monsoon. |

### Scenario 3: Urban Expansion & Load Forecasting

> **"Where is urbanization outpacing our grid capacity?"**

| | |
|---|---|
| **The problem** | Indonesia urbanizes at 4.1%/year. New housing and factories need power. PLN must build infrastructure BEFORE demand arrives, not after blackouts. |
| **Today's solution** | Census data (5-year lag) and manual surveys. Too slow. |
| **AI solution** | Satellite photo becomes a land cover map showing exactly what is on the ground: buildings, farms, forest, water. Track changes over time. |
| **What you see** | Satellite photo next to a color-coded land use map + bar chart showing percentages. |
| **Key number** | "Built Area %" -- high means dense urban (high electricity demand). "Agriculture %" converting to built = growth area needing new substations. |
| **Action** | Plan 150kV feeder line expansion and new substations in areas with >20% agricultural-to-urban conversion. |

### Scenario 4: Terrain & Route Planning

> **"What is the best route for our new 275kV transmission line?"**

| | |
|---|---|
| **The problem** | Building transmission lines through mountains requires knowing exact terrain. Ground surveys cost Rp 50-100 million/km and take 6+ months. |
| **Today's solution** | Slow, expensive field surveys before any route can be approved. |
| **AI solution** | From a satellite photo, AI generates an elevation map. Engineers can screen dozens of candidate routes in minutes instead of months. |
| **What you see** | Satellite photo, radar image, and elevation map with color gradient. |
| **Key number** | "Elevation Range" -- tells you how hilly the route is. Flat = easy, steep = expensive. "Elevation Variance" -- high variance means difficult tower placement. |
| **Action** | Route along ridges (avoid valleys that flood). Use AI elevation as preliminary screen, then ground-truth only the top 2-3 candidate routes. |

### Scenario 5: Comprehensive Multi-Modal Survey

> **"Give me the complete picture for our new substation site."**

| | |
|---|---|
| **The problem** | Siting a new Rp 200+ billion substation requires terrain, vegetation, land use, flood risk, and all-weather data. Procuring all this separately takes weeks and multiple vendors. |
| **Today's solution** | Order DEM from one vendor, SAR from another, land cover from a third. Wait weeks, pay $$$. |
| **AI solution** | ONE satellite photo generates ALL FOUR data products in seconds. Complete site intelligence package. |
| **What you see** | 5 panels side-by-side: original photo + 4 generated products (SAR, DEM, LULC, NDVI). |
| **The "wow" moment** | One input, four outputs, seconds of processing. This is what replaces weeks of multi-vendor procurement. |
| **Action** | Compare candidate sites, shortlist the best 2-3, then send a field team only to those locations. |

---

## 6. Glossary -- Every Term Explained Simply

### Satellite & Data Terms

| Term | Simple Explanation | PLN Analogy |
|------|-------------------|-------------|
| **Sentinel-2** | A free European satellite that takes photographs of Earth every 5 days. Like Google Earth but updated regularly with 12 color channels (not just RGB). | Like having a free helicopter with a very good camera flying over all of Indonesia every 5 days. |
| **Sentinel-1** | A different satellite that uses radar instead of a camera. Works through clouds and at night. | Like having a helicopter with night-vision that can see through monsoon clouds. |
| **GeoTIFF** | A standard file format for satellite images. Like a JPEG but with geographic coordinates embedded so you know exactly where the image is on Earth. | Like a labeled photograph that includes GPS coordinates for every pixel. |
| **Bands / Spectral Bands** | A satellite camera captures light in multiple wavelengths (colors), including infrared that human eyes cannot see. Sentinel-2 has 12 bands. | Like having 12 different colored flashlights -- each one reveals different things about the ground below. |
| **Pixel** | One tiny square in the satellite image. For Sentinel-2, each pixel covers 10m x 10m on the ground. | One pixel = one small room on a map. |
| **224 x 224** | The image size the AI expects: 224 pixels wide by 224 pixels tall. At 10m/pixel, this covers approximately 2.2 km x 2.2 km. | A square patch of land roughly 2 km on each side. |

### AI Output Terms (The 5 Modalities)

| Term | Full Name | What It Shows | How to Read the Colors |
|------|-----------|---------------|----------------------|
| **S2 L2A** | Sentinel-2 Level 2A | The original satellite photograph in true color | Normal photo: green=vegetation, blue=water, brown=soil, gray=buildings |
| **SAR / S1 GRD** | Synthetic Aperture Radar (Sentinel-1 Ground Range Detected) | Radar view of the same area. Works through clouds, day and night. | Bright = rough surfaces (buildings, forest). Dark = smooth surfaces (water, flat ground). |
| **DEM** | Digital Elevation Model | A height map showing how high or low the ground is | Brown/dark = high ground. Green/light = low ground. Used to find flood-prone valleys. |
| **LULC** | Land Use / Land Cover | A classified map showing WHAT is on the ground | Each color = a different land type (see table below). Like a zoning map. |
| **NDVI** | Normalized Difference Vegetation Index | A measure of how green and healthy vegetation is | Green = healthy dense vegetation. Yellow = sparse. Red/brown = bare soil or water. Values range -1 to +1. Above 0.5 = very dense vegetation. |

### LULC Color Classes (What Each Color Means)

| Color | Class | Description | Risk for Power Lines |
|-------|-------|-------------|---------------------|
| Black | No Data | No information | -- |
| Blue | Water | Rivers, lakes, sea | Low (but flood concern) |
| Dark Green | Trees | Forest, tall trees | HIGH -- branches can touch lines |
| Light Green | Grass | Short grass, lawns | Low -- too short to reach lines |
| Yellow-Green | Flooded Vegetation | Swamp, mangrove, wetland | HIGH -- flooding + vegetation |
| Yellow | Crops | Agricultural fields | Medium -- seasonal tall crops |
| Purple | Scrub / Shrub | Bushes, low woody plants | Medium -- can grow into lines |
| Light Blue | Built Area | Buildings, roads, cities | Low -- managed clearance |
| Orange-Brown | Bare Ground | Exposed soil, rock | None -- clear corridor |
| White | Snow / Ice | (Not applicable in Indonesia) | -- |

### AI Model Terms

| Term | Simple Explanation | Analogy |
|------|-------------------|---------|
| **TerraMind** | The AI model created by IBM and the European Space Agency. It learned from millions of satellite images to understand Earth. | A very experienced satellite analyst who has studied millions of images and can "imagine" what the terrain, vegetation, and radar would look like just from a photograph. |
| **Foundation Model** | A large, general-purpose AI trained on massive data. Can be adapted to many tasks without starting from scratch. | Like hiring a senior engineer who already knows the fundamentals -- you just need to brief them on PLN-specific requirements. |
| **Any-to-Any Generation** | The AI can take any type of satellite data as input and generate any other type as output. Photo to radar, radar to elevation, etc. | Like a translator who can translate between any pair of languages, not just one specific pair. |
| **Diffusion / Diffusion Steps / Timesteps** | The method the AI uses to create images. It starts with random noise (TV static) and gradually refines it into a clear image. More steps = clearer result. | Like an artist starting with a rough sketch and adding detail with each pass. 2 passes = rough draft, 10 passes = good painting, 30 passes = museum quality. |
| **Tokenizer** | A component that translates between satellite data and the AI's internal language. Each data type (SAR, DEM, LULC) has its own tokenizer. | Like a specialized interpreter -- one for radar language, one for elevation language, one for land cover language. |
| **Inference** | Running the AI model to get results. "Running inference" = asking the AI to analyze an image. | "Running the analysis" -- the moment the AI is working on your image. |
| **Model Size (tiny/small/base/large)** | How many "brain cells" the AI has. Larger models make better predictions but need more computing power and time. | Like choosing between a junior analyst (tiny, fast), a senior analyst (base, recommended), or a whole department (large, best but slowest). |
| **Fine-tuning** | Teaching the pre-trained AI to be even better at a specific task using PLN's own labeled data. | Like a senior engineer doing on-the-job training specific to PLN's corridors and infrastructure. |
| **Pretrained** | The model already learned from millions of images before we use it. We don't train from zero. | The analyst already has 10 years of experience before joining PLN. |

### Analysis Metrics

| Metric | What It Means | When to Worry |
|--------|--------------|---------------|
| **NDVI Mean** | Average vegetation health across the image. Range -1 to +1. | Above 0.5 = very dense vegetation near lines |
| **High Vegetation %** | Percentage of area with NDVI > 0.5 (thick vegetation) | Above 30% in a transmission corridor |
| **Critical Risk %** | Percentage of area covered by trees or flooded vegetation | Above 20% = schedule clearing, above 40% = urgent |
| **Elevation Min/Max** | Lowest and highest points in the area (meters above sea level) | Min below 5m near coast = flood risk |
| **High Flood Risk %** | Percentage of area susceptible to flooding | Above 50% = install flood protection |
| **Built Area %** | Percentage covered by buildings and roads | High = high electricity demand zone |

### Infrastructure Terms

| Term | Meaning |
|------|---------|
| **500kV / 275kV / 150kV** | Voltage levels of transmission lines. Higher voltage = more power carried over longer distances. 500kV is the backbone (like a highway), 150kV is regional distribution (like a city road). |
| **Substation (Gardu Induk)** | A facility that transforms electricity between voltage levels. Contains expensive transformers vulnerable to flooding. |
| **Transmission corridor** | The strip of land (typically 40-60m wide) cleared beneath and around power lines. Vegetation must be kept short within this corridor. |
| **Right-of-way (ROW)** | The legal right PLN has to maintain clear space around transmission lines. |
| **Flashover** | An electrical arc that jumps from a power line to a nearby tree or object. Causes fires and blackouts. |
| **PLTU / PLTA / PLTGU / PLTP** | PLN power plant codes: U=Uap (steam/coal), A=Air (hydro), GU=Gas-Uap (combined cycle gas), P=Panas bumi (geothermal). |

### Computing Terms

| Term | Meaning | For PLN |
|------|---------|---------|
| **GPU (cuda)** | Graphics Processing Unit -- specialized chip that makes AI run 10-50x faster. | Recommended for production use. NVIDIA cards work best. |
| **MPS** | Apple Silicon's GPU acceleration. Available on newer Macs. | Good for demos on MacBook. |
| **CPU** | Regular computer processor. AI runs but is much slower. | Usable for demos, not recommended for production. |
| **Streamlit** | A tool that creates web dashboards from Python code. | The technology behind the browser-based dashboard. |
| **TerraTorch** | IBM's toolkit for working with TerraMind models. Handles model loading, data processing, and training. | The "engine" that makes everything work behind the scenes. |
| **HuggingFace** | A website where AI models are stored and shared (like GitHub for AI). Models download from here on first use. | Where the AI brain is downloaded from. First run takes a few minutes. |

---

## 7. Frequently Asked Questions

**Q: Does PLN need to buy satellite data?**
No. Sentinel-2 data is free, provided by the European Space Agency. New
images are available every 5 days for any location on Earth.

**Q: Does this work during monsoon / cloudy weather?**
The input needs a cloud-free Sentinel-2 image. However, once processed,
the AI generates a SAR radar view that represents all-weather visibility.
For persistent cloud cover, actual Sentinel-1 SAR data can be used as
input instead.

**Q: How accurate are the AI-generated outputs?**
The outputs are AI predictions, not ground truth. They are best used for
**screening and prioritization** (e.g., which 10 corridors out of 1,000
need attention first). Critical decisions should be verified with field
inspection. The demo includes "AI vs Ground Truth" comparison panels so
you can see the quality yourself.

**Q: Can this be customized for PLN specifically?**
Yes. This demo uses the "out-of-the-box" model. With PLN's own labeled
data (e.g., confirmed vegetation incidents, flood zones), the model can
be **fine-tuned** to become much more accurate for Indonesian conditions.
Fine-tuning configs are already provided in the `configs/` folder.

**Q: How large an area can be analyzed at once?**
The standard processing covers ~2.2 km x 2.2 km per patch. For larger
areas, the system uses "tiled inference" -- it processes the area in
overlapping patches and stitches them together. The demo includes a
large-tile example covering 20 km x 20 km.

**Q: What hardware does PLN need?**
- For **demos**: Any modern laptop (Mac or PC)
- For **production**: A server with an NVIDIA GPU (e.g., A100, V100) to
  process thousands of images per day
- For **cloud deployment**: AWS, GCP, or Azure with GPU instances

**Q: How fast is the analysis?**
- GPU (NVIDIA A100): ~2-5 seconds per image
- Apple M-series: ~5-20 seconds per image
- CPU only: ~1-5 minutes per image

---

## 8. Technical Reference

### Project Structure

```
terramind/
  pln_demo/
    app.py              # Streamlit web dashboard
    demo_script.py      # Guided 5-scenario CLI demo
    pln_analysis.py     # Core analysis engine
    pln_config.py       # PLN infrastructure data
    DEMO_GUIDE.md       # This guide
    demo_output/        # Generated visualizations
    output/             # CLI demo outputs
  demo_data/
    scenario_1..5/      # Organized example data per scenario
  configs/
    pln_vegetation_monitoring.yaml
    pln_flood_detection.yaml
    pln_land_use_classification.yaml
  examples/             # Raw TerraMesh example tiles
  notebooks/            # Original TerraMind Jupyter notebooks
  venv/                 # Python virtual environment
```

### Setup from Scratch

```bash
# 1. Clone repository
git clone https://github.com/IBM/terramind.git
cd terramind

# 2. Create Python 3.12 environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install terratorch==1.1 diffusers==0.30.0
pip install streamlit folium streamlit-folium rioxarray matplotlib

# 4. Run dashboard
streamlit run pln_demo/app.py

# 5. Run CLI demo
python pln_demo/demo_script.py --quick
```

### Fine-Tuning with PLN Data

```bash
terratorch fit -c configs/pln_vegetation_monitoring.yaml
terratorch fit -c configs/pln_flood_detection.yaml
terratorch fit -c configs/pln_land_use_classification.yaml
```

### Demo Output Files

All 19 output images are saved to `pln_demo/demo_output/`:

| File | Content |
|------|---------|
| `s1a_coastal_vegetation_risk.png` | 4-panel vegetation analysis |
| `s1a_validation.png` | AI vs ground truth comparison |
| `s1b_dense_vegetation_risk.png` | Dense vegetation HIGH RISK |
| `s2a_coastal_flood.png` | Coastal flood assessment |
| `s2a_validation.png` | AI vs ground truth DEM/SAR |
| `s2b_river_basin_flood.png` | River basin flood assessment |
| `s3a_jakarta_urban.png` | Jakarta urban classification |
| `s3b_periurban.png` | Peri-urban transition zone |
| `s4a_highland_terrain.png` | Highland terrain analysis |
| `s4a_dem_validation.png` | AI vs ground truth elevation |
| `s4b_mixed_terrain.png` | Mixed terrain routing |
| `s5_*_full_survey.png` | 5-panel comprehensive surveys |

### Demo Data Reference

```
demo_data/
  scenario_1_vegetation_risk/
    input/
      coastal_transmission_corridor.tif    # Coastal, 37% trees
      dense_vegetation_corridor.tif        # 88% flooded vegetation
    ground_truth/                          # Validation data

  scenario_2_flood_assessment/
    input/
      coastal_substation_area.tif          # 0-16m elevation
      river_basin_plant.tif               # 100-250m river basin

  scenario_3_urban_expansion/
    input/
      jakarta_industrial_zone.tif          # 100% built area
      peri_urban_development.tif           # Mixed development

  scenario_4_terrain_routing/
    input/
      highland_transmission_route.tif      # 1,200m plateau
      mixed_terrain_corridor.tif           # 100-250m varied

  scenario_5_comprehensive_survey/
    input/
      5 example tiles for full generation
```

### Closing Talking Points for Presentations

| # | Value Proposition | Business Impact |
|---|-------------------|-----------------|
| 1 | Replace helicopter patrols with satellite AI | Save Rp 15-25M/hr x thousands of hours/year |
| 2 | 5 intelligence products from 1 free satellite image | Eliminate multi-vendor data procurement costs |
| 3 | Screen transmission routes in seconds | Reduce 6-month survey process to days |
| 4 | All-weather radar generation | Maintain monitoring during Oct-Mar monsoon |
| 5 | Predictive flood assessment | Reduce portion of Rp 1.2T/year flood damage |

### Proposed Next Steps

1. **Pilot**: Select 1 transmission corridor for 6-month validation
2. **Fine-tune**: Label 500-1,000 PLN-specific training samples
3. **Integrate**: Connect to PLN GIS platform (ArcGIS/QGIS)
4. **Deploy**: Automated monitoring pipeline on PLN cloud
5. **Expand**: IKN (new capital) development monitoring
