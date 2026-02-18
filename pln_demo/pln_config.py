"""
PLN (Perusahaan Listrik Negara) Configuration
Key locations and parameters for Indonesia's electricity infrastructure monitoring.
"""

# ============================================================================
# PLN Key Infrastructure Locations (lon, lat)
# Major power plants, substations, and transmission corridors in Indonesia
# ============================================================================

PLN_SITES = {
    # Java-Bali Grid (largest interconnected grid)
    "PLTU Suralaya": {
        "lon": 106.0464, "lat": -6.0203,
        "type": "Coal Power Plant",
        "capacity_mw": 4025,
        "region": "Banten",
        "description": "Largest coal-fired power plant in Southeast Asia"
    },
    "PLTU Paiton": {
        "lon": 113.5525, "lat": -7.7197,
        "type": "Coal Power Plant",
        "capacity_mw": 4710,
        "region": "East Java",
        "description": "Major base-load power plant in East Java"
    },
    "PLTA Cirata": {
        "lon": 107.3647, "lat": -6.7217,
        "type": "Hydroelectric",
        "capacity_mw": 1008,
        "region": "West Java",
        "description": "Largest pumped-storage hydroelectric plant in Indonesia"
    },
    "PLTA Saguling": {
        "lon": 107.3094, "lat": -6.9100,
        "type": "Hydroelectric",
        "capacity_mw": 700,
        "region": "West Java",
        "description": "Hydroelectric dam on Citarum River"
    },
    "PLTGU Muara Karang": {
        "lon": 106.7666, "lat": -6.1056,
        "type": "Gas Power Plant",
        "capacity_mw": 1544,
        "region": "DKI Jakarta",
        "description": "Combined cycle gas plant serving Jakarta"
    },
    # Aceh & Sumatra Barat (with satellite data)
    "PLTU Nagan Raya": {
        "lon": 96.2000, "lat": 4.1500,
        "type": "Coal Power Plant",
        "capacity_mw": 210,
        "region": "Aceh",
        "description": "Coal-fired plant in western Aceh coast with satellite data"
    },
    "PLN Padang Hub": {
        "lon": 100.3543, "lat": -0.9471,
        "type": "Gas Power Plant",
        "capacity_mw": 150,
        "region": "Sumatra Barat",
        "description": "Distribution hub in Padang with satellite data"
    },
    # Sumatra Grid
    "PLTA Asahan": {
        "lon": 99.4536, "lat": 2.6283,
        "type": "Hydroelectric",
        "capacity_mw": 603,
        "region": "North Sumatra",
        "description": "Hydroelectric plant on Lake Toba outflow"
    },
    "PLTP Sarulla": {
        "lon": 99.0567, "lat": 2.1647,
        "type": "Geothermal",
        "capacity_mw": 330,
        "region": "North Sumatra",
        "description": "Largest single-contract geothermal plant in the world"
    },
    # Kalimantan
    "PLTU Balikpapan": {
        "lon": 116.8286, "lat": -1.2675,
        "type": "Coal Power Plant",
        "capacity_mw": 220,
        "region": "East Kalimantan",
        "description": "Power plant near IKN (new capital city)"
    },
    # Sulawesi
    "PLTA Bakaru": {
        "lon": 119.4300, "lat": -3.4500,
        "type": "Hydroelectric",
        "capacity_mw": 126,
        "region": "South Sulawesi",
        "description": "Hydroelectric plant in South Sulawesi"
    },
}

# ============================================================================
# Key Transmission Line Corridors
# ============================================================================

PLN_TRANSMISSION_CORRIDORS = {
    "Java 500kV Backbone": {
        "from": "Suralaya, Banten",
        "to": "Paiton, East Java",
        "voltage_kv": 500,
        "length_km": 900,
        "description": "Main Java-Bali 500kV backbone transmission line"
    },
    "Sumatra 275kV Backbone": {
        "from": "Banda Aceh",
        "to": "Lampung",
        "voltage_kv": 275,
        "length_km": 1800,
        "description": "Sumatra backbone transmission connecting north to south"
    },
    "Java-Bali Crossing": {
        "from": "Banyuwangi, East Java",
        "to": "Gilimanuk, Bali",
        "voltage_kv": 150,
        "length_km": 45,
        "description": "Submarine cable connecting Java and Bali grids"
    },
}

# ============================================================================
# PLN Use Cases for TerraMind Satellite Monitoring
# ============================================================================

PLN_USE_CASES = {
    "vegetation_encroachment": {
        "name": "Vegetation Encroachment Detection",
        "description": (
            "Monitor vegetation growth near transmission line corridors. "
            "Tall trees and fast-growing vegetation can cause line faults, "
            "wildfires, and power outages. Uses NDVI and LULC analysis."
        ),
        "modalities_in": ["S2L2A"],
        "modalities_out": ["NDVI", "LULC"],
        "priority": "HIGH"
    },
    "flood_risk": {
        "name": "Flood Risk Assessment",
        "description": (
            "Detect flooding near substations and power plants using SAR data. "
            "Indonesia's tropical climate and geography make infrastructure "
            "vulnerable to seasonal flooding."
        ),
        "modalities_in": ["S2L2A"],
        "modalities_out": ["S1GRD", "DEM"],
        "priority": "HIGH"
    },
    "land_use_change": {
        "name": "Land Use Change Monitoring",
        "description": (
            "Track land use changes around PLN infrastructure. Detect urban "
            "expansion, deforestation, mining activity that may affect grid "
            "planning and right-of-way management."
        ),
        "modalities_in": ["S2L2A"],
        "modalities_out": ["LULC"],
        "priority": "MEDIUM"
    },
    "terrain_analysis": {
        "name": "Terrain & Elevation Analysis",
        "description": (
            "Generate elevation models for new transmission line route planning. "
            "Essential for tower placement and clearance calculations in "
            "Indonesia's mountainous terrain."
        ),
        "modalities_in": ["S2L2A"],
        "modalities_out": ["DEM"],
        "priority": "MEDIUM"
    },
    "multi_modal_site_survey": {
        "name": "Multi-Modal Site Survey",
        "description": (
            "Comprehensive site assessment combining optical, SAR, elevation, "
            "vegetation, and land use data for new infrastructure planning "
            "or disaster response."
        ),
        "modalities_in": ["S2L2A"],
        "modalities_out": ["S1GRD", "DEM", "LULC", "NDVI"],
        "priority": "LOW"
    },
}

# ============================================================================
# LULC Class Labels (ESA WorldCover 10-class scheme used by TerraMind)
# ============================================================================

LULC_CLASSES = {
    0: "No Data",
    1: "Water",
    2: "Trees",
    3: "Grass",
    4: "Flooded Vegetation",
    5: "Crops",
    6: "Scrub/Shrub",
    7: "Built Area",
    8: "Bare Ground",
    9: "Snow/Ice",
}

LULC_RISK_FOR_TRANSMISSION = {
    "Trees": "HIGH - risk of line contact and wildfire",
    "Flooded Vegetation": "HIGH - flooding and vegetation risk",
    "Crops": "MEDIUM - seasonal growth patterns",
    "Scrub/Shrub": "MEDIUM - potential growth into lines",
    "Built Area": "LOW - urban area, managed clearance",
    "Water": "LOW - no vegetation risk, but flood concern",
    "Grass": "LOW - minimal risk to overhead lines",
    "Bare Ground": "NONE - clear corridor",
}
