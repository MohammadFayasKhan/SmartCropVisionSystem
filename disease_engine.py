"""
disease_engine.py
-----------------
Agricultural disease knowledge base and threshold-based detection engine.

Sensor inputs:
    temperature   → °C from DHT11
    humidity      → % from DHT11
    rain          → 1 = raining / 0 = dry  (YL-83 binary digital output)
    soil_moisture → 0-100% from soil sensor

NOTE: The rain parameter is now binary (0 or 1), NOT millimetres.
      All disease conditions that previously used rainfall thresholds
      in mm have been translated to binary logic:
          rain == 1  → rain currently detected (wet conditions)
          rain == 0  → no rain currently detected (dry conditions)
"""

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "WATCH": 3}

SEVERITY_COLORS = {
    "CRITICAL": "#ef233c",
    "HIGH":     "#ff6b35",
    "MODERATE": "#ffd166",
    "WATCH":    "#4cc9f0",
}

DISEASE_DATABASE = [
    {
        "id"        : "late_blight",
        "name"      : "Late Blight",
        "type"      : "Fungal",
        "severity"  : "CRITICAL",
        "trigger"   : "Humidity > 85% and Temperature 10°C to 25°C",
        "condition" : lambda t, h, rain, sm: h > 85 and 10 <= t <= 25,
        "symptoms"  : "Brown-black lesions on leaves with white mold at edges. "
                      "Affected tissue collapses rapidly. Strong unpleasant odour.",
        "pesticide" : "Mancozeb 75% WP : 2.5 g/L water. OR Cymoxanil + Mancozeb (Curzate M8) 2.5 g/L. Spray every 7 days.",
        "technique" : "Remove infected plant parts immediately. Avoid overhead irrigation. "
                      "Increase row spacing for air circulation. Apply fungicide before rain, not after.",
    },
    {
        "id"        : "powdery_mildew",
        "name"      : "Powdery Mildew",
        "type"      : "Fungal",
        "severity"  : "HIGH",
        "trigger"   : "Humidity 60-80% and Temperature 20°C to 30°C",
        "condition" : lambda t, h, rain, sm: 60 <= h <= 80 and 20 <= t <= 30,
        "symptoms"  : "White powdery coating on upper leaf surface. Leaves curl and turn yellow. Stunted growth.",
        "pesticide" : "Sulphur 80% WP : 2 g/L. OR Hexaconazole (Contaf 5 EC) 1 mL/L. Apply at first sight of patches.",
        "technique" : "Improve ventilation. Avoid excess nitrogen fertilizer. Remove infected leaves. Plant resistant varieties.",
    },
    {
        "id"        : "gray_mold",
        "name"      : "Gray Mold (Botrytis)",
        "type"      : "Fungal",
        "severity"  : "HIGH",
        "trigger"   : "Humidity > 90% and Temperature 15°C to 25°C",
        "condition" : lambda t, h, rain, sm: h > 90 and 15 <= t <= 25,
        "symptoms"  : "Grey fuzzy mold on flowers, fruits and leaves. Brown water-soaked lesions. Flowers rot rapidly.",
        "pesticide" : "Iprodione (Rovral 50 WP) 2 g/L. OR Carbendazim 50% WP 1 g/L.",
        "technique" : "Reduce humidity immediately. Prune dense foliage. Avoid wetting foliage. Remove all fallen plant matter.",
    },
    {
        "id"        : "anthracnose",
        "name"      : "Anthracnose",
        "type"      : "Fungal",
        "severity"  : "HIGH",
        "trigger"   : "Humidity > 80% and Temperature 24°C to 32°C and Rain Detected",
        "condition" : lambda t, h, rain, sm: h > 80 and 24 <= t <= 32 and rain == 1,
        "symptoms"  : "Dark sunken circular spots on fruits and leaves. Pink-orange spore masses in wet conditions.",
        "pesticide" : "Copper Oxychloride 50% WP : 3 g/L. OR Azoxystrobin (Amistar) 1 mL/L. Spray before and after rainfall.",
        "technique" : "Use drip irrigation instead of sprinkler. Prune for air flow. Apply pre-harvest fungicide sprays.",
    },
    {
        "id"        : "downy_mildew",
        "name"      : "Downy Mildew",
        "type"      : "Fungal",
        "severity"  : "HIGH",
        "trigger"   : "Humidity > 85% and Temperature 10°C to 18°C",
        "condition" : lambda t, h, rain, sm: h > 85 and 10 <= t <= 18,
        "symptoms"  : "Yellow angular patches on upper leaf, grey-purple fuzz below. Leaves dry and curl.",
        "pesticide" : "Metalaxyl + Mancozeb (Ridomil Gold MZ) 2.5 g/L. OR Cymoxanil 8% + Mancozeb 64% WP 2 g/L.",
        "technique" : "Avoid evening irrigation. Apply potassium phosphonate to strengthen cell walls. Improve field drainage.",
    },
    {
        "id"        : "bacterial_blight",
        "name"      : "Bacterial Blight",
        "type"      : "Bacterial",
        "severity"  : "CRITICAL",
        "trigger"   : "Temperature > 30°C and Humidity > 75% and Rain Detected",
        "condition" : lambda t, h, rain, sm: t > 30 and h > 75 and rain == 1,
        "symptoms"  : "Water-soaked angular lesions that turn yellow then brown. Bacterial ooze from stem wounds. Wilting.",
        "pesticide" : "Copper Hydroxide (Kocide 77 WP) 3 g/L. OR Streptomycin Sulphate + Tetracycline (Plantomycin) 0.5 g/L.",
        "technique" : "Avoid working when plants are wet. Disinfect tools with 1% bleach. Remove and burn infected plants.",
    },
    {
        "id"        : "root_rot",
        "name"      : "Root Rot (Pythium)",
        "type"      : "Oomycete",
        "severity"  : "CRITICAL",
        "trigger"   : "Rain Detected and Soil Moisture > 85%",
        "condition" : lambda t, h, rain, sm: rain == 1 and sm > 85,
        "symptoms"  : "Yellowing and wilting despite adequate moisture. Dark mushy roots. Plant collapses even with water.",
        "pesticide" : "Metalaxyl (Ridomil) 2 mL/L as soil drench. OR Fosetyl-Al (Aliette 80 WP) 2.5 g/L at root zone.",
        "technique" : "Improve drainage immediately. Create raised beds. Install drainage channels. Reduce irrigation completely.",
    },
    {
        "id"        : "spider_mite",
        "name"      : "Spider Mite Infestation",
        "type"      : "Pest",
        "severity"  : "HIGH",
        "trigger"   : "Temperature > 32°C and Humidity < 40%",
        "condition" : lambda t, h, rain, sm: t > 32 and h < 40,
        "symptoms"  : "Fine webbing on leaf undersides. Tiny yellow-bronze stippling on leaves. Severe defoliation.",
        "pesticide" : "Abamectin (Vertimec 1.8 EC) 0.5 mL/L. OR Spiromesifen (Oberon 240 SC) 0.5 mL/L. Spray leaf undersides.",
        "technique" : "Increase humidity if possible. Release predatory mites as biological control. Remove heavily infested leaves.",
    },
    {
        "id"        : "aphid",
        "name"      : "Aphid Infestation",
        "type"      : "Pest",
        "severity"  : "MODERATE",
        "trigger"   : "Temperature 18°C to 28°C and Humidity < 50%",
        "condition" : lambda t, h, rain, sm: 18 <= t <= 28 and h < 50,
        "symptoms"  : "Clusters of small soft insects on new shoots. Sticky honeydew on leaves. Curled leaves. Sooty mould.",
        "pesticide" : "Imidacloprid (Confidor 200 SL) 0.3 mL/L. OR Dimethoate 30 EC 1.5 mL/L.",
        "technique" : "Encourage natural enemies. Apply neem oil (5 mL/L) as organic alternative. Avoid excess nitrogen.",
    },
    {
        "id"        : "heat_stress",
        "name"      : "Heat Stress",
        "type"      : "Abiotic",
        "severity"  : "HIGH",
        "trigger"   : "Temperature > 38°C",
        "condition" : lambda t, h, rain, sm: t > 38,
        "symptoms"  : "Flower drop. Fruit set failure. Leaf rolling and scorching. Wilting despite adequate soil moisture.",
        "pesticide" : "Kaolin clay particle film (Surround WP) 50 g/L to reflect heat. OR Salicylic acid 0.5 mM spray.",
        "technique" : "Provide shade nets (30-50%). Irrigate in early morning and evening. Mulch soil to reduce root temperature.",
    },
    {
        "id"        : "drought_stress",
        "name"      : "Drought Stress",
        "type"      : "Abiotic",
        "severity"  : "HIGH",
        "trigger"   : "No Rain Detected and Soil Moisture < 25%",
        "condition" : lambda t, h, rain, sm: rain == 0 and sm < 25,
        "symptoms"  : "Leaf rolling and curling. Wilting in afternoon that recovers at night. Premature flowering. Crispy leaf edges.",
        "pesticide" : "Potassium Silicate spray 3 g/L to strengthen cell walls. Anti-transpirant kaolin spray to reduce water loss.",
        "technique" : "Mulch immediately to conserve moisture. Switch to drip irrigation. Apply potassium to improve resistance.",
    },
]


def detect_diseases(temperature, humidity, rain, soil_moisture):
    """
    Run all disease conditions against the provided sensor values.

    Parameters
    ----------
    temperature   : float → °C from DHT11
    humidity      : float → % from DHT11
    rain          : int   → 1 = rain detected, 0 = no rain (YL-83 binary)
    soil_moisture : float → 0-100% from soil sensor

    Returns
    -------
    List of triggered disease dicts sorted CRITICAL → HIGH → MODERATE.
    """
    detected = []
    rain_int = int(rain)
    for disease in DISEASE_DATABASE:
        try:
            if disease["condition"](temperature, humidity, rain_int, soil_moisture):
                detected.append(disease)
        except Exception:
            pass

    detected.sort(key=lambda d: SEVERITY_ORDER.get(d.get("severity", "WATCH"), 99))
    return detected
