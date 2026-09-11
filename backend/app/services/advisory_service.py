"""
Agronomic Advisory Service for Smart Plant Intelligence System.
Synthesizes crop condition, pathology classification, and validated triage damage percentage
into actionable, responsible, evidence-based agricultural recommendations.
Enforces low-confidence safeguards to prevent premature or inappropriate chemical interventions.
"""

from typing import Dict, Any, Optional

ADVISORY_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "late_blight": {
        "immediate_action": "Isolate affected plants immediately. Prune and safely bag diseased foliage during dry weather to prevent spore spread.",
        "treatment_protocol": "Consult local agricultural extension specialists for regionally approved preventative foliar protectants and resistance schedules.",
        "cultural_practices": "Halt overhead irrigation immediately. Switch to drip lines and ensure adequate plant spacing for canopy ventilation."
    },
    "early_blight": {
        "immediate_action": "Scout and remove lower infected foliage exhibiting concentric rings before pathogen transmission occurs.",
        "treatment_protocol": "Consult extension services for integrated pest management (IPM) approved foliar protectants or bio-control options.",
        "cultural_practices": "Apply soil mulch to prevent soil-borne spores from splashing onto lower leaves during rainfall."
    },
    "powdery_mildew": {
        "immediate_action": "Prune severely colonized leaves to improve inner-canopy sunlight penetration.",
        "treatment_protocol": "Review organic horticultural oils, sulfur-based products, or registered IPM protectants with your local extension service.",
        "cultural_practices": "Improve passive greenhouse ventilation and moderate nitrogen fertilization to avoid excessive soft succulent growth."
    },
    "apple_scab": {
        "immediate_action": "Inspect young leaves and fruit clusters for olive-green velvety lesions.",
        "treatment_protocol": "Consult orchard management guidelines for timely protectant applications during primary ascospore discharge windows.",
        "cultural_practices": "Rake and compost or shred fallen orchard leaf litter in autumn to disrupt the pathogen's overwintering cycle."
    },
    "black_rot": {
        "immediate_action": "Excise mummified fruit clusters and cankered vines promptly.",
        "treatment_protocol": "Follow regional viticulture extension schedules for bloom and post-bloom canopy management.",
        "cultural_practices": "Prune vines to establish an open, well-aerated canopy promoting rapid drying after rain events."
    },
    "rust": {
        "immediate_action": "Examine leaf undersides for powdery orange-brown pustules and monitor adjacent plants.",
        "treatment_protocol": "Refer to cereal or crop-specific IPM protocols for early-season threshold-based protectant applications.",
        "cultural_practices": "Manage volunteer plants and alternate host weeds in field borders to break pathogen life cycles."
    },
    "bacterial_spot": {
        "immediate_action": "Avoid working in the field while foliage is wet to prevent mechanical spreading of bacterial ooze.",
        "treatment_protocol": "Consult agricultural extension for registered copper-based bactericides or biological alternatives.",
        "cultural_practices": "Utilize certified disease-free seeds and enforce crop rotation with non-host species."
    },
    "viral": {
        "immediate_action": "Rogue and safely remove stunted or mottled plants; viral infections cannot be cured with chemical sprays.",
        "treatment_protocol": "Focus on vector management (aphids, thrips, whiteflies) following regional threshold-based IPM guidance.",
        "cultural_practices": "Deploy insect exclusion netting and reflective mulches to minimize vector landing in early growth stages."
    },
    "healthy": {
        "immediate_action": "Maintain routine field scouting schedule and monitor general canopy vigor.",
        "treatment_protocol": "No chemical intervention needed. Avoid unnecessary preventative pesticide applications.",
        "cultural_practices": "Maintain balanced irrigation and proper soil nutrition to preserve natural foliar vigor."
    }
}

def generate_agronomic_advisory(
    crop: str,
    condition_type: str,
    disease_name: str,
    is_infected: bool,
    damage_pct: Optional[float],
    confidence_pct: float
) -> Dict[str, str]:
    """
    Synthesizes tailored agronomic advice based on pathology and damage metrics.
    Enforces low-confidence and unverified triage guards.
    """
    is_low_conf = confidence_pct < 50.0

    if not is_infected:
        base = ADVISORY_KNOWLEDGE_BASE["healthy"]
        if is_low_conf:
            return {
                "immediate_action": f"Tentative Match ({crop.capitalize()} - Healthy Foliage, {confidence_pct:.1f}%). Model confidence is low. Inspect leaf in field under natural diffused lighting.",
                "treatment_protocol": "No chemical intervention is recommended based on preliminary low-confidence automated screening.",
                "cultural_practices": base["cultural_practices"],
                "uncertainty_guidance": "Confidence score is below 50%. While visual features closest match healthy foliage, cross-reference symptoms with adjacent leaves."
            }
        return {
            "immediate_action": f"{crop.capitalize()} foliage displays healthy characteristics. " + base["immediate_action"],
            "treatment_protocol": base["treatment_protocol"],
            "cultural_practices": base["cultural_practices"],
            "uncertainty_guidance": None
        }

    # Match key in knowledge base
    key = "early_blight"
    d_lower = disease_name.lower()
    if "late blight" in d_lower:
        key = "late_blight"
    elif "early blight" in d_lower or "blight" in d_lower:
        key = "early_blight"
    elif "mildew" in d_lower:
        key = "powdery_mildew"
    elif "scab" in d_lower:
        key = "apple_scab"
    elif "black rot" in d_lower or "rot" in d_lower:
        key = "black_rot"
    elif "rust" in d_lower:
        key = "rust"
    elif "bacterial" in d_lower:
        key = "bacterial_spot"
    elif "virus" in d_lower or "curl" in d_lower or "mosaic" in d_lower:
        key = "viral"

    base = ADVISORY_KNOWLEDGE_BASE.get(key, ADVISORY_KNOWLEDGE_BASE["early_blight"])

    # If low confidence, do NOT display panic or emergency chemical spray orders
    if is_low_conf:
        return {
            "immediate_action": (
                f"Low Confidence Screening: The system detected visual features resembling {crop.capitalize()} - {disease_name}, "
                f"but model confidence ({confidence_pct:.1f}%) is insufficient for definitive diagnosis. "
                f"Do not apply emergency chemical treatments. Inspect the canopy manually and capture a clear close-up photograph in diffuse daylight."
            ),
            "treatment_protocol": "Chemical application deferred until physical symptoms or higher confidence diagnosis is confirmed.",
            "cultural_practices": base["cultural_practices"],
            "uncertainty_guidance": (
                f"Model uncertainty is high ({100.0 - confidence_pct:.1f}%). "
                f"Verify whether environmental factors (sun glare, shadow, background clutter) affected image quality."
            )
        }

    # Confirmed diagnosis: format severity note
    if damage_pct is not None and damage_pct > 0.0:
        if damage_pct < 5.0:
            stage_note = f"Stage 1 Mild Foliar Spotting ({damage_pct:.1f}% tissue necrosis). Targeted localized treatment recommended."
        elif damage_pct <= 15.0:
            stage_note = f"Stage 2 Moderate Lesion Spread ({damage_pct:.1f}% tissue necrosis). Active canopy intervention required."
        else:
            stage_note = f"Stage 3 Severe Tissue Necrosis ({damage_pct:.1f}% tissue necrosis). Active intervention and row containment required."
    else:
        stage_note = f"Active Foliar Infection Detected ({confidence_pct:.1f}% confidence). Prompt scouting and management advised."

    uncertainty = None
    if confidence_pct < 65.0:
        uncertainty = "Diagnostic confidence is moderate. Cross-reference symptoms with local agricultural extension guidelines."

    return {
        "immediate_action": f"{stage_note} {base['immediate_action']}",
        "treatment_protocol": base["treatment_protocol"],
        "cultural_practices": base["cultural_practices"],
        "uncertainty_guidance": uncertainty
    }
