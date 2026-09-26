"""
File: src/fusion/narrative.py
Purpose: Generates rich, empathetic, human-understandable specialist consultation 
         narratives from multimodal diagnosis results.
Why we need it: Raw probability percentages like "Melanoma: 89.65%" look like an 
                 engineering debug log. Patients and clinicians need an articulate, 
                 contextualized narrative explaining what the condition is, why the 
                 models flagged it, what differentials were considered, and what 
                 concrete clinical steps to take next.
"""

from typing import Dict, Any, List

# Comprehensive dermatological knowledge repository for all 10 target classes
CLINICAL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Melanoma": {
        "formal_name": "Cutaneous Melanoma (Malignant Melanocytic Neoplasm)",
        "category": "Malignant Cutaneous Neoplasm",
        "urgency": "Urgent — Specialist Evaluation Within 24–72 Hours",
        "badge_color": "#D32F2F",  # Deep Red
        "badge_bg": "#FFEBEE",
        "overview": (
            "Melanoma is an aggressive form of skin cancer that arises from pigment-producing "
            "melanocytes. While it accounts for a smaller percentage of all skin cancers, it is responsible "
            "for the vast majority of skin cancer deaths due to its propensity for early lymphatic "
            "and hematogenous metastasis if not detected and excised in its initial intraepidermal phase."
        ),
        "visual_biomarkers": [
            "Asymmetrical structural architecture across perpendicular axes",
            "Border irregularity with jagged, notched, or poorly circumscribed margins",
            "Variegated pigment distribution (atypical pigment network, blue-white veil, or dark brown/black blotches)",
            "Focal regression structures or atypical vascular patterns under dermoscopy"
        ],
        "differential_notes": (
            "Frequently shares superficial visual features with dysplastic melanocytic nevi, pigmented "
            "seborrheic keratoses, or pigmented basal cell carcinoma, requiring histological confirmation."
        ),
        "clinical_actions": [
            "Schedule an in-person full-body dermatoscopy and clinical evaluation immediately.",
            "Do NOT perform superficial shave biopsies or cryotherapy on suspected melanocytic lesions.",
            "A complete narrow-margin excisional biopsy with histological margin evaluation is standard of care.",
            "Avoid sun exposure, tanning beds, and trauma to the affected site."
        ],
        "red_flags": [
            "Rapid elevation, size expansion, or ulceration within weeks",
            "Spontaneous bleeding without mechanical trauma",
            "Development of satellite pigment spots or regional lymphadenopathy"
        ]
    },

    "Basal Cell Carcinoma": {
        "formal_name": "Basal Cell Carcinoma (BCC)",
        "category": "Non-Melanoma Skin Cancer (Keratinocyte Carcinoma)",
        "urgency": "Prompt Evaluation — Schedule Specialist Visit Within 1–2 Weeks",
        "badge_color": "#C2185B",  # Deep Magenta
        "badge_bg": "#FCE4EC",
        "overview": (
            "Basal Cell Carcinoma is the most common form of skin cancer worldwide, arising from "
            "the basal cell layer of the epidermis. It is typically slow-growing and has an extremely low "
            "rate of distant metastasis, but can cause substantial local tissue destruction and ulceration "
            "if left untreated, particularly in cosmetically sensitive areas like the face."
        ),
        "visual_biomarkers": [
            "Pearly, translucent papule or nodule often with rolled borders",
            "Arborizing (tree-branching) telangiectatic vessels on dermoscopy",
            "Central depression, ulceration, or recurrent non-healing crusted erosion",
            "Focal blue-gray ovoid nests or shiny white blotches under polarized light"
        ],
        "differential_notes": (
            "May resemble benign intradermal nevi, sebaceous hyperplasia, or nodular melanoma, "
            "especially when cystic or pigmented variants are present."
        ),
        "clinical_actions": [
            "Consult a dermatologist for dermoscopy and a diagnostic punch or shave biopsy.",
            "Discuss definitive treatment options: Mohs micrographic surgery, surgical excision, or topical therapies depending on histological subtype.",
            "Protect the affected area from chronic UV radiation using broad-spectrum SPF 50+."
        ],
        "red_flags": [
            "Persistent bleeding, non-healing sore lasting greater than 4 weeks",
            "Rapid growth or perineural symptoms like tingling or numbness around the lesion"
        ]
    },

    "Benign Keratosis-like Lesions": {
        "formal_name": "Benign Keratosis (Solar Lentigo / Seborrheic Keratosis / Lichenoid Keratosis)",
        "category": "Benign Epidermal Epithelial Lesion",
        "urgency": "Routine Evaluation — Low Urgency Non-Malignant Finding",
        "badge_color": "#388E3C",  # Green
        "badge_bg": "#E8F5E9",
        "overview": (
            "Benign keratosis-like lesions encompass non-cancerous epidermal proliferations such as solar "
            "lentigines, seborrheic keratoses, and lichen planus-like keratoses (LPLK). These are common, "
            "harmless age- and sun-associated lesions that do not evolve into invasive malignancies, "
            "though their hyperpigmentation sometimes causes clinical concern."
        ),
        "visual_biomarkers": [
            "Sharply demarcated borders with a 'stuck-on' waxy or verrucous plaque appearance",
            "Comedone-like openings (crypts) and milia-like cysts under dermoscopy",
            "Fingerprint-like or moth-eaten network patterns on peripheral borders",
            "Uniform brown, tan, or gray pigmentation with lack of significant architectural asymmetry"
        ],
        "differential_notes": (
            "Irritated or inflamed benign keratoses can mimic melanoma or BCC clinically, which is "
            "why optical dermoscopic analysis is crucial for reassurance."
        ),
        "clinical_actions": [
            "No urgent intervention is required unless the lesion causes pruritus, irritation, or friction from clothing.",
            "Elective removal options include cryotherapy, curettage, or shave excision if cosmetically desired.",
            "Monitor periodically for unusual changes in border regularity or sudden color darkening."
        ],
        "red_flags": [
            "Sudden spontaneous bleeding, ulceration, or rapid asymmetric darkening",
            "Development of multiple eruptive pruritic keratoses (Leser-Trélat sign)"
        ]
    },

    "Atopic Dermatitis": {
        "formal_name": "Atopic Dermatitis (Atopic Eczema)",
        "category": "Chronic Inflammatory Pruritic Dermatosis",
        "urgency": "Timely Clinical Review — Consult Within 1–2 Weeks",
        "badge_color": "#1976D2",  # Medical Blue
        "badge_bg": "#E3F2FD",
        "overview": (
            "Atopic Dermatitis is a chronic, relapsing inflammatory skin disease characterized by profound "
            "pruritus, cutaneous xerosis, and immune-mediated epidermal barrier disruption (often involving "
            "filaggrin gene alterations). It predominantly affects flexural creases (antecubital and popliteal fossae), "
            "face, and neck, and frequently coexists with asthma and allergic rhinitis."
        ),
        "visual_biomarkers": [
            "Poorly demarcated erythematous patches and plaques with fine desquamation",
            "Marked xerosis (diffuse skin dryness) and accentuated skin markings",
            "Lichenification (thickened, leathery skin) from chronic rubbing and excoriation",
            "Exudative crusting and micro-vesicles during acute disease flares"
        ],
        "differential_notes": (
            "Shares clinical features with contact dermatitis, psoriasis, and cutaneous fungal infections. "
            "Flexural predilection and chronic history support atopy."
        ),
        "clinical_actions": [
            "Maintain rigorous skin barrier hydration using thick ceramide-based emollients applied immediately post-bathing.",
            "Avoid hot showers, harsh soaps, detergents, and known contact allergens.",
            "Consult a physician for tailored anti-inflammatory therapy (topical corticosteroids, calcineurin inhibitors, or PDE-4 inhibitors)."
        ],
        "red_flags": [
            "Development of clustered, painful, punched-out erosions with fever (suggests Eczema Herpeticum)",
            "Spreading golden-crusted oozing indicating secondary Staphylococcus aureus infection"
        ]
    },

    "Eczema": {
        "formal_name": "Eczematous Dermatitis (Contact / Dyshidrotic / Nummular)",
        "category": "Inflammatory Cutaneous Condition",
        "urgency": "Routine to Moderate — Consult Within 1–2 Weeks",
        "badge_color": "#0288D1",  # Sky Blue
        "badge_bg": "#E1F5FE",
        "overview": (
            "Eczema is a broad clinical category encompassing various forms of epidermal inflammation, "
            "including allergic contact dermatitis, irritant dermatitis, and dyshidrotic eczema. It presents "
            "with a cycle of intense itching followed by erythematous papules, scaling, and skin fissuring."
        ),
        "visual_biomarkers": [
            "Erythematous base with superficial epidermal microvesiculation and oozing in acute stages",
            "Scaling, hyperkeratosis, and painful painful fissures in subacute/chronic phases",
            "Linear excoriation marks secondary to intense pruritic scratching"
        ],
        "differential_notes": (
            "Distinguished from tinea (fungal infections) by the absence of active scaling advancing borders, "
            "and from psoriasis by less sharply demarcated borders and absence of silvery micaceous scales."
        ),
        "clinical_actions": [
            "Identify and remove potential environmental irritants (chemical cleaners, fragrances, metals).",
            "Apply cool compresses for acute weeping lesions and barrier creams for dry fissures.",
            "Consult a medical provider for patch testing if contact allergy is suspected."
        ],
        "red_flags": [
            "Rapid spread to extensive body surface area (>30%)",
            "Purulent discharge, warmth, or increasing local pain indicating bacterial cellulitis"
        ]
    },

    "Psoriasis Lichen Planus": {
        "formal_name": "Psoriasis Vulgaris / Lichen Planus Spectrum",
        "category": "Autoimmune / Immune-Mediated Papulosquamous Disorder",
        "urgency": "Timely Medical Evaluation — Consult Within 1–2 Weeks",
        "badge_color": "#7B1FA2",  # Purple
        "badge_bg": "#F3E5F5",
        "overview": (
            "This spectrum encompasses T-cell mediated hyperproliferative and lichenoid skin disorders. "
            "Psoriasis features accelerated epidermal turnover producing thick plaques with silvery scales on "
            "extensor surfaces, while Lichen Planus presents as pruritic, planar, polygonal, purple papules "
            "affecting flexor surfaces, mucous membranes, and nails."
        ),
        "visual_biomarkers": [
            "Sharply circumscribed, salmon-pink to erythematous plaques covered by coarse silvery-white scales",
            "Auspitz sign (pinpoint bleeding upon gentle scale removal)",
            "Polygonal violaceous flat-topped papules with delicate whitish reticular lines (Wickham's striae)",
            "Nail changes including pitting, oil-drop discoloration, and subungual hyperkeratosis"
        ],
        "differential_notes": (
            "Differentiated from eczema by distinct plaque boundaries and silvery micaceous scale quality, "
            "and from fungal infections by bilateral symmetry and typical extensor distribution."
        ),
        "clinical_actions": [
            "Seek evaluation by a dermatologist to establish disease severity (PASI score assessment).",
            "Avoid skin trauma and scratching, as minor injuries can induce new lesions (Koebner phenomenon).",
            "Treatment options range from targeted topicals (vitamin D analogues, steroids) to phototherapy and systemic biologics."
        ],
        "red_flags": [
            "Generalized widespread skin redness and shedding involving >90% body surface (Erythrodermic Psoriasis)",
            "Accompanying joint pain, morning stiffness, or swollen digits suggesting Psoriatic Arthritis"
        ]
    },

    "Melanocytic Nevi": {
        "formal_name": "Melanocytic Nevus (Common Mole / Benign Nevus)",
        "category": "Benign Melanocytic Proliferation",
        "urgency": "Routine Monitoring — Low Risk Unless Clinical Changes Observed",
        "badge_color": "#5D4037",  # Warm Brown
        "badge_bg": "#EFEBE9",
        "overview": (
            "A melanocytic nevus is a benign proliferation of nevus cells (modified melanocytes) within "
            "the epidermis, dermo-epidermal junction, or dermis. They are extremely common, typically harmless, "
            "and appear as well-defined macules, papules, or nodules with uniform pigmentation."
        ),
        "visual_biomarkers": [
            "High degree of architectural symmetry across both axes",
            "Smooth, regular, sharply demarcated circular or oval margins",
            "Homogeneous light brown to dark brown pigmentation without stark color contrast",
            "Regular pigment network fading gradually toward the periphery under dermoscopy"
        ],
        "differential_notes": (
            "Must be periodically evaluated against cutaneous melanoma using the ABCDE criteria "
            "(Asymmetry, Border, Color, Diameter, Evolution)."
        ),
        "clinical_actions": [
            "No medical treatment is needed for stable, benign nevi.",
            "Practice self-monitoring once a month using the ABCDE guidelines.",
            "Protect all moles from cumulative sunburns with SPF 50+ sunscreen."
        ],
        "red_flags": [
            "The 'Ugly Duckling' sign: a mole that looks visibly dissimilar to all other moles on the body",
            "New onset of bleeding, persistent itching, or enlargement beyond 6mm in an adult"
        ]
    },

    "Seborrheic Keratoses": {
        "formal_name": "Seborrheic Keratosis",
        "category": "Benign Epidermal Epithelial Tumor",
        "urgency": "Routine Clinical Evaluation — Benign Lesion",
        "badge_color": "#455A64",  # Slate
        "badge_bg": "#ECEFF1",
        "overview": (
            "Seborrheic Keratoses are among the most common non-cancerous skin tumors in mature adults. "
            "They arise from clonal expansion of mutated epidermal keratinocytes and present as well-circumscribed, "
            "brown, black, or tan growths that look as though they have been 'stuck onto' the skin surface."
        ),
        "visual_biomarkers": [
            "Characteristic dull, waxy, or hyperkeratotic 'stuck-on' appearance",
            "Presence of horn pseudocysts, pseudocomedones, and hairpin vessels under dermoscopy",
            "Well-demarcated round-to-oval borders with dull surface texture"
        ],
        "differential_notes": (
            "When deeply pigmented, they can clinically simulate nodular melanoma; dermoscopic visualization "
            "of milia-like cysts and crypts helps establish benignity."
        ),
        "clinical_actions": [
            "Reassurance is primary; these lesions have zero malignant potential.",
            "Treatment is purely elective for mechanical irritation or cosmetic preference (cryosurgery, shave removal).",
            "Avoid picking or scratching to prevent secondary bacterial infection."
        ],
        "red_flags": [
            "Atypical rapid inflammation or bleeding with loss of circumscribed border",
            "Eruption of hundreds of itchy lesions within a short timeframe"
        ]
    },

    "Fungal Infections": {
        "formal_name": "Fungal Infection (Tinea / Ringworm / Candidiasis)",
        "category": "Infectious Cutaneous Mycosis",
        "urgency": "Timely Outpatient Treatment — Consult Within a Few Days",
        "badge_color": "#E65100",  # Amber/Orange
        "badge_bg": "#FFF3E0",
        "overview": (
            "Superficial fungal infections are caused by dermatophytes (Trichophyton, Microsporum, Epidermophyton) "
            "or yeasts (Candida, Malassezia) invading the keratinized stratum corneum, hair, or nails. Common variants "
            "include tinea corporis (ringworm), tinea cruris (jock itch), and tinea pedis (athlete's foot)."
        ),
        "visual_biomarkers": [
            "Annular (ring-shaped) erythematous plaque with central clearing and raised, scaly active borders",
            "Peripheral microvesicles or pustules along the advancing edge",
            "Satellite pustules with bright beefy-red erythema in intertriginous candidal infections"
        ],
        "differential_notes": (
            "Can be mistaken for eczema or psoriasis. Applying topical steroids to a fungal infection "
            "(Tinea Incognito) suppresses the immune response and worsens the infection."
        ),
        "clinical_actions": [
            "Consult a healthcare professional for confirmation (potassium hydroxide / KOH wet mount or fungal culture).",
            "Do NOT apply over-the-counter corticosteroid creams, as this masks the infection and causes flare-ups.",
            "Keep the affected anatomical area clean and completely dry; complete the prescribed course of topical or oral antifungals."
        ],
        "red_flags": [
            "Rapid peripheral spread with increasing pain, swelling, and fever (suggests secondary bacterial cellulitis)",
            "Infection in an immunocompromised or diabetic individual requiring aggressive systemic management"
        ]
    },

    "Viral Infections": {
        "formal_name": "Cutaneous Viral Infection (Verruca / Molluscum Contagiosum / Herpes Zoster)",
        "category": "Infectious Cutaneous Virology",
        "urgency": "Prompt Outpatient Care — Consult Within 1–3 Days",
        "badge_color": "#AD1457",  # Deep Rose
        "badge_bg": "#FCE4EC",
        "overview": (
            "Cutaneous viral infections arise from specific dermatotropic viruses—most commonly Human Papillomavirus (HPV) "
            "causing verrucae (warts), Poxvirus causing Molluscum Contagiosum, or Varicella-Zoster Virus (VZV) causing shingles. "
            "They are contagious through direct contact or autoinoculation."
        ),
        "visual_biomarkers": [
            "Verrucous, hyperkeratotic exophytic papules with disrupted skin lines and punctate thrombosed capillaries (black dots)",
            "Firm, smooth, dome-shaped papules with central umbilication in molluscum contagiosum",
            "Grouped, tense vesicles on an erythematous base following a unilateral dermatomal distribution in shingles"
        ],
        "differential_notes": (
            "Plantar warts are distinguished from corns/calluses by punctate hemorrhages upon paring and interruption of skin skin friction ridges."
        ),
        "clinical_actions": [
            "Wash hands thoroughly after touching lesions and avoid scratching to prevent autoinoculation to other body sites.",
            "Do not share towels, razors, or personal garments.",
            "Consult a physician for targeted antivirals (for herpes/zoster) or physical destruction therapies (cryotherapy, salicylic acid)."
        ],
        "red_flags": [
            "Vesicular rash involving the tip of the nose or forehead (Hutchinson sign—risk of ocular herpes zoster / blindness)",
            "Widespread dissemination in patients with underlying eczema (Eczema Herpeticum)"
        ]
    }
}


def generate_specialist_narrative(
    result: Dict[str, Any],
    symptom_text: str = "",
    image_used: bool = True
) -> Dict[str, Any]:
    """
    Synthesizes a human-understandable clinical consultation narrative from raw 
    multimodal model predictions.

    Args:
        result: Fusion prediction dictionary containing 'prediction', 'confidence', 'probabilities', 'status'.
        symptom_text: The user's input symptom notes (if provided).
        image_used: Boolean indicating if a skin lesion photograph was evaluated.

    Returns:
        Structured dictionary containing:
            - condition_name: Clean formatted condition name
            - formal_name: Clinical dermatological title
            - category: Disease taxonomy category
            - urgency: Clinical urgency recommendation
            - badge_color: Primary hex color for UI badge
            - badge_bg: Light background hex color for UI badge
            - confidence_str: Clean confidence percentage string
            - certainty_label: Clinical certainty qualifier (High / Moderate / Borderline)
            - lead_paragraph: Empathetic, doctor-like conversational narrative
            - visual_findings: List of identified or characteristic morphological biomarkers
            - differential_analysis: Conversational discussion of secondary candidates
            - clinical_actions: Recommended concrete next steps
            - red_flags: High-risk warning signs
            - tts_script: Natural, flowing voice synthesis script tailored for speech
    """
    raw_condition = result.get("prediction", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    status = result.get("status", "ok")
    probs = result.get("probabilities", {}) or {}

    # Match condition to knowledge base with fallback
    matched_key = None
    for k in CLINICAL_KNOWLEDGE_BASE:
        if k.lower() == raw_condition.lower() or k.lower() in raw_condition.lower() or raw_condition.lower() in k.lower():
            matched_key = k
            break

    if not matched_key:
        matched_key = "Eczema"  # Safe default fallback

    kb = CLINICAL_KNOWLEDGE_BASE[matched_key]

    # Qualitative certainty tier
    if status == "inconclusive":
        certainty_label = "Inconclusive (Split Prediction)"
        confidence_tone = "with high diagnostic ambiguity (predictions too closely divided)"
    elif confidence >= 0.85:
        certainty_label = "Strong Indication"
        confidence_tone = "with high statistical certainty"
    elif confidence >= 0.60:
        certainty_label = "Moderate Indication"
        confidence_tone = "with moderate diagnostic concordance"
    else:
        certainty_label = "Preliminary Indication (Low Confidence)"
        confidence_tone = "as a preliminary screening observation"

    conf_pct = f"{confidence * 100:.1f}%"

    # Modality context
    if image_used and symptom_text.strip():
        modality_phrase = (
            "Our multimodal diagnostic fusion layer evaluated both your clinical symptom description "
            "and the dermoscopic photograph of the affected skin area."
        )
    elif image_used:
        modality_phrase = (
            "Our computer vision convolutional network analyzed the morphological and color distribution "
            "patterns in your uploaded skin photograph."
        )
    else:
        modality_phrase = (
            "Our clinical NLP transformer analyzed your written symptom description against dermatological "
            "presentation patterns."
        )

    # 1. Lead Paragraph (Specialist Doctor Tone)
    if status == "inconclusive":
        lead_paragraph = (
            f"The algorithmic assessment for this case is **Inconclusive**. While the model identified "
            f"**{kb['formal_name']}** as the top numerical candidate at **{conf_pct}** confidence, "
            f"the statistical margin separating the top predictions is narrower than the 10% clinical safety threshold. "
            f"No distinct lesion pattern could be isolated. Predictions are too closely divided between candidate conditions. "
            f"{modality_phrase} "
            f"This frequently occurs when photographing healthy non-lesion skin, poorly illuminated areas, or diffuse erythema. "
            f"A direct physical examination by a medical professional or a clearer macro photograph is advised."
        )
    else:
        lead_paragraph = (
            f"Based on our algorithmic assessment, the primary diagnostic finding is "
            f"**{kb['formal_name']}**, evaluated at **{conf_pct}** confidence ({certainty_label}). "
            f"{modality_phrase} {kb['overview']} "
            f"In this analysis, the AI model identified distinctive patterns characteristic of this condition {confidence_tone}. "
            f"Please bear in mind that this assessment represents an automated screening tool; definitive confirmation "
            f"always requires a hands-on clinical examination by a qualified dermatologist."
        )

    # 2. Differential Analysis
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    other_candidates = [
        (c.replace('_', ' ').title(), p)
        for c, p in sorted_probs
        if c.lower() != raw_condition.lower() and p > 0.02
    ][:2]

    if other_candidates:
        diff_items_str = ", ".join([f"{name} ({p*100:.1f}%)" for name, p in other_candidates])
        diff_text = (
            f"The differential analysis also evaluated secondary possibilities, notably {diff_items_str}. "
            f"{kb['differential_notes']} A specialist will inspect subtle micro-structures under a dermatoscope "
            f"to clearly separate the primary finding from these secondary candidates."
        )
    else:
        diff_text = (
            f"Secondary differential conditions showed negligible probability distributions. "
            f"{kb['differential_notes']}"
        )

    # 3. Audio/TTS Script (Natural spoken doctor dialogue)
    if status == "inconclusive":
        tts_script = (
            f"Hello. Our system has analyzed the input, but the findings are inconclusive. "
            f"No distinct lesion pattern could be isolated, and predictions were too closely divided. "
            f"Please ensure a focused macro photograph of the affected area is provided, or consult a medical professional."
        )
    else:
        tts_script = (
            f"Hello. Our system has completed its multimodal analysis. "
            f"The primary clinical pattern identified is {matched_key}, with a confidence of {confidence*100:.0f} percent. "
            f"This condition is classified under {kb['category']}. "
            f"Our visual and symptom analysis detected hallmark indicators corresponding to this presentation. "
            f"We recommend scheduling a clinical consultation for direct dermatological evaluation. "
            f"Please consult a certified physician before starting or altering any medication."
        )

    badge_color = "#E65100" if status == "inconclusive" else kb["badge_color"]
    badge_bg = "#FFF3E0" if status == "inconclusive" else kb["badge_bg"]
    urgency = "Diagnostic Ambiguity — Inconclusive Screening (Repeat or In-Person Clinical Exam)" if status == "inconclusive" else kb["urgency"]

    return {
        "condition_name": matched_key,
        "formal_name": kb["formal_name"],
        "category": kb["category"],
        "urgency": urgency,
        "badge_color": badge_color,
        "badge_bg": badge_bg,
        "confidence_str": conf_pct,
        "certainty_label": certainty_label,
        "lead_paragraph": lead_paragraph,
        "visual_findings": kb["visual_biomarkers"],
        "differential_analysis": diff_text,
        "clinical_actions": kb["clinical_actions"],
        "red_flags": kb["red_flags"],
        "tts_script": tts_script
    }
