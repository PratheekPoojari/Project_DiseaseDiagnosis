"""
File: src/fusion/fuse.py
Purpose: The Late Fusion module that mathematically combines the probabilities from 
         the NLP (text) and DIP (image) branches to create a single, highly accurate prediction.
Why we need it: A multimodal system requires a way to weigh and merge decisions. 
                By relying on probabilities rather than hard labels, the system can 
                correct itself (e.g., if the text model is 51% sure it's Eczema, but the 
                image model is 95% sure it's Psoriasis, the 60/40 weighted fusion will 
                correctly output Psoriasis).
"""

CONFIDENCE_THRESHOLD = 0.40

def fuse_predictions(text_result: dict = None, image_result: dict = None, 
                     weight_image: float = 0.6, weight_text: float = 0.4) -> dict:
    """
    Takes the structured output dictionaries from the NLP and DIP prediction scripts
    and returns a final combined prediction.
    
    Expected behavior:
    - Text only -> Returns text_result
    - Image only -> Returns image_result
    - Both -> Mathematically fuses the probabilities based on the weights.
    """
    
    # 1. Single Modality Fallbacks
    if text_result and not image_result:
        return text_result
        
    if image_result and not text_result:
        return image_result
        
    if not text_result and not image_result:
        return {
            "status": "error",
            "message": "No input provided to the fusion layer.",
            "prediction": None,
            "probabilities": None
        }

    # 2. Check for errors in the incoming branches
    if text_result["status"] == "error":
        return image_result
    if image_result["status"] == "error":
        return text_result

    # 3. Late Fusion Math
    # We iterate over all the classes found in the image result (we assume 
    # both models output the exact same 10 clean class keys, which we ensured!)
    fused_probabilities = {}
    
    text_probs = text_result.get("probabilities", {})
    image_probs = image_result.get("probabilities", {})
    
    for cls_name in image_probs.keys():
        # Get the probability from each branch (default to 0.0 if something weird happens)
        p_text = text_probs.get(cls_name, 0.0)
        p_image = image_probs.get(cls_name, 0.0)
        
        # Apply the 60/40 weighting
        weighted_score = (p_image * weight_image) + (p_text * weight_text)
        fused_probabilities[cls_name] = weighted_score

    # 4. Find the winning class after fusion
    sorted_fused = sorted(fused_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_fused[0]

    # Check if the combined confidence is critically low
    status = "ok"
    if top_prob < CONFIDENCE_THRESHOLD:
        status = "low_confidence"

    return {
        "status": status,
        "message": "Fused Prediction (60% Image / 40% Text)" if status == "ok" else "Low confidence even after fusion.",
        "prediction": top_class,
        "confidence": top_prob,
        "probabilities": fused_probabilities,
        "source": "multimodal_fusion"
    }
