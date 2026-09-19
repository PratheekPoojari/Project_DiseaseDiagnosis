"""
File: src/dip/predict.py
Purpose: Loads the trained EfficientNet-B0 CNN and classifies a given skin image.
Why we need it: This is the inference interface for the DIP branch — used both for 
                interactive testing and imported by the Fusion Layer in the final app.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

MODEL_PATH = "models/dip/efficientnet_b0_skin.pth"
NUM_CLASSES = 10

# A model with 10 classes that's completely clueless scores ~10% per class.
# We treat anything below 30% confidence as "uncertain".
CONFIDENCE_THRESHOLD = 0.30

# This is the exact mapping from the messy Kaggle folder names to our clean class names
# We MUST use this so the CNN's output dictionary exactly matches the NLP's output dictionary!
CLASS_MAPPING = {
    "1. Eczema 1677": "eczema",
    "2. Melanoma 15.75k": "melanoma",
    "3. Atopic Dermatitis - 1.25k": "atopic_dermatitis",
    "4. Basal Cell Carcinoma (BCC) 3323": "basal_cell_carcinoma",
    "5. Melanocytic Nevi (NV) - 7970": "melanocytic_nevi",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "benign_keratosis",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "psoriasis_lichen_planus",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "seborrheic_keratoses",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "fungal_infections",
    "10. Warts Molluscum and other Viral Infections - 2103": "viral_infections"
}

# The standard transforms used during validation
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    """
    Loads the trained EfficientNet-B0 model and the raw class names from disk.
    Why we need it: Reconstructs the exact architecture and restores the learned weights.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please download it from Kaggle.")
        
    # Use CPU by default for laptop inference
    device = torch.device("cpu")
    
    # 1. Initialize the architecture
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 2. Load the weights and class names
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    raw_class_names = checkpoint['class_names']
    
    model.to(device)
    model.eval() # Set to evaluation mode!
    
    return model, raw_class_names, device


def predict_image(model, raw_class_names, device, image_path: str) -> dict:
    """
    Takes an image path and returns a structured prediction dictionary.
    Why we need it: Returns a dict mapping clean class names to probabilities, 
                    ready to be consumed by the Late Fusion layer.
    """
    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": f"Image not found: {image_path}",
            "prediction": None,
            "probabilities": None
        }

    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = image_transforms(image).unsqueeze(0) # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
            # Apply softmax to convert raw logits into percentage probabilities (0.0 to 1.0)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Map the probabilities to the clean class names
        results_dict = {}
        for i, raw_name in enumerate(raw_class_names):
            clean_name = CLASS_MAPPING.get(raw_name, raw_name.lower().replace(" ", "_"))
            results_dict[clean_name] = probabilities[i].item()

        # Sort the results to find the top prediction
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_results[0]

        # Edge Case: Low confidence
        if top_prob < CONFIDENCE_THRESHOLD:
            return {
                "status": "low_confidence",
                "message": f"Low confidence ({top_prob*100:.1f}%). The image might not be a clear skin condition.",
                "prediction": top_class,
                "confidence": top_prob,
                "probabilities": results_dict
            }

        return {
            "status": "ok",
            "message": None,
            "prediction": top_class,
            "confidence": top_prob,
            "probabilities": results_dict
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "prediction": None,
            "probabilities": None
        }


def format_result(result: dict) -> str:
    """
    Converts the structured prediction dict into a human-readable string for CLI display.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    if result["status"] == "low_confidence":
        output = f"Warning: {result['message']}\nBest guess: {result['prediction'].upper()}"
    else:
        output = f"Prediction: {result['prediction'].upper()} (Confidence: {result['confidence']*100:.2f}%)\n"
        
    output += "\nFull Class Probabilities:\n"
    
    # Sort all classes by probability
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        output += f"  - {cls}: {prob*100:.2f}%\n"
        
    return output


# ==============================================================================
# MAIN EXECUTION — Interactive CLI loop
# ==============================================================================
if __name__ == "__main__":
    print("Loading PyTorch CNN model...")
    try:
        model, raw_class_names, device = load_model()
        print("Model loaded successfully. Type 'quit' to exit.\n")

        while True:
            image_path = input("Enter path to skin image: ").strip()
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
            
            image_path = os.path.expanduser(image_path)
            
            result = predict_image(model, raw_class_names, device, image_path)
            print(f"\n{format_result(result)}\n")
            print("-" * 50)
            
    except Exception as e:
        print(f"Failed to initialize: {e}")
