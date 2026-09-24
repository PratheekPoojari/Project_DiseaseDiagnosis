"""
File: src/dip/predict.py
Purpose: Loads the trained ConvNeXt-Base CNN and classifies a given skin image.
Why we need it: This is the inference interface for the DIP branch — used both for 
                interactive testing and imported by the Fusion Layer in the final app.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Base_Weights
from PIL import Image

MODEL_PATH = "models/dip/convnext_base_skin_v2.pth"
CONFIDENCE_THRESHOLD = 0.40

# This mapping ensures the messy Kaggle folder names are converted to clean class names 
# that perfectly match the NLP output dictionary for the Fusion Layer.
CLEAN_MAPPING = {
    "1. Eczema 1677": "Eczema",
    "10. Warts Molluscum and other Viral Infections - 2103": "Viral Infections",
    "2. Melanoma 15.75k": "Melanoma",
    "3. Atopic Dermatitis - 1.25k": "Atopic Dermatitis",
    "4. Basal Cell Carcinoma (BCC) 3323": "Basal Cell Carcinoma",
    "5. Melanocytic Nevi (NV) - 7970": "Melanocytic Nevi",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "Benign Keratosis-like Lesions",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "Psoriasis Lichen Planus",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "Seborrheic Keratoses",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Fungal Infections"
}

# The identical validation transforms used in training (Resize 232 preserves aspect ratio)
val_transforms = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    """
    Loads the trained ConvNeXt-Base model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved dict which contains weights and the class names list
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    
    # Initialize ConvNeXt architecture without downloading external weights
    # (checkpoint contains all 344 layer weights trained on Kaggle)
    model = models.convnext_base(weights=None)
    
    # Replace the final classification layer to match our 10 classes
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, len(class_names))
    
    # Load our trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, device


def predict_image(model, class_names, device, image_path: str) -> dict:
    """
    Takes an image file path and returns a structured prediction dictionary.
    """
    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": f"Image not found at {image_path}",
            "prediction": None,
            "probabilities": None
        }

    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = val_transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(input_batch)
            # Convert logits to probabilities using Softmax
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
        # Map raw Kaggle class names to clean names with native Python floats for JSON serialization
        clean_classes = [CLEAN_MAPPING.get(c, c) for c in class_names]
        prob_dict = {cls: float(p) for cls, p in zip(clean_classes, probabilities)}
        results = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = results[0]

        if top_prob < CONFIDENCE_THRESHOLD:
            return {
                "status": "low_confidence",
                "message": f"Low confidence ({top_prob*100:.1f}%). Image might be unclear or unrelated.",
                "prediction": top_class,
                "confidence": float(top_prob),
                "probabilities": dict(results)
            }

        return {
            "status": "ok",
            "message": None,
            "prediction": top_class,
            "confidence": float(top_prob),
            "probabilities": dict(results)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process image: {str(e)}",
            "prediction": None,
            "probabilities": None
        }


def format_result(result: dict) -> str:
    """Converts the structured prediction dict into a human-readable string for CLI display."""
    if result["status"] == "error":
        return f"Error: {result['message']}"

    if result["status"] == "low_confidence":
        return (f"Warning: {result['message']}\n"
                f"Best guess: {result['prediction']}")

    top3 = list(result["probabilities"].items())[:3]
    output = f"Prediction: {result['prediction']} "
    output += f"(Confidence: {result['confidence']*100:.2f}%)\n"
    output += "Other possibilities:\n"
    for cls, prob in top3[1:]:
        output += f"  - {cls}: {prob*100:.2f}%\n"
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/dip/predict.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    print("Loading ConvNeXt DIP model...")
    model, class_names, device = load_model()
    print(f"Loaded! Using device: {device}")
    
    print(f"Analyzing {image_path}...\n")
    result = predict_image(model, class_names, device, image_path)
    print(f"{format_result(result)}\n")
