"""
File: src/nlp/predict.py
Purpose: Loads the trained Bio_ClinicalBERT model and classifies a given symptom string.
Why we need it: This is the inference interface for the NLP branch — used both for 
                interactive testing and imported by the Fusion Layer in the final app.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/nlp/bio_clinical_bert_frozen.pth"
CONFIDENCE_THRESHOLD = 0.40

# Wrap the model and tokenizer together so load_model() can return a single object
class NLPModelWrapper:
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        # Invert label map for inference (idx -> string)
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.device = device


def load_model():
    """
    Loads the trained PyTorch BERT model and HuggingFace Tokenizer.
    """
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        alt_path = "models/nlp/bio_clinical_bert_frozen.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Error: Model not found at {MODEL_PATH} (or {alt_path}).")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved dict which contains weights, label map, and model name
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_name = checkpoint['model_name']
    label_map = checkpoint['label_map']
    
    # Initialize tokenizer and architecture (try local cache first for offline resilience)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
    
    # Load weights and set to eval mode
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return NLPModelWrapper(model, tokenizer, label_map, device)


def predict_symptom(wrapper: NLPModelWrapper, text: str) -> dict:
    """
    Takes a raw symptom string and returns a structured prediction dictionary.
    """
    text = text.strip()

    # Edge Case 1: Empty or too short
    if not text or len(text) < 10:
        return {
            "status": "error",
            "message": "Input too short. Please describe your symptoms in detail.",
            "prediction": None,
            "probabilities": None
        }

    # Tokenize input
    encoding = wrapper.tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(wrapper.device)
    attention_mask = encoding['attention_mask'].to(wrapper.device)

    # Forward pass without gradients
    with torch.no_grad():
        outputs = wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Convert logits to probabilities using Softmax
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Map probabilities to class names with native Python floats for JSON serialization
    classes = [wrapper.idx_to_label[i] for i in range(len(wrapper.label_map))]
    prob_dict = {cls: float(p) for cls, p in zip(classes, probabilities)}
    results = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = results[0]

    # Edge Case 2: Low confidence
    if top_prob < CONFIDENCE_THRESHOLD:
        return {
            "status": "low_confidence",
            "message": f"Low confidence ({top_prob*100:.1f}%). Are these skin-related symptoms?",
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


def format_result(result: dict) -> str:
    """Converts the structured prediction dict into a human-readable string for CLI display."""
    if result["status"] == "error":
        return f"Error: {result['message']}"

    if result["status"] == "low_confidence":
        return (f"Warning: {result['message']}\n"
                f"Best guess: {result['prediction'].upper()}")

    top3 = list(result["probabilities"].items())[:3]
    output = f"Prediction: {result['prediction'].upper()} "
    output += f"(Confidence: {result['confidence']*100:.2f}%)\n"
    output += "Other possibilities:\n"
    for cls, prob in top3[1:]:
        output += f"  - {cls}: {prob*100:.2f}%\n"
    return output


if __name__ == "__main__":
    print("Loading BERT NLP model...")
    wrapper = load_model()
    print(f"Loaded! Using device: {wrapper.device}")
    print("Ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Describe your symptoms: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            result = predict_symptom(wrapper, user_input)
            print(f"\n{format_result(result)}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            break
