"""
File: src/nlp/predict.py
Purpose: Loads the trained NLP model and sentence encoder, and classifies a given symptom string.
Why we need it: This is the inference interface for the NLP branch — used both for 
                interactive testing and imported by the Fusion Layer in the final app.
"""

import joblib
import os
import sys

MODEL_PATH   = "models/nlp/svm_tfidf_model.pkl"

# A model with 10 classes that's completely clueless scores ~10% per class.
# We treat anything below 30% confidence as "uncertain" and warn the user.
CONFIDENCE_THRESHOLD = 0.30


def load_model():
    """
    Loads the trained SVM TF-IDF pipeline from disk/cache.
    Why we need it: The pipeline handles both TF-IDF vectorization and SVM classification internally.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train_nlp_model.py first.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    return model


def predict_symptom(model, text: str) -> dict:
    """
    Takes a raw symptom string and returns a structured prediction dictionary.
    Why we need it: Returns a dict (not just a string) so the Fusion Layer can 
                    directly consume the probability scores alongside the CNN's scores.

    Edge cases handled:
      - Empty / too short input → returns an error dict
      - Gibberish / unrelated input → returns a low-confidence warning dict
      - Valid symptom input → returns top prediction + all class probabilities
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

    # The pipeline handles vectorization internally!
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    top_class, top_prob = results[0]

    # Edge Case 2: Low confidence (likely unrelated / gibberish input)
    if top_prob < CONFIDENCE_THRESHOLD:
        return {
            "status": "low_confidence",
            "message": f"Low confidence ({top_prob*100:.1f}%). Are these skin-related symptoms?",
            "prediction": top_class,
            "confidence": top_prob,
            "probabilities": dict(results)
        }

    # Normal case: confident prediction
    return {
        "status": "ok",
        "message": None,
        "prediction": top_class,
        "confidence": top_prob,
        "probabilities": dict(results)
    }


def format_result(result: dict) -> str:
    """
    Converts the structured prediction dict into a human-readable string for CLI display.
    Why we need it: Keeps the predict() function clean and dict-returning (for the Fusion 
                    Layer), while giving the CLI a nicely formatted output separately.
    """
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


# ==============================================================================
# MAIN EXECUTION — Interactive CLI loop
# ==============================================================================
if __name__ == "__main__":
    print("Loading NLP model...")
    model = load_model()
    print("Ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Describe your symptoms: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            result = predict_symptom(model, user_input)
            print(f"\n{format_result(result)}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            break
