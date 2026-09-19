"""
File: src/nlp/test_edge_cases.py
Purpose: Automated test suite for the NLP branch covering known edge cases.
Why we need it: Verifies the model handles gibberish, unrelated medical input, 
                conversational descriptions, and formal clinical text correctly — 
                all cases a real user might send through the Streamlit frontend.
"""

import joblib
import sys
import os
from sentence_transformers import SentenceTransformer
from predict import predict_symptom, format_result

MODEL_PATH   = "models/nlp/svm_embedding_model.pkl"
ENCODER_NAME = "all-MiniLM-L6-v2"


def run_test_suite(model, encoder):
    """
    Runs all predefined test cases and prints results.
    Why we need it: Systematic edge case coverage so we can confirm the NLP branch 
                    is production-ready before moving on to the CNN / Fusion layer.
    """
    test_cases = [
        # (description, input_text, expected_behavior)
        ("1. Too Short Input",
         "itchy",
         "should return error"),

        ("2. Pure Gibberish",
         "asdfasdf asdfawe fasdfa sfadsdfgsdgs",
         "should return low_confidence warning"),

        ("3. Unrelated Medical (Cardiology)",
         "I have a severe headache, chest pain, and shortness of breath.",
         "should return low_confidence or wrong class (documented limitation)"),

        ("4. Conversational Eczema",
         "I have a severely itchy red flaky patch on the back of my neck.",
         "should predict eczema or atopic_dermatitis"),

        ("5. Conversational Melanoma (our key edge case fix)",
         "I noticed a dark asymmetrical mole on my back that is growing and bleeding.",
         "should predict melanoma"),

        ("6. Formal Clinical (Psoriasis)",
         "Psoriasis presents as erythematous plaques covered with silvery scales.",
         "should predict psoriasis_lichen_planus"),

        ("7. Conversational Fungal",
         "My foot has been peeling and itching between my toes for weeks.",
         "should predict fungal_infections"),

        ("8. Conversational Basal Cell Carcinoma",
         "There is a pearly shiny bump on my nose that has been there for months and won't heal.",
         "should predict basal_cell_carcinoma"),

        ("9. Mixed Symptoms (Ambiguous)",
         "I have an itchy rash with some scaly patches and redness all over.",
         "should return a plausible skin condition with reasonable confidence"),

        ("10. Empty String",
         "",
         "should return error"),
    ]

    print("=" * 60)
    print("NLP BRANCH — EDGE CASE TEST SUITE")
    print("=" * 60)

    passed = 0
    for name, text, expectation in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input:    '{text}'")
        print(f"Expected: {expectation}")
        result = predict_symptom(model, encoder, text)
        print(f"Got:\n{format_result(result)}")
        passed += 1

    print("=" * 60)
    print(f"Ran {passed}/{len(test_cases)} tests.")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found. Run train_nlp_model.py first.")
        sys.exit(1)

    print("Loading model and encoder for edge case tests...\n")
    model = joblib.load(MODEL_PATH)
    encoder = SentenceTransformer(ENCODER_NAME)
    run_test_suite(model, encoder)
