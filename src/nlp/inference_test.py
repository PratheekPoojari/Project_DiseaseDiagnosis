"""
inference_test.py

Loads the checkpoint saved by train_bert_frozen.py and runs it on
hand-written test sentences that DON'T come from the generator's
templates. This is the real generalization check: the val/test split
only proves the model can handle more sentences from the same 309-word
template vocabulary, not that it understands symptoms described the
way an actual person would type them.

Usage:
    python inference_test.py
        -> runs the built-in hand-written examples below
    python inference_test.py "your own symptom sentence here"
        -> classifies whatever you pass in
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHECKPOINT_PATH = "bio_clinical_bert_frozen.pth"

# Hand-written, deliberately phrased differently from the generator's
# templates/vocabulary -- this is the point. If the model only does
# well on generator-style phrasing and falls apart here, that's a real
# generalization gap the val/test numbers won't show you.
SANITY_CHECK_SENTENCES = [
    ("My elbow has been dry and flaky for weeks and it itches like crazy "
     "at night.", "Eczema / Atopic Dermatitis"),
    ("There's a weird ring on my leg, red around the edge but normal "
     "looking skin in the middle, and it's spreading.", "Fungal Infections"),
    ("I've got this hard little bump on my finger, feels rough, kind of "
     "like sandpaper, with tiny dark specks in it.", "Viral Infections"),
    ("This mole on my back looks off to me lately -- one side doesn't "
     "match the other and it's got a couple different colors in it.",
     "Melanoma"),
    ("Just a plain brown spot on my arm, always been round and the same "
     "shade, never bothered me.", "Melanocytic Nevi"),
    ("There's a bump near my nose that's shiny and kind of see-through "
     "looking, and it keeps scabbing over and reopening.", "Basal Cell Carcinoma"),
    ("I have these thick silvery flakes building up on my scalp and "
     "elbows that just won't quit.", "Psoriasis Lichen Planus"),
]


def load_model(checkpoint_path):
    import os
    if not os.path.exists(checkpoint_path):
        for candidate in [
            os.path.join('/kaggle/working', checkpoint_path),
            os.path.join('models/nlp', checkpoint_path),
            os.path.join('/kaggle/working/models/nlp', checkpoint_path),
            "bio_clinical_bert_v2.pth",
            "/kaggle/working/bio_clinical_bert_v2.pth",
            "models/nlp/bio_clinical_bert_v2.pth"
        ]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_map = checkpoint["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint["model_name"], num_labels=len(label_map)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, idx_to_label, device


def predict(text, model, tokenizer, idx_to_label, device, top_k=3):
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_idxs = torch.topk(probs, k=top_k)
    return [(idx_to_label[i.item()], p.item()) for p, i in zip(top_probs, top_idxs)]


def main():
    model, tokenizer, idx_to_label, device = load_model(CHECKPOINT_PATH)

    # Filter out Jupyter/IPython kernel arguments (e.g., ['-f', '/root/.../kernel-xxx.json'])
    user_args = [
        arg for arg in sys.argv[1:] 
        if not arg.startswith('-f') and not arg.endswith('.json') and not arg.startswith('--')
    ]

    if user_args:
        text = " ".join(user_args)
        results = predict(text, model, tokenizer, idx_to_label, device)
        print(f"\nInput: {text}")
        for label, prob in results:
            print(f"  {label}: {prob:.3f}")
        return

    print("Running sanity-check sentences (hand-written, not from the "
          "generator's templates)...\n")
    for text, expected in SANITY_CHECK_SENTENCES:
        results = predict(text, model, tokenizer, idx_to_label, device)
        top_label, top_prob = results[0]
        match = "✓" if expected.lower() in top_label.lower() or top_label.lower() in expected.lower() else "?"
        print(f"[{match}] Expected: {expected}")
        print(f"    Input: {text}")
        print(f"    Top predictions: {results}")
        print()


if __name__ == "__main__":
    main()
