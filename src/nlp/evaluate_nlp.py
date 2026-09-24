"""
src/nlp/evaluate_nlp.py

Purpose:
Evaluates the trained Bio_ClinicalBERT model on the held-out test split (15% = 1,200 samples)
of real_symptoms_dataset.csv. Computes full classification metrics (Precision, Recall, F1)
and generates a high-resolution confusion matrix image for the formal project report.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nlp", "bio_clinical_bert_frozen.pth")
DATASET_PATH = os.path.join(PROJECT_ROOT, "src", "nlp", "real_symptoms_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "nlp_confusion_matrix.png")
OUTPUT_METRICS_PATH = os.path.join(OUTPUT_DIR, "nlp_metrics.json")


class SymptomDataset(Dataset):
    def __init__(self, df, label_map, tokenizer, max_length=128):
        self.data = df.reset_index(drop=True)
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label_int = self.label_map[self.data.iloc[idx]['label']]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_int, dtype=torch.long)
        }


def main():
    print("=" * 70)
    print("📊 Bio_ClinicalBERT HELD-OUT TEST EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model checkpoint
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading checkpoint from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    label_map = checkpoint['label_map']
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(label_map))]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 2. Load dataset and reproduce exact train/val/test split
    print(f"Loading dataset from: {DATASET_PATH}")
    raw_df = pd.read_csv(DATASET_PATH)

    n_total = len(raw_df)
    train_size = int(0.70 * n_total)
    val_size = int(0.15 * n_total)
    test_size = n_total - train_size - val_size

    # Reproduce the exact PyTorch random_split
    dataset_indices = list(range(n_total))
    generator = torch.Generator().manual_seed(42)
    splits = torch.utils.data.random_split(dataset_indices, [train_size, val_size, test_size], generator=generator)
    test_indices = list(splits[2])

    test_df = raw_df.iloc[test_indices].copy()
    print(f"Total dataset: {n_total} rows | Evaluated Test Set: {len(test_df)} rows")

    test_dataset = SymptomDataset(test_df, label_map, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Run Inference on Test Set
    all_preds = []
    all_labels = []

    print("Running inference across test split...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 4. Compute Metrics
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n" + report_text)

    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save metrics JSON
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved metric report JSON to: {OUTPUT_METRICS_PATH}")

    # 5. Plot & Save High-Resolution Confusion Matrix
    print("Generating confusion matrix visualization...")
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Bio_ClinicalBERT Symptom Classification — Confusion Matrix\n(Held-out Test Split: 1,200 Samples)', fontsize=14, pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Normalize matrix for percentage text display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt = f"{count}\n({pct:.1f}%)" if count > 0 else "0"
            plt.text(j, i, txt,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

    plt.ylabel('True Clinical Condition', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Condition', fontsize=12, labelpad=10)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved high-resolution confusion matrix plot to: {OUTPUT_PLOT_PATH}")
    print("=" * 70)
    print("🎉 NLP EVALUATION ARTIFACTS GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
