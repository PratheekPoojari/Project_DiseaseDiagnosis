import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==========================================
# 1. HYPERPARAMETERS & SETUP
# ==========================================
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HuggingFace Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class SymptomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        
        # Map string labels to integers based on unique labels in the dataset
        unique_labels = sorted(self.data['label'].unique())
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label_str = self.data.iloc[idx]['label']
        label_int = self.label_map[label_str]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_int, dtype=torch.long)
        }

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    # Load Dataset
    dataset = SymptomDataset('expanded_symptoms_dataset.csv', tokenizer)
    
    # Split 80/20 for Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    
    # Wrap in DataParallel if multiple GPUs are available (like Kaggle T4x2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed Precision Setup
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # DataParallel wrapping changes how we call the model slightly
                if isinstance(model, nn.DataParallel):
                    outputs = model.module(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                loss = outputs.loss
                logits = outputs.logits
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total:.4f}")
        
        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                if isinstance(model, nn.DataParallel):
                    outputs = model.module(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Epoch {epoch+1} | Val Acc: {val_correct/val_total:.4f}")

if __name__ == '__main__':
    train()
