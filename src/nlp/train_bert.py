import os
import copy
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
EPOCHS = 8
LEARNING_RATE = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 16 per GPU on T4x2 = effective global batch of 32; fall back to 32 on CPU
BATCH_SIZE = 16 if torch.cuda.is_available() else 32
print(f"Using device: {device} | GPUs: {torch.cuda.device_count()} | Batch size: {BATCH_SIZE}")

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
    # Auto-detect Kaggle input path
    csv_path = 'expanded_symptoms_dataset.csv'
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    print(f"Found dataset at: {csv_path}")
                    break

    # Load Dataset
    dataset = SymptomDataset(csv_path, tokenizer)
    
    # Split 80/20 for Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training on {train_size} sentences, Validating on {val_size}")
    
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    
    # Wrap in DataParallel if multiple GPUs are available (like Kaggle T4x2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # We remove weight_decay to let the model fully memorize the synthetic patterns
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # We keep the scheduler but reduce warmup so it reaches max LR faster
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if isinstance(model, nn.DataParallel):
                        loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
            if scheduler:
                scheduler.step()
                
            train_loss += loss.item()
            
            # Calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_correct/train_total:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
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
                
        epoch_val_acc = val_correct/val_total
        print(f"Epoch {epoch+1} | Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
        
    print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.4f}")
    # Save the model state and the label mapping so we can decode predictions later
    save_path = '/kaggle/working/bio_clinical_bert_v2.pth' if os.path.exists('/kaggle/working') else 'bio_clinical_bert_v2.pth'
    
    torch.save({
        'model_state_dict': best_model_wts,
        'label_map': dataset.label_map,
        'model_name': MODEL_NAME
    }, save_path)
    print(f"Best model successfully saved to {save_path}")

if __name__ == '__main__':
    train()
