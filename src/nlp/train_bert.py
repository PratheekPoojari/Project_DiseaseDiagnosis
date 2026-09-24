import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

# ==========================================
# 1. HYPERPARAMETERS & SETUP
# ==========================================
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NUM_CLASSES = 10
EPOCHS = 20
LEARNING_RATE = 3e-5
BATCH_SIZE = 16  # Per GPU

# How many of the top transformer layers to unfreeze and actually train.
# Bio_ClinicalBERT (bert-base) has 12 encoder layers total (indices 0-11).
# Unfreezing only the last few means the optimizer is adjusting a few
# million parameters instead of all ~110M -- this is the main fix for
# the overfitting you were seeing, since the model no longer has enough
# free capacity to just memorize ~1800 training examples.
N_UNFROZEN_LAYERS = 2

# Train / val / test split. Val drives early stopping and model
# selection; test is only touched once, at the very end, so the final
# reported number isn't inflated by having been used for selection.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SymptomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        unique_labels = sorted(self.data['label'].unique())
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def freeze_backbone(model, n_unfrozen_layers, rank):
    """
    Freezes every parameter in the model, then selectively unfreezes:
      - the last `n_unfrozen_layers` transformer encoder layers
      - the pooler (if the architecture has one)
      - the classification head (always trainable -- it's randomly
        initialized for this task and has to learn from scratch)

    Freezing embeddings + early/middle encoder layers keeps the model's
    general language representations intact (this is what it already
    learned from its original pretraining) and restricts fine-tuning to
    the deeper, more task-specific layers plus the new classifier.
    """
    for param in model.parameters():
        param.requires_grad = False

    # AutoModelForSequenceClassification on a BERT checkpoint exposes
    # the base transformer as `model.bert`.
    encoder_layers = model.bert.encoder.layer
    for layer in encoder_layers[-n_unfrozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    if getattr(model.bert, "pooler", None) is not None:
        for param in model.bert.pooler.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )
    return model


def train_ddp(rank, world_size):
    setup(rank, world_size)

    csv_path = 'real_symptoms_dataset.csv'
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    break

    dataset = SymptomDataset(csv_path, tokenizer)

    # =================================================================
    # ANTI-OVERFITTING: LABEL SMOOTHING + CLASS WEIGHTS
    # =================================================================
    counts = dataset.data['label'].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = class_weights.to(rank)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    # =================================================================

    # ---- three-way split: train / val / test ----
    n_total = len(dataset)
    train_size = int(TRAIN_FRAC * n_total)
    val_size = int(VAL_FRAC * n_total)
    test_size = n_total - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    if rank == 0:
        print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15
    )

    model = freeze_backbone(model, N_UNFROZEN_LAYERS, rank)

    model.to(rank)
    # find_unused_parameters=False is fine here: every parameter that's
    # still part of the graph (frozen or not) is used in every forward
    # pass, so there are no "unused" params for DDP to worry about.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Only pass trainable parameters to the optimizer -- this avoids
    # AdamW allocating momentum/variance buffers for ~100M frozen
    # parameters that will never receive a gradient.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.module.state_dict())
    patience = 4
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        if rank == 0:
            epoch_start_time = time.time()

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fct(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_correct_tensor = torch.tensor(train_correct).to(rank)
        train_total_tensor = torch.tensor(train_total).to(rank)
        train_loss_tensor = torch.tensor(train_loss).to(rank)

        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)

        epoch_train_acc = train_correct_tensor.item() / train_total_tensor.item()
        epoch_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                labels = batch['labels'].to(rank)

                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fct(outputs.logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_correct_tensor = torch.tensor(val_correct).to(rank)
        val_total_tensor = torch.tensor(val_total).to(rank)
        val_loss_tensor = torch.tensor(val_loss).to(rank)

        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        epoch_val_acc = val_correct_tensor.item() / val_total_tensor.item()
        epoch_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)

        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
            print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.module.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")

        stop_tensor = torch.tensor(1 if epochs_no_improve >= patience else 0).to(rank)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            break

    if rank == 0:
        print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.4f}")

        # ---- final, one-time evaluation on the held-out TEST set ----
        # This set was never used for early stopping or model
        # selection, so this is the honest generalization number.
        print("\nGenerating Per-Class Classification Report on TEST set (best model)...")
        model.module.load_state_dict(best_model_wts)
        model.eval()

        final_preds = []
        final_labels = []

        test_loader_single = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for batch in test_loader_single:
                outputs = model.module(batch['input_ids'].to(rank), attention_mask=batch['attention_mask'].to(rank))
                final_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                final_labels.extend(batch['labels'].numpy())

        idx_to_class = {v: k for k, v in dataset.label_map.items()}
        target_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
        print("\n" + classification_report(final_labels, final_preds, target_names=target_names))

        save_path = '/kaggle/working/bio_clinical_bert_frozen.pth' if os.path.exists('/kaggle/working') else 'bio_clinical_bert_frozen.pth'
        torch.save({
            'model_state_dict': best_model_wts,
            'label_map': dataset.label_map,
            'model_name': MODEL_NAME,
            'n_unfrozen_layers': N_UNFROZEN_LAYERS,
        }, save_path)
        print(f"Best model successfully saved to {save_path}")

    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Starting DDP with {world_size} GPUs...")
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("DDP requires at least 2 GPUs. Falling back to single GPU.")
        train_ddp(0, 1)
