import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Base_Weights
from tqdm import tqdm
import gc

# ---------------------------------------------------------
# 1. PATH RESOLUTION (Auto-detects Kaggle Paths)
# ---------------------------------------------------------
base_input = '/kaggle/input'
DATA_DIR = None
for root, dirs, files in os.walk(base_input):
    if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
        DATA_DIR = os.path.dirname(root)
        break

if not DATA_DIR:
    raise FileNotFoundError("Could not find dataset in /kaggle/input.")

print(f"Dataset found at: {DATA_DIR}")

# ---------------------------------------------------------
# 2. HEAVY AUGMENTATION (RandAugment)
# ---------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ---------------------------------------------------------
# 3. DATA LOADERS & 80/20 SPLIT
# ---------------------------------------------------------
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

image_datasets = {
    'train': DatasetWrapper(train_subset, data_transforms['train']),
    'val': DatasetWrapper(val_subset, data_transforms['val'])
}

# ---------------------------------------------------------
# Dynamic Batch Sizing (Test 64, then 32, then 16)
# ---------------------------------------------------------
BATCH_SIZE = 64

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0, pin_memory=False)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(f"Classes: {class_names}")
print(f"Training images: {dataset_sizes['train']}")

# ---------------------------------------------------------
# 4. MODEL SETUP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = ConvNeXt_Base_Weights.DEFAULT
model = models.convnext_base(weights=weights)

num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, len(class_names))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ---------------------------------------------------------
# 5. MIXED PRECISION & GRADIENT ACCUMULATION LOOP
# ---------------------------------------------------------
# Fix the FutureWarning by using the updated PyTorch 2.0 syntax
scaler = torch.amp.GradScaler('cuda')

# Automatically calculates how many steps to accumulate to maintain a mathematical global batch of 64
accumulation_steps = max(1, 64 // BATCH_SIZE) 

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad() # Zero gradients at start of epoch

            # Iterate over data
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Fix the FutureWarning for autocast
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Normalize loss for gradient accumulation
                        if phase == 'train':
                            loss = loss / accumulation_steps

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        
                        # Step optimizer only after accumulating N batches
                        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloaders[phase]):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                # Multiply loss back to get the true running loss
                current_loss = (loss.item() * accumulation_steps) if phase == 'train' else loss.item()
                running_loss += current_loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Force garbage collection of heavy tensors to prevent any creeping leaks
                del inputs, labels, outputs, preds, loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Safely clone weights to CPU to prevent GPU/CPU memory entanglement leaks
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                best_model_wts = {k: v.cpu().clone() for k, v in state.items()}

            # CRITICAL: Force python garbage collector to clear the main thread memory
            gc.collect()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    save_path = '/kaggle/working/convnext_base_skin_v2.pth'
    torch.save({
        'model_state_dict': best_model_wts,
        'class_names': class_names
    }, save_path)
    print(f"Model saved to {save_path}")

    return model

if __name__ == '__main__':
    train_model(model, criterion, optimizer, scheduler, num_epochs=10)
