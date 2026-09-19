"""
File: src/dip/train_cnn.py
Purpose: Trains a PyTorch CNN (EfficientNet-B0) on the 10-class skin disease dataset.
         Designed to be run in a Kaggle Notebook environment with a GPU.
Why we need it: This script leverages transfer learning to build a highly accurate 
                image classifier for the DIP branch of the project.
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ==============================================================================
# CONFIGURATION
# ==============================================================================
base_input = '/kaggle/input'
DATA_DIR = 'data/processed' # default local fallback

if os.path.exists(base_input):
    # Search for the first image file to deduce the dataset structure
    for root, dirs, files in os.walk(base_input):
        found = False
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # root = class folder (e.g., ".../1. Eczema")
                # parent of root = DATA_DIR we need for ImageFolder!
                DATA_DIR = os.path.dirname(root)
                found = True
                break
        if found:
            break

print(f"Auto-detected DATA_DIR: {DATA_DIR}")
OUTPUT_MODEL_PATH = '/kaggle/working/efficientnet_b0_skin.pth'

BATCH_SIZE = 64 # T4x2 GPUs have plenty of VRAM, 64 is safe
NUM_EPOCHS = 10
NUM_CLASSES = 10
LEARNING_RATE = 0.001

# Automatically use the GPU if available (which it is on Kaggle)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. DATA TRANSFORMS & LOADERS
# ==============================================================================
# We use data augmentation on the training set to prevent overfitting
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # Standard ImageNet normalization values
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_data():
    print(f"Loading data from {DATA_DIR}...")
    
    # We use ImageFolder, which automatically infers classes from folder names
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply the respective transforms (PyTorch doesn't natively support different transforms 
    # for splits out-of-the-box easily, so we wrap them to apply the correct transform)
    train_dataset.dataset.transform = data_transforms['train']
    # A small hack for validation transforms:
    val_dataset = copy.deepcopy(val_dataset)
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = full_dataset.classes
    
    print(f"Classes found: {class_names}")
    print(f"Training images: {train_size} | Validation images: {val_size}")
    
    return dataloaders, dataset_sizes, class_names

# ==============================================================================
# 2. MODEL BUILDING (Transfer Learning)
# ==============================================================================
def build_model(num_classes):
    print("\nInitializing EfficientNet-B0...")
    # Load the pre-trained weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Replace the final classification head to match our 10 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass & optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save the model and the class names together
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names
                }, OUTPUT_MODEL_PATH)
                print(f"*** New best model saved to {OUTPUT_MODEL_PATH} ***")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print(f"Using compute device: {device}")
    
    dataloaders, dataset_sizes, class_names = load_data()
    
    model = build_model(NUM_CLASSES)
    
    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    print("\nAll done! You can now download the .pth file from Kaggle's /kaggle/working/ directory.")
