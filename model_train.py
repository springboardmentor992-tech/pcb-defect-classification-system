"""
PCB Defect Classification Training Pipeline (Module 3)
Requirements: PyTorch, torchvision, scikit-learn, matplotlib, seaborn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import copy
import time
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "data_dir": "/Users/arya/Documents/Infosys/pcbS/processed_rois/rois", # Path to your extracted ROIs
    "output_dir": "/Users/arya/Documents/Infosys/pcbS/models",
    "img_size": 128,
    "batch_size": 32,
    "epochs": 25,
    "learning_rate": 0.001,
    "seed": 42
}

# Set device with Apple Silicon (MPS) support
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # <--- This unlocks your M1 GPU
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def get_data_transforms():
    """
    Define preprocessing and augmentation pipelines.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def prepare_datasets(data_dir):
    """
    Load data, split into Train/Val/Test, and handle class imbalance.
    """
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Split: 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms
    transforms_dict = get_data_transforms()
    train_dataset.dataset.transform = transforms_dict['train']
    val_dataset.dataset.transform = transforms_dict['val']
    test_dataset.dataset.transform = transforms_dict['test']
    
    print(f"Data Split: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")
    
    # --- HANDLE CLASS IMBALANCE (WeightedRandomSampler) ---
    # We calculate weights for the training set only
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                          sampler=sampler, num_workers=2), # Use sampler here
        'val': DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                        shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=False, num_workers=2)
    }
    
    return dataloaders, class_names

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Training loop with validation tracking and best model saving.
    """
    since = time.time()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, class_names, output_dir):
    """
    Evaluate on Test set: Confusion Matrix & Classification Report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 1. Classification Report
    print("\nTest Set Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save report
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix 
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.show()

def plot_history(history, output_dir):
    """
    Plot training/validation accuracy and loss curves.
    """
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot 
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.savefig(f"{output_dir}/training_plots.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Create output dir
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # 1. Data Preparation
    print("Preparing Data...")
    try:
        dataloaders, class_names = prepare_datasets(CONFIG['data_dir'])
    except Exception as e:
        print(f"Error loading data. Did you run the ROI extraction script first? \n{e}")
        exit()
        
    # 2. Model Setup (Transfer Learning)
    print("Initializing ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze initial layers (optional, but good for small datasets)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Replace final layer for our specific number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 4. Training
    print("Starting Training...")
    model, history = train_model(model, dataloaders, criterion, optimizer, CONFIG['epochs'])
    
    # 5. Save Model
    save_path = f"{CONFIG['output_dir']}/pcb_defect_resnet18.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # 6. Evaluation
    print("Evaluating on Test Set...")
    plot_history(history, CONFIG['output_dir'])
    evaluate_model(model, dataloaders['test'], class_names, CONFIG['output_dir'])