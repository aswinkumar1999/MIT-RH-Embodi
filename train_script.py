import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
data_dir = "./images"

# Hyperparameters
batch_size = 64  # Increased for faster training
num_epochs = 25
num_classes = 4
train_ratio = 0.8  # Ratio of training data

# Data transformations with augmentation
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

# Load full dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform['train'])

# Split dataset into train and val
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Update transforms for validation dataset
val_dataset.dataset.transform = transform['val']

# # Compute class weights for imbalance
# class_counts = np.zeros(num_classes)
# for _, label in dataset:
#     class_counts[label] += 1
# class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
# sample_weights = [class_weights[label] for _, label in train_dataset]
# sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Data loaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

data_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Modify the final layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024,512),
    nn.Dropout(0.5),
    nn.Linear(512,256),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

# Scheduler for learning rate decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

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

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Evaluation and visualization
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# Train and save the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
torch.save(model.state_dict(), 'fine_tuned_resnet50.pth')

# Evaluate and visualize
class_names = dataset.classes
labels, preds = evaluate_model(model, dataloaders['val'])

