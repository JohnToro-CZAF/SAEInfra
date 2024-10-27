import os
import time
import json
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pretrained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Modify the final layer for CIFAR-10 (10 classes)
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='data', train=True,
                                    download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024,
                                            shuffle=True, num_workers=16)

# Simulate loading a checkpoint if exists
total_epochs = 100  # Simulate 10 steps
model.train()
running_loss = 0.0
for inputs, labels in tqdm.tqdm(train_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)

    # Simulate a short sleep to mimic training time

epoch_loss = running_loss / len(train_loader.dataset)