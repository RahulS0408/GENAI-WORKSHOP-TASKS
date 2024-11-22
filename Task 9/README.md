# Few-Shot Learning for Image Classification

## Description
This project demonstrates how to implement few-shot learning for image classification using a pre-trained VGG16 model. The model is fine-tuned on the CIFAR-10 dataset. Few-shot learning leverages prior knowledge from a large dataset to learn new tasks with only a few training examples.
## Setup

### Step 1: Install Necessary Libraries
First, install the required libraries including `transformers`, `datasets`, `torch`, `torchvision`, and `tqdm`.

```python
!pip install transformers datasets torch torchvision tqdm

---

```
### Step 2: Import Libraries and Load Data
Import the necessary libraries and load a sample dataset. For this project, we use the CIFAR-10 dataset, which is commonly used for image classification tasks.

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

---

```
### Step 3: Define the Model
Load a pre-trained VGG16 model and modify its classifier to fit the CIFAR-10 dataset.

```python
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

# Load the pre-trained VGG16 model
model = vgg16(pretrained=True)

# Modify the classifier to fit the CIFAR-10 dataset (10 classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)

---

```
### Step 4: Define the Training and Testing Loops with Progress Bar
Define functions to train and test the model, incorporating a progress bar to track training progress.

```python
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(trainloader))

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%)

---

```
### Step 5: Train and Test the Model
Train the model and evaluate its performance.

```python
train_model(model, trainloader, criterion, optimizer, epochs=2)
test_model(model, testloader)
