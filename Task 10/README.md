# Neural Network Quantization for Image Classification

## Description
This project demonstrates how to implement neural network quantization for a simple CNN model trained on the MNIST dataset. Quantization reduces the precision of the numbers used to represent a model's parameters, leading to faster inference and reduced model size.

## Setup

### Step 1: Install Necessary Libraries
First, install the required libraries including `torch` and `torchvision`.

```python
!pip install torch torchvision

---

```
### Step 2: Import Libraries and Define Model
Import the necessary libraries and define a sample neural network model. For this project, we use a simple CNN model trained on the MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = SimpleCNN()

---

```
### Step 3: Prepare Data
Load and preprocess the MNIST dataset.

```python
# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training and test sets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

---

```
### Step 4: Train the Model
Define the training loop and train the model on the MNIST dataset.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, trainloader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

train_model(model, trainloader, criterion, optimizer, epochs=1)

---

```
### Step 5: Evaluate the Model
Define a function to evaluate the model and check its accuracy before quantization.

```python
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
    accuracy = 100 * correct / total
    return accuracy

# Evaluate the model before quantization
initial_accuracy = test_model(model, testloader)
print(f"Initial accuracy: {initial_accuracy}%")

---

```
### Step 6: Quantize the Model
Apply dynamic quantization to the trained model.

```python
# Apply dynamic quantization
model.to('cpu')
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
print("Quantized model saved as quantized_model.pth")

---

```
### Step 7: Evaluate the Quantized Model
Evaluate the accuracy of the quantized model.

```python
# Load the quantized model for evaluation
quantized_model.load_state_dict(torch.load("quantized_model.pth"))
quantized_model.to(device)

# Evaluate the quantized model
quantized_accuracy = test_model(quantized_model, testloader)
print(f"Quantized accuracy: {quantized_accuracy}%")
