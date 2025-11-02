import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn import datasets

class SKLearnMNISTDataset(Dataset):
    """MNIST dataset implementation using scikit-learn"""

    def __init__(self, train=True):
        # Load MNIST from scikit-learn
        mnist = datasets.fetch_openml('mnist_784', version=1)

        # Convert from pandas DataFrame to numpy array if needed
        if hasattr(mnist['data'], 'values'):
            data = mnist['data'].values.astype('float32')
        else:
            data = mnist['data'].astype('float32')

        if hasattr(mnist['target'], 'values'):
            targets = mnist['target'].values.astype('int64')
        else:
            targets = mnist['target'].astype('int64')

        # Normalize the data
        data = data / 255.0
        data = (data - 0.1307) / 0.3081  # Apply MNIST normalization

        # Split into train (first 60000) and test (last 10000)
        if train:
            self.data = data[:60000]
            self.targets = targets[:60000]
        else:
            self.data = data[60000:]
            self.targets = targets[60000:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        idx = int(idx)

        # Reshape to flattened vector (784 features) instead of 1x28x28
        image = torch.tensor(self.data[idx], dtype=torch.float)
        label = torch.tensor(int(self.targets[idx]), dtype=torch.long)

        return image, label


# Load train and test datasets
print("Loading MNIST dataset from scikit-learn...")
train_dataset = SKLearnMNISTDataset(train=True)
test_dataset = SKLearnMNISTDataset(train=False)

# Create train and validation splits
generator = torch.Generator().manual_seed(42)  # For reproducibility
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=generator
)

# Define batch size
batch_size = 64  # You can adjust this value based on your needs

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"Dataset sizes: Training={len(train_subset)}, Validation={len(val_subset)}, Test={len(test_dataset)}")

class SimpleModel(nn.Module):
    # More simple network architecture
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleModel, self).__init__()

        # Update this line - flatten images before the linear layer
        # 1x28x28 = 784 input features
        self.layer1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape the input to flatten it
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten from [batch_size, 1, 28, 28] to [batch_size, 784]

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class EnhancedModel(nn.Module):
    ## Deeper Network Architecture
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(EnhancedModel, self).__init__()

        # First hidden layer
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.3),

            # Second hidden layer
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(hidden_sizes[1], num_classes)
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop with accuracy evaluation
    """
    # Move model to device
    model.to(device)

    # History tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}')
        print(f'Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}')

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    """
    Plot training and validation metrics
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Model hyperparameters
input_size = 784  # Example for MNIST
# for the simple model use just 128
# hidden_size_simple = 128
hidden_size = [256, 128]
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Initialize model, loss, and optimizer
# model = SimpleModel(input_size, hidden_size_simple, num_classes)
model = EnhancedModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs
)

# Final evaluation
print("Final evaluation:")
test_loss, test_acc = evaluate_model(model, test_loader, criterion,
                                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
