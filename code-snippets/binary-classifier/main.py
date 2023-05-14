import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class VectorDataset(Dataset):
    def __init__(self, length=1000, dim=10, transform=None):
        self.length = length
        self.dim = dim
        self.transform = transform

        # Generate random vectors
        self.data = torch.rand((length, dim))

        # Generate random binary labels
        self.labels = torch.randint(0, 2, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.data[index], self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample


# Create a dataset instance
dataset = VectorDataset()


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),  # Changed input dimension to 10
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Create a model instance
model = BinaryClassifier()
# Training parameters
learning_rate = 0.01
num_epochs = 50

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.float(), labels.float().view(-1, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Assume you have some new data
new_data = torch.rand((5, 10))  # 5 new samples, each with 10 features

# Ensure the model is in evaluation mode
model.eval()

# No need to track gradients for inference
with torch.no_grad():
    # Forward pass
    outputs = model(new_data)

# Apply threshold to get binary outputs
predicted_labels = (outputs > 0.5).int()

print(predicted_labels)
