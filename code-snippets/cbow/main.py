import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class CBOWDataset(Dataset):
    def __init__(self, training_data, word_to_ix):
        self.training_data = training_data
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        context, target = self.training_data[index]
        context_ids = [self.word_to_ix[word] for word in context.split()]
        target_id = self.word_to_ix[target]
        return context_ids, target_id


def collate_fn(batch):
    batch_context, batch_target = zip(*batch)
    batch_context_padded = pad_sequence([torch.tensor(context) for context in batch_context],
                                        batch_first=True,
                                        padding_value=0)
    batch_target_padded = torch.tensor(batch_target)
    return batch_context_padded, batch_target_padded


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, vocab_size + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        summed = torch.sum(embedded, dim=1)
        output = self.linear(summed)
        output = self.softmax(output)
        return output


# Example training data
training_data = [
    ("I love deep learning", "studying"),
    ("I enjoy NLP lecture", "those"),
    ("I like PyTorch library", "using"),
    ("I prefer Python courses", "taking")
]

# Prepare vocabulary
vocab = set()
for context, target in training_data:
    vocab.update(context.split())
    vocab.update(target.split())
vocab_size = len(vocab)
word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}

# Hyperparameters
embedding_dim = 16
context_size = 2
learning_rate = 0.01
epochs = 300
batch_size = 1

# Create CBOW model
model = CBOW(vocab_size, embedding_dim)

# Define loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Create CBOWDataset and DataLoader
dataset = CBOWDataset(training_data, word_to_ix)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch_context, batch_target in dataloader:
        inputs = batch_context
        labels = batch_target

        # Forward pass
        model.zero_grad()
        output = model(inputs)

        # Compute loss
        loss = loss_function(output, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

# Get word embeddings
embedding_weights = model.embedding.weight.data

# Print word embeddings
for i, word in enumerate(vocab):
    print(f"Word: {word}, Embedding: {embedding_weights[i]}")
