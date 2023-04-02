"""
Created by cyborgoat at 2022/8/28
Email: cyborgoat@outlook.com

Description
-----------

"""
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dtsets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using{device} as device')

# Declaring Hyper-parameters
input_size = 28
seqlen = 28
numlayrs = 2
hidden_size = 254
num_classes = 10
lr = 0.001
batch_size = 256
num_epochs = 2


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_class, seqlen):
        super(RNNModel, self).__init__()
        self.hidensize = hidden_size
        self.numlayrs = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * seqlen, num_class)

    def forward(self, data):
        h = torch.zeros(self.numlayrs, data.size(0), self.hidensize).to(device)
        c = torch.zeros(self.numlayrs, data.size(0), self.hidensize).to(device)

        outp, _ = self.rnn(data, h)
        outp = outp.reshape(outp.shape[0], -1)
        outp = self.fc(outp)
        return outp


traindt = dtsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
testdt = dtsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=traindt, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testdt, batch_size=batch_size, shuffle=True)

model = RNNModel(input_size, hidden_size, numlayrs, num_classes, seqlen).to(device)

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=lr)

# Training Loop
ep = 1
pbar = tqdm(range(num_epochs), desc=f'Training model for epoch {ep}/{num_epochs}', total=num_epochs)
for epoch in pbar:
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        trgts = trgt.to(device)
        scores = model(data)
        loss = criterion(scores, trgts)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"Epoch:{epoch}/{num_epochs}  Batch:{batch_idx}/{len(train_loader)} Loss:{loss}")

    ep += 1
    # Evaluating our RNN model


def check_accuracy(ldr, modlrnnlm):
    if ldr.dataset.train:
        print('Check accuracy on training data')
    else:
        print('Check accuracy on test data')

    num_correct = 0
    num_samples = 0
    modlrnnlm.eval()
    with torch.no_grad():
        for i, j in ldr:
            i = i.to(device).squeeze(1)
            j = j.to(device)
            score = modlrnnlm(i)
            _, predictions = score.max(1)
            num_correct += (predictions == j).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)