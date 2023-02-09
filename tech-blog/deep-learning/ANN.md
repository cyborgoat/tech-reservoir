---
title: MNIST Handwritten Digit Recognition
summary: Hands on small project to build a neural net to recognize hand written digits from MNIST dataset with PyTorch. 
author: Junxiao Guo
date: 2021-05-05
tags:
  - deep-learning
  - computer-vision
---

## Introduction on MNIST

The MNIST dataset (Mixed National Institute of Standards and Technology database) is a large handwritten digital database collected by the National Institute of Standards and Technology, including a training set of 60,000 examples and a test set of 10,000 examples.
![](https://bbs-img.huaweicloud.com/blogs/img/MNIST.png)

This tutorial will use PyTorch to implement a simple neural network model to recognize handwritten digits

## Initialize

```python
import torch
import torchvision
```

## Setting model parameters

```python
n_epochs = 3 
batch_size_train = 64    
batch_size_test = 1000    
learning_rate = 0.01
momentum = 0.5
log_interval = 100 # Frequency for log printing
```

## Loading dataset

```python
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
```

```python
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
```

```python
example_data.shape
```

    torch.Size([1000, 1, 28, 28])

```python
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig
```

![](https://bbs-img.huaweicloud.com/blogs/img/6f1ff120ae0cac7ecce64bd72348e18b_405x267.png@900-0-90-f.png)

## Building Neural Net

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

### Defining network structure

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=-1)
```

### CNN(Convolution Neural Network) Version

```python
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=-1)
```

### Defining the output path

```python
import os
cur_dir = os.getcwd()
output_path = os.path.join(cur_dir,"results")
from pathlib import Path
Path(output_path).mkdir(parents=True, exist_ok=True)
```

### Instantiate the Net object and choose the optimizer

```python
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
```

```python
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

```

### Training & Testing

```python
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/results/model.pth')
      torch.save(optimizer.state_dict(), '/results/optimizer.pth')
```

```python
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
```

```python
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
```

    C:\ProgramData\Anaconda3\lib\site-packages\torch\nn\_reduction.py:44: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
      warnings.warn(warning.format(ret))

​
    Test set: Avg. loss: 2.3236, Accuracy: 913/10000 (9%)

    Train Epoch: 1 [0/60000 (0%)] Loss: 2.359348
    Train Epoch: 1 [6400/60000 (11%)] Loss: 0.912735
    Train Epoch: 1 [12800/60000 (21%)] Loss: 0.791347
    Train Epoch: 1 [19200/60000 (32%)] Loss: 0.675237
    Train Epoch: 1 [25600/60000 (43%)] Loss: 0.598521
    Train Epoch: 1 [32000/60000 (53%)] Loss: 0.471895
    Train Epoch: 1 [38400/60000 (64%)] Loss: 0.458211
    Train Epoch: 1 [44800/60000 (75%)] Loss: 0.328482
    Train Epoch: 1 [51200/60000 (85%)] Loss: 0.386726
    Train Epoch: 1 [57600/60000 (96%)] Loss: 0.434701
    
    Test set: Avg. loss: 0.2936, Accuracy: 9181/10000 (92%)
    
    Train Epoch: 2 [0/60000 (0%)] Loss: 0.758641
    Train Epoch: 2 [6400/60000 (11%)] Loss: 0.531642
    Train Epoch: 2 [12800/60000 (21%)] Loss: 0.474639
    Train Epoch: 2 [19200/60000 (32%)] Loss: 0.587206
    Train Epoch: 2 [25600/60000 (43%)] Loss: 0.557518
    Train Epoch: 2 [32000/60000 (53%)] Loss: 0.563830
    Train Epoch: 2 [38400/60000 (64%)] Loss: 0.340438
    Train Epoch: 2 [44800/60000 (75%)] Loss: 0.486597
    Train Epoch: 2 [51200/60000 (85%)] Loss: 0.517953
    Train Epoch: 2 [57600/60000 (96%)] Loss: 0.418494
    
    Test set: Avg. loss: 0.2365, Accuracy: 9314/10000 (93%)
    
    Train Epoch: 3 [0/60000 (0%)] Loss: 0.397563
    Train Epoch: 3 [6400/60000 (11%)] Loss: 0.190083
    Train Epoch: 3 [12800/60000 (21%)] Loss: 0.423875
    Train Epoch: 3 [19200/60000 (32%)] Loss: 0.422053
    Train Epoch: 3 [25600/60000 (43%)] Loss: 0.380608
    Train Epoch: 3 [32000/60000 (53%)] Loss: 0.266605
    Train Epoch: 3 [38400/60000 (64%)] Loss: 0.343277
    Train Epoch: 3 [44800/60000 (75%)] Loss: 0.305574
    Train Epoch: 3 [51200/60000 (85%)] Loss: 0.679861
    Train Epoch: 3 [57600/60000 (96%)] Loss: 0.372059
    
    Test set: Avg. loss: 0.2112, Accuracy: 9380/10000 (94%)

​

## Validate the performance

```python
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig
```

![](https://bbs-img.huaweicloud.com/blogs/img/result.png)

## Sampling & Testing

```python
with torch.no_grad():
  output = network(example_data)
```

```python
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig
```

![](https://bbs-img.huaweicloud.com/blogs/img/digit.png)
