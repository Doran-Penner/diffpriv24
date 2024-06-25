import torch
import math
from torch import nn, optim
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
import numpy as np
import torch.utils.tensorboard as tb
import datetime
import time
import os

#setup device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = 'cpu'
print(device)

#setup transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)

num_teachers = 250

train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, len(dataset) - train_size - valid_size])

train_partition = [math.floor(train_size / num_teachers) + 1 for i in range(train_size % num_teachers)] + [math.floor(train_size / num_teachers) for i in range(num_teachers - (train_size % num_teachers))]
valid_partition = [math.floor(valid_size / num_teachers) + 1 for i in range(valid_size % num_teachers)] + [math.floor(valid_size / num_teachers) for i in range(num_teachers - (valid_size % num_teachers))]
train_sets = torch.utils.data.random_split(train_set, train_partition)
valid_sets = torch.utils.data.random_split(valid_set, valid_partition)

# Normalize 
normalize = transforms.Normalize((0.1307,), (0.3081,))

# Default architecture
arch = [(5, 64), (5, 128)]

class CNN(nn.Module):

    def __init__(self, arch=[],padding=True):
        super().__init__()
        pad = 'same' if padding else 0
        size = 32
        layers = [nn.Conv2d(3,64,5, padding=pad)]
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        
        layers.append(nn.Conv2d(64, 128, 5, padding=pad))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(8192,384))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(384,192))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(192,10))       

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def train(training_data, valid_data, arch=[], lr=1e-3, epochs=159, batch_size=16, momentum=0.9,padding=True):
    print("training...")
    #print("training data size:",np.shape(training_data))
    #print("valid data size:",np.shape(valid_data))
    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    network = CNN(arch=arch,padding=padding).to(device)
    opt = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    loss = nn.CrossEntropyLoss()

    train_accs = []
    valid_accs = []

    for i in range(epochs):
        # print("Epoch",i)
        network.train()
        train_acc = []
        for batch_xs, batch_ys in train_loader:
            batch_xs = normalize(batch_xs).to(device)
            batch_ys = batch_ys.to(device)

            preds = network(batch_xs)
            acc = (preds.argmax(dim=1) == batch_ys).float().mean()
            train_acc.append(acc)

            loss_val = loss(preds, batch_ys)

            opt.zero_grad()
            loss_val.backward()
            opt.step()

        train_accs.append(torch.tensor(train_acc).mean())
        # print("Training accuracy:",train_accs[-1])

        network.eval()
        accs = []
        losses = []
        for batch_xs, batch_ys in valid_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            preds = network(normalize(batch_xs))
            accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
        acc = torch.tensor(accs).mean()
        # print("Valid accuracy:",acc)
        valid_accs.append(acc)
    return (network, valid_accs)
teachers = []
for i in range(num_teachers):
    print(f"Training teacher {i} now!")
    start_time = time.time()
    n, accs = train(train_sets[i],valid_sets[i],arch=arch)
    print("TEACHER",i,"ACC",accs[-1])
    teachers.append(n)
    torch.save(n.state_dict(),"./saved/teacher_" + str(i) + ".txt")
    duration = time.time()- start_time
    print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")
