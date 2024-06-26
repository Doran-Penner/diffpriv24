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
import random
from models import CNN

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

#Verify random seed
#print([i for i in torch.utils.data.random_split([0,1,2,3,4,5,6,7,8,9],[5,5])[0]])


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

train_dataset = torchvision.datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
extra_dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)
dataset = torch.utils.data.ConcatDataset([train_dataset,extra_dataset])
valid_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
train_size = len(dataset)

num_teachers = 250

train_partition = [math.floor(train_size / num_teachers) + 1 for i in range(train_size % num_teachers)] + [math.floor(train_size / num_teachers) for i in range(num_teachers - (train_size % num_teachers))]
train_sets = torch.utils.data.random_split(dataset, train_partition)

# Normalize 
normalize = transforms.Normalize((0.1307,), (0.3081,))

# Default architecture
arch = [(5, 64), (5, 128)]

def train(training_data, valid_data, arch=[], lr=1e-3, epochs=70, batch_size=16, momentum=0.9,padding=True):
    print("training...")
    #print("training data size:",np.shape(training_data))
    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    

    network = CNN(arch=arch,padding=padding).to(device)
    opt = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
    loss = nn.CrossEntropyLoss()

    train_accs = []

    for i in range(epochs):
        if i % 10 == 0:
            print("Epoch",i)
            '''network.eval()
            accs = []
            losses = []
            for batch_xs, batch_ys in valid_loader:
                batch_xs = batch_xs.to(device)
                batch_ys = batch_ys.to(device)
                preds = network(normalize(batch_xs))
                accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
            acc = torch.tensor(accs).mean()
            print("Acc:",acc)'''
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

        acc = torch.tensor(train_acc).mean()
        train_accs.append(acc)
 
    network.eval()
    accs = []
    losses = []
    for batch_xs, batch_ys in valid_loader:
        batch_xs = batch_xs.to(device)
        batch_ys = batch_ys.to(device)
        preds = network(normalize(batch_xs))
        accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
    acc = torch.tensor(accs).mean()
    return (network, acc)


teachers = []
for i in range(num_teachers):
    print(f"Training teacher {i} now!")
    start_time = time.time()
    n, acc = train(train_sets[i],valid_dataset,arch=arch)
    print("TEACHER",i,"ACC",acc)
    teachers.append(n)
    torch.save(n.state_dict(),"./saved/teacher_" + str(i) + ".txt")
    duration = time.time()- start_time
    print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")
