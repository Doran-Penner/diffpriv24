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

def load_partitioned_dataset(dataset, num_teachers):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
        extra_dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)
        valid_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
        normalize = transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])
        train_dataset, extra_dataset, valid_dataset = normalize(train_dataset), normalize(extra_dataset), normalize(valid_dataset)
    else:
        print("Check value of dataset flag.")
        return False
    dataset = torch.utils.data.ConcatDataset([train_dataset,extra_dataset])
    train_size = len(dataset)
    train_partition = [math.floor(train_size / num_teachers) + 1 for i in range(train_size % num_teachers)] + [math.floor(train_size / num_teachers) for i in range(num_teachers - (train_size % num_teachers))]
    train_sets = torch.utils.data.random_split(dataset, train_partition, generator = torch.Generator().manual_seed(0))
    return train_sets, valid_dataset

def train(training_data, valid_data, device='cpu', arch=[(5, 64), (5, 128)], lr=1e-3, epochs=70, batch_size=16, momentum=0.9,padding=True):
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
                preds = network(batch_xs)
                accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
            acc = torch.tensor(accs).mean()
            print("Acc:",acc)'''
        network.train()
        train_acc = []
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)
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
        preds = network(batch_xs)
        accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
    acc = torch.tensor(accs).mean()
    return (network, acc)

def train_all(dataset='svhn', num_teachers=250,):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = 'cpu'
    print(device)
    train_sets, valid_dataset = load_partitioned_dataset(dataset, num_teachers)
    for i in range(num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        n, acc = train(train_sets[i],valid_dataset, device)
        print("TEACHER",i,"ACC",acc)
        torch.save(n.state_dict(),"./saved/teacher_" + str(i) + ".txt")
        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")

def main():
    train_all('svhn', 250)

if __name__ == '__main__':
    main()
