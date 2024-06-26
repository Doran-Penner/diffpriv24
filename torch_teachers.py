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
import random
from models import CNN

def load_partitioned_dataset(dataset, num_teachers):
    """
    This function loads a specified training dataset, partitioned according to the number of teachers, as well as a validation dataset.
    :param dataset: string specifying which dataset to load (one of svhn, mnist, and cifar-10)
    :param num_teachers: integer specifying the number of teachers, for partitioning the dataset
    :return: Tuple containing an array containing the partitioned datasets for training and the dataset for validation, or False if anything breaks.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
        extra_dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)
        valid_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
        normalize = transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])
        train_dataset, extra_dataset, valid_dataset = normalize(train_dataset), normalize(extra_dataset), normalize(valid_dataset)
        dataset = torch.utils.data.ConcatDataset([train_dataset,extra_dataset])
    elif dataset == 'mnist':
        dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        dataset, valid_dataset = normalize(dataset), normalize(valid_dataset)
    else:
        print("Check value of dataset flag.")
        return False
    train_size = len(dataset)
    train_partition = [math.floor(train_size / num_teachers) + 1 for i in range(train_size % num_teachers)] + [math.floor(train_size / num_teachers) for i in range(num_teachers - (train_size % num_teachers))]
    train_sets = torch.utils.data.random_split(dataset, train_partition, generator = torch.Generator().manual_seed(0))
    return train_sets, valid_dataset

def train(training_data, valid_data, dataset, device='cpu', lr=1e-3, epochs=70, batch_size=16, momentum=0.9,padding=True):
    """
    This is a function that trains the model on a specified dataset.
    :param training_data: dataset containing the training data for the model
    :param valid_data: dataset containing the validation data for the model
    :param dataset: string containing the name of the dataset that the model is being trained on. This is to tell model.py which model to give us
    :param device: string specifying which device the code is running on, so that the code can be appropriately optimized
    :param lr: float specifying the learning rate for the model
    :param epochs: int specifying the length of training
    :param batch_size: int specifying the amount of data being trained on per batch
    :param momentum: float specifying the momentum of the learning process. Carter says to set it at 0.9 and not worry about it
    :param padding: boolean specifying whether we want to do padding
    :return: Tuple containing the model being trained and the accuracy of the model on the validation set at the end of training
    """
    print("training...")
    #print("training data size:",np.shape(training_data))
    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    

    network = CNN(padding=padding,dataset=dataset).to(device)
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

def train_all(dataset='svhn', num_teachers=250):
    """
    This function trains all of the teacher models on the specified dataset
    :param dataset: string specifying which dataset to train the teachers on
    :param num_teachers: integer specifying the number of teachers to train
    :return: Does not return anything, but saves the models instead
    """
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
        n, acc = train(train_sets[i],valid_dataset,dataset,device)
        print("TEACHER",i,"ACC",acc)
        torch.save(n.state_dict(),f"./saved/{dataset}_teacher_{i}_of_{num_teachers}.txt")
        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")

def main():
    train_all('mnist', 250)

if __name__ == '__main__':
    main()
