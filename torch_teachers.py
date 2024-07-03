import torch
from torch import nn, optim
import time
import math
import helper
from models import CNN

def load_partitioned_dataset(dataset, num_teachers):
    """
    This function loads a specified training dataset, partitioned according to the number of teachers, as well as a validation dataset.
    :param dataset: string specifying which dataset to load (one of svhn, mnist, and cifar-10)
    :param num_teachers: integer specifying the number of teachers, for partitioning the dataset
    :return: Tuple containing an array containing the partitioned datasets for training and the dataset for validation.
    """
    train_data, valid_data, _test_data = helper.load_dataset(dataset_name=dataset, split="teach", make_normal=True)
    generator = torch.Generator().manual_seed(0)
    train_size = len(train_data)
    valid_size = len(valid_data)
    train_partition = ([math.floor(train_size / num_teachers) + 1 for i in range(train_size % num_teachers)]
                    + [math.floor(train_size / num_teachers) for i in range(num_teachers - (train_size % num_teachers))])
    valid_partition = ([math.floor(valid_size / num_teachers) + 1 for i in range(valid_size % num_teachers)]
                    + [math.floor(valid_size / num_teachers) for i in range(num_teachers - (valid_size % num_teachers))])
    train_sets = torch.utils.data.random_split(train_data, train_partition, generator=generator)
    valid_sets = torch.utils.data.random_split(valid_data, valid_partition, generator=generator)
    return train_sets, valid_sets

def train(training_data, valid_data, dataset, device='cpu', lr=1e-3, epochs=70, batch_size=16, momentum=0.9, padding=True, model="teacher"):
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
    valid_accs = []  # only used for student

    for i in range(epochs):
        if i % 5 == 0:
            print("Epoch",i)
            ### check valid accuracy
            network.eval()
            accs = []
            for batch_xs, batch_ys in valid_loader:
                batch_xs = batch_xs.to(device)
                batch_ys = batch_ys.to(device)
                preds = network(batch_xs)
                accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
            acc = torch.tensor(accs).mean()
            print("Valid acc:",acc)
            valid_accs.append(acc)
            if model == "student":
                torch.save(network.state_dict(),f"./saved/{dataset}_student_{i}.ckp")
            ### end check
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
        print(acc)  # see trianing accuracy
        train_accs.append(acc)
    
    if model == "student":
        best_num_epochs = torch.argmax(valid_accs) * 5
        print("Final num epochs:", best_num_epochs)
        st_dict = torch.load(f"./saved/{dataset}_student_{best_num_epochs}.ckp",map_location=device)
        network.load_state_dict(st_dict)
    
    network.eval()
    accs = []
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
    train_sets, valid_sets = load_partitioned_dataset(dataset, num_teachers)
    for i in range(num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        n, acc = train(train_sets[i], valid_sets[i], dataset, helper.device)
        print("TEACHER",i,"ACC",acc)
        torch.save(n.state_dict(),f"./saved/{dataset}_teacher_{i}_of_{num_teachers-1}.tch")
        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")

def main():
    dataset = "svhn"
    num_teachers = 250
    train_all(dataset, num_teachers)

if __name__ == '__main__':
    main()
