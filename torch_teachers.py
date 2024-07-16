import torch
from torch import nn, optim
import time
from models import CNN
import globals


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
    :param model: string representing whether the model is meant to be a teacher or a student, which changes what is saved
    :return: Tuple containing the model being trained and the accuracy of the model on the validation set at the end of training
    """
    print("training...")
    #print("training data size:",np.shape(training_data))
    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    

    network = CNN(globals.dataset).to(device)
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
            opt.zero_grad()
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)

            preds = network(batch_xs)
            acc = (preds.argmax(dim=1) == batch_ys).float().mean()
            train_acc.append(acc)

            loss_val = loss(preds, batch_ys)

            loss_val.backward()
            opt.step()

        acc = torch.tensor(train_acc).mean()
        print(acc)  # see trianing accuracy
        train_accs.append(acc)
    
    if model == "student":
        # NOTE this does not work with multiple students in the same folder at the same time
        best_num_epochs = torch.argmax(torch.tensor(valid_accs)) * 5
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
    ds = globals.dataset
    train_sets = ds.teach_train
    valid_sets = ds.teach_valid
    for i in range(num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        n, acc = train(train_sets[i], valid_sets[i], dataset, globals.device)
        print("TEACHER",i,"ACC",acc)
        torch.save(n.state_dict(),f"./saved/{dataset}_teacher_{i}_of_{num_teachers-1}.tch")
        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")

def main():
    dataset = globals.dataset.name
    num_teachers = globals.dataset.num_teachers
    train_all(dataset, num_teachers)

if __name__ == '__main__':
    main()
