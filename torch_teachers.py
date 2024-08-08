import torch
from torch import nn, optim
import time
from models import CNN
import globals
import numpy as np
from os.path import isfile


def train(training_data, valid_data, dat_obj, lr=1e-3, epochs=70, batch_size=16, momentum=0.9, model="teacher", net=CNN):
    """
    This is a function that trains the model on a specified dataset.
    :param training_data: dataset containing the training data for the model
    :param valid_data: dataset containing the validation data for the model
    :param dat_obj: datasets._Dataset object representing the dataset being trained on
    :param device: string specifying which device the code is running on, so that the code can be appropriately optimized
    :param lr: float specifying the learning rate for the model
    :param epochs: int specifying the length of training
    :param batch_size: int specifying the amount of data being trained on per batch
    :param momentum: float specifying the momentum of the learning process. Carter says to set it at 0.9 and not worry about it
    :param padding: boolean specifying whether we want to do padding
    :param model: string representing whether the model is meant to be a teacher or a student, which changes what is saved
    :return: Tuple containing the model being trained and the accuracy of the model on the validation set at the end of training
    """
    assert model in ["teacher", "student"], "misnamed model parameter!"
    print("training...")
    #print("training data size:",np.shape(training_data))
    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    

    network = net(dat_obj).to(globals.device)
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
                batch_xs = batch_xs.to(globals.device,dtype=torch.float32)
                batch_ys = batch_ys.to(globals.device,dtype=torch.float32) # added this purely to try to get rid of an error?
                preds = network(batch_xs)
                accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
            acc = torch.tensor(accs).mean()
            print("Valid acc:",acc)
            valid_accs.append(acc)
            torch.save(network.state_dict(),f"{globals.SAVE_DIR}/{dat_obj.name}_{model}_{i}.ckp")
            ### end check
        network.train()
        train_acc = []
        for batch_xs, batch_ys in train_loader:
            opt.zero_grad()
            batch_xs = batch_xs.to(globals.device,dtype=torch.float32)
            batch_ys = batch_ys.to(globals.device,dtype=torch.float32)

            preds = network(batch_xs)
            acc = (preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean()
            train_acc.append(acc)

            loss_val = loss(preds, batch_ys)

            loss_val.backward()
            opt.step()

        acc = torch.tensor(train_acc).mean()
        print(acc)  # see trianing accuracy
        train_accs.append(acc)
    
    # NOTE this does not work with multiple students in the same folder at the same time
    best_num_epochs = torch.argmax(torch.tensor(valid_accs)) * 5
    print("Final num epochs:", best_num_epochs)
    st_dict = torch.load(f"{globals.SAVE_DIR}/{dat_obj.name}_{model}_{best_num_epochs}.ckp",map_location=globals.device)
    network.load_state_dict(st_dict)
    
    network.eval()
    accs = []
    for batch_xs, batch_ys in valid_loader:
        batch_xs = batch_xs.to(globals.device,dtype=torch.float32)
        batch_ys = batch_ys.to(globals.device,dtype=torch.float32)
        preds = network(batch_xs)
        accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
    acc = torch.tensor(accs).mean()
    return (network, acc)

def train_all(dat_obj):
    """
    This function trains all of the teacher models on the specified dataset
    :param dat_obj: datasets._Dataset object representing the dataset being trained on
    :return: Does not return anything, but saves the models instead
    """
    train_sets = dat_obj.teach_train
    valid_sets = dat_obj.teach_valid
    for i in range(dat_obj.num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        # n, acc = train(train_sets[i], valid_sets[i], dat_obj, epochs = 100)
        n = CNN(dat_obj).to(globals.device)
        acc = 0
        print("TEACHER",i,"ACC",acc)
        # torch.save(n.state_dict(),f"{globals.SAVE_DIR}/{dat_obj.name}_teacher_{i}_of_{dat_obj.num_teachers-1}.tch")


        print("Model",str(i))
        n.eval()

        ballot = [] # user i's voting record: 2-axis array
        correct = 0
        guessed = 0

        data_loader = torch.utils.data.DataLoader(dat_obj.student_data, shuffle=False, batch_size=256)

        for batch, labels in data_loader:
            breakpoint()
            batch, labels = batch.to(globals.device), labels.to(globals.device)
            pred_vectors = n(batch)  # 2-axis arr of model's prediction vectors
            preds = torch.argmax(pred_vectors, dim=1)  # gets highest-value indices e.g. [2, 4, 1, 1, 5, ...]
            correct_arr = torch.eq(preds, labels.argmax(dim=1))  # compare to true labels, e.g. [True, False, False, ...]
            correct += torch.sum(correct_arr)  # finds number of correct labels
            guessed += len(batch)
            ballot.append(preds.to(torch.device('cpu')))
        
        ballot = np.concatenate(ballot)
        if isfile(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy"):
            votes = np.load(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
            votes = np.append(votes, ballot)
        else:
            votes = ballot
        np.save(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", votes)

        print(f"teacher {i}'s accuracy:", correct/guessed)

        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")
    all_votes = np.load(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    correct_shape_votes = all_votes.reshape((dat_obj.num_teachers, len(dat_obj.student_data)))
    np.save(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy",correct_shape_votes)
def main():
    dat_obj = globals.dataset
    train_all(dat_obj)

if __name__ == '__main__':
    main()
