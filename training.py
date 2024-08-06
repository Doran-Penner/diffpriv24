import torch
from torch import nn, optim
from models import CNN
import globals
from copy import deepcopy
import helper
import math
import time
import os


def eval_model_preds(network, data_loader, device):
    """
    Outputs a tuple of the model's predictions and its accuracy;
    meant for checking validation accuracy and final test accuracy
    rather than training the model. Note that this function does not
    use model.train() or model.eval() --- the calling code needs to handle that.
    Assumes that the network is on `device` and labels are vectors.
    """
    preds = []
    total_correct = torch.zeros((), dtype=int)
    total_seen = torch.zeros((), dtype=int)
    with torch.no_grad():
        for batch_xs, labels in data_loader:
            batch_preds = network(batch_xs.to(device))
            preds.append(batch_preds.to(torch.device("cpu")))
            total_seen += len(batch_xs)
            total_correct += (
                (batch_preds.argmax(dim=1).to(labels.device)) == labels.argmax(dim=1)
            ).sum()
    return torch.cat(preds), (total_correct / total_seen)


# this is currenlty broken since globals.prefix no longer exists, but we're not using it anymore so :shrug:
def train(training_data, valid_data, dat_obj, lr=1e-3, epochs=70, batch_size=16, momentum=0.9, model="teacher", arch=CNN):
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
    

    network = arch(dat_obj).to(globals.device)
    opt = optim.SGD(network.parameters(), lr=lr, momentum=momentum, nesterov=True)
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
                batch_xs = batch_xs.to(globals.device)
                batch_ys = batch_ys.to(globals.device)
                preds = network(batch_xs)
                accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
            acc = torch.tensor(accs).mean()
            print("Valid acc:",acc)
            valid_accs.append(acc)
            # FIXME used to have globals.prefix, now what?
            torch.save(network.state_dict(),f"{globals.SAVE_DIR}/{dat_obj.name}_{model}_{i}.ckp")
            ### end check
        network.train()
        train_acc = []
        for batch_xs, batch_ys in train_loader:
            opt.zero_grad()
            batch_xs = batch_xs.to(globals.device)
            batch_ys = batch_ys.to(globals.device)

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
    # FIXME prefix
    st_dict = torch.load(f"{globals.SAVE_DIR}/{dat_obj.name}_{model}_{best_num_epochs}.ckp",map_location=globals.device)
    network.load_state_dict(st_dict)
    
    network.eval()
    accs = []
    for batch_xs, batch_ys in valid_loader:
        batch_xs = batch_xs.to(globals.device)
        batch_ys = batch_ys.to(globals.device)
        preds = network(batch_xs)
        accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
    acc = torch.tensor(accs).mean()
    return (network, acc)

def train_ssl(training_data, unlabeled_data, valid_data, dat_obj, confidence_threshold, num_rounds=-1, lr=1e-3, epochs=70, batch_size=16, momentum=0.9, arch=CNN):
    """TODO: Document"""
    r = 0
    self_labeled = deepcopy(unlabeled_data)
    self_labeled.indices = []
    while r < num_rounds or num_rounds == -1:
        print("Starting round",r)
        print("Training data",len(training_data))
        print("Self labeled",len(self_labeled))
        print("Unlabeled",len(unlabeled_data))
        r += 1
        # train model
        net, _ = train(torch.utils.data.ConcatDataset((training_data, self_labeled)), valid_data, dat_obj, lr=lr, epochs=epochs, batch_size=batch_size, momentum=momentum, model="student", arch=arch)
        # label unlabeled data
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_data, shuffle=False, batch_size=batch_size)
        net.eval()
        to_remove = []
        with torch.no_grad():
            preds = torch.Tensor([])
            for batch_xs, _ in unlabeled_loader:
                batch_xs = batch_xs.to(globals.device)
                preds = torch.cat((preds,net(batch_xs).to('cpu')))
            del net
            for i, p in enumerate(preds):
                _p = torch.softmax(p, dim=0)
                if torch.max(_p[1]) >= confidence_threshold:
                    to_remove.append(i)
                    self_labeled.indices.append(unlabeled_data.indices[i])
                    label = torch.eye(dat_obj.num_labels)[torch.argmax(_p)]
                    print("Overwriting with label",label)
                    self_labeled.dataset.labels[unlabeled_data.indices[i]] = label
        # remove labeled data from unlabeled set
        mask = torch.ones(len(unlabeled_data), dtype=torch.bool)
        mask[to_remove] = False
        unlabeled_data.indices = unlabeled_data.indices[mask]
        if len(unlabeled_data) == 0:
            break
    return train(torch.cat((training_data, self_labeled)), valid_data, dat_obj, lr=lr, epochs=epochs, batch_size=batch_size, momentum=momentum, model="student", arch=arch)

def train_fm(labeled_data, unlabeled_data, valid_data, dat_obj, lr=1e-3, epochs=70, batch_size=16, max_unlab_ratio=7, momentum=0.9, model="teacher", arch=CNN, tau = 0.95, lmbd = 1):
    """
    This is a function that trains the model on a specified dataset beep boop fixmatch.
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

    # (hopefully) unique temporary filename for saving best model
    savefile = f"{globals.SAVE_DIR}/{str(time.time_ns())}_{str(os.getpid())}.ckp"

    # calculate batch size for unlab_loader to sync up with train_loader
    num_iters = math.ceil(len(labeled_data) / batch_size)
    unlab_batch_size = len(unlabeled_data) // num_iters
    # for ease of computation, we shrink the unlabeled dataset
    unlab_batch_size = min(unlab_batch_size, max_unlab_ratio * batch_size)

    train_loader = torch.utils.data.DataLoader(labeled_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    unlab_loader = torch.utils.data.DataLoader(unlabeled_data, shuffle=True, batch_size=unlab_batch_size)

    network = arch(dat_obj).to(globals.device)
    opt = optim.SGD(network.parameters(), lr=lr, momentum=momentum, nesterov=True)
    loss = nn.CrossEntropyLoss()

    train_accs = []

    best_valid_acc = 0  # for comparison
    best_num_epochs = 0  # for our info
    torch.save(network.state_dict(), savefile)  # initialize in case model never improves

    _train_accs = []
    _valid_accs = []
    _unlab_accs = []

    network.train()
    for i in range(epochs + 1):  # +1 to get final valid acc at i == epochs
        ### BEGIN DEBUGGING BLOCK
        network.eval()
        _, _train_acc = eval_model_preds(network, train_loader, globals.device)
        _, _valid_acc = eval_model_preds(network, valid_loader, globals.device)
        _, _unlab_acc = eval_model_preds(network, unlab_loader, globals.device)
        _train_accs.append(_train_acc)
        _valid_accs.append(_valid_acc)
        _unlab_accs.append(_unlab_acc)
        network.train()
        ### END DEBUGGING BLOCK
        if i % 5 == 0:
            print(f"Epoch {i}")
            ### check valid accuracy
            network.eval()
            _, acc = eval_model_preds(network, valid_loader, globals.device)
            print(f"Valid acc: {acc:0.4f}")
            if acc > best_valid_acc:
                torch.save(network.state_dict(), savefile)
                best_valid_acc = acc
                best_num_epochs = i
            ### end check
            if i == epochs:
                break
            network.train()
        train_acc = []
        unsuper_acc = []
        for train_data, unlab_data in zip(train_loader, unlab_loader):
            batch_xs, batch_ys = train_data
            unlab_batch_xs, _ = unlab_data

            opt.zero_grad()
            batch_xs = batch_xs.to(globals.device)
            batch_ys = batch_ys.to(globals.device).argmax(dim=1)
            unlab_batch_xs = unlab_batch_xs.to(globals.device)

            preds = network(helper.weak_augment(batch_xs, dat_obj))
            super_acc = (preds.argmax(dim=1) == batch_ys).float().mean()

            super_loss = loss(preds, batch_ys) / len(preds)

            weak_preds = network(helper.weak_augment(unlab_batch_xs, dat_obj))
            
            weak_preds_max, weak_preds_argmax = weak_preds.max(dim=1)
            confident_unlab_indices = weak_preds_max >= tau

            unlab_batch_new = unlab_batch_xs[confident_unlab_indices]
            weak_preds_new = weak_preds_argmax[confident_unlab_indices]

            strong_preds = network(helper.strong_augment(unlab_batch_new))

            uns_acc = (strong_preds.argmax(dim = 1) == weak_preds_new).float().mean()

            train_acc.append(super_acc)
            unsuper_acc.append(uns_acc)

            unsuper_loss = loss(strong_preds, weak_preds_new) / len(unlab_batch_xs)
            
            loss_val = super_loss + (unsuper_loss * lmbd)
            loss_val.backward()
            opt.step()

        acc = torch.tensor(train_acc).mean()
        un_acc = torch.tensor(unsuper_acc).mean()
        print(acc, un_acc)  # see trianing accuracy
        train_accs.append(acc)
    
    ### BEGIN DEBUGGING BLOCK
    import csv
    with open("debugging.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow((f"learningrate={lr} lambda={lmbd}", "train acc", "valid acc", "unlab acc"))
        for epoch, (t, v, u) in enumerate(zip(_train_accs, _valid_accs, _unlab_accs)):
            writer.writerow((epoch, float(t), float(v), float(u)))
    breakpoint()
    ### END DEBUGGING BLOCK

    print("Final num epochs:", best_num_epochs)

    st_dict = torch.load(savefile, map_location=globals.device)
    network.load_state_dict(st_dict)

    os.remove(savefile)
    return (network, best_valid_acc)
