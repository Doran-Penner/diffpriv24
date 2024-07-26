import torch
from torch import nn, optim
from models import CNN
import globals
from copy import deepcopy

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
                batch_xs = batch_xs.to(globals.device)
                batch_ys = batch_ys.to(globals.device)
                preds = network(batch_xs)
                accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
            acc = torch.tensor(accs).mean()
            print("Valid acc:",acc)
            valid_accs.append(acc)
            torch.save(network.state_dict(),f"./saved/{globals.prefix}_{dat_obj.name}_{model}_{i}.ckp")
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
    st_dict = torch.load(f"./saved/{globals.prefix}_{dat_obj.name}_{model}_{best_num_epochs}.ckp",map_location=globals.device)
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
