import torch
import numpy as np
from torch_teachers import train
from helper import load_dataset, device

def load_and_part_sets(dataset, num_teachers):
    """
    Function that loads datasets according to the needs of the student model.
    :param dataset: string representing the dataset that we want to load (one of `'mnist'` or `'svhn'`)
    :param num_teachers: int representing the number of teachers models that we were training
    :returns: datasets representing the training data, validation data, and test data for the student model. training
              and validation data each use predicted labels, and test data uses true labels.
    """
    train_set, valid_set, test_set = load_dataset(dataset, 'student', False)
    labels = np.load(f"./saved/{dataset}_{num_teachers}_agg_teacher_predictions.npy", allow_pickle=True)
    labels = list(filter(lambda x: x != -1, labels))
    label_len = min(len(labels),len(train_set) + len(valid_set))
    
    train_set.indices = list(filter(lambda i: i < label_len, train_set.indices))
    valid_set.indices = list(filter(lambda i: i < label_len, valid_set.indices))
    
    joint_set = torch.utils.data.ConcatDataset([train_set, valid_set])
    joint_set.datasets[0].dataset.labels[:label_len] = labels[:label_len]  # NOTE this is very sketchy
    train_set, valid_set = torch.utils.data.random_split(joint_set, [0.8, 0.2])
    return train_set, valid_set, test_set

def calculate_test_accuracy(network, test_data):
    """
    Function to calculate the accuracy of the student model on the test data
    :param network: student model
    :param test_data: dataset containing the test data
    :returns: number representing the accuracy of the student model on the test data
    """
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    accs = []
    for batch_xs, batch_ys in test_loader:
        batch_xs = batch_xs.to(device)
        batch_ys = batch_ys.to(device)
        preds = network(batch_xs)
        accs.append((preds.argmax(dim=1) == batch_ys).float())
    acc = torch.cat(accs).mean()
    return acc  # we don't see that :)

def main():
    # this is where we set the parameters that are used by the functions in this file (ie, if we
    # want to use a different database, we would change it here)
    dataset = 'svhn'
    num_teachers = 250

    train_set, valid_set, test_set = load_and_part_sets(dataset, num_teachers)

    n, val_acc = train(train_set, valid_set, dataset, device=device, epochs=200, model="student")

    print(f"Validation Accuracy: {val_acc}")
    test_acc = calculate_test_accuracy(n, test_set)
    print(f"Test Accuracy: {test_acc}")
    torch.save(n.state_dict(), f"./saved/{dataset}_student_final.ckp")

if __name__ == '__main__':
    main()