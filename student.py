import torch
import numpy as np
from torch_teachers import train
import globals

def calculate_test_accuracy(network, test_data):
    """
    Function to calculate the accuracy of the student model on the test data
    :param network: student model
    :param test_data: dataset containing the test data
    :returns: number representing the accuracy of the student model on the test data
    """
    network.eval()
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    accs = []
    for batch_xs, batch_ys in test_loader:
        batch_xs = batch_xs.to(globals.device)
        batch_ys = batch_ys.to(globals.device)
        preds = network(batch_xs)
        accs.append((preds.argmax(dim=1) == batch_ys).float())
    acc = torch.cat(accs).mean()
    return acc  # we don't see that :)

def main():
    # this is where we set the parameters that are used by the functions in this file (ie, if we
    # want to use a different database, we would change it here)
    ds = globals.dataset
    dataset_name = ds.name
    num_teachers = ds.num_teachers

    labels = np.load(f"./saved/{dataset_name}_{num_teachers}_agg_teacher_predictions.npy", allow_pickle=True)

    train_set, valid_set = ds.student_overwrite_labels(labels)
    test_set = ds.student_test

    n, val_acc = train(train_set, valid_set, ds, epochs=200, model="student")

    print(f"Validation Accuracy: {val_acc}")
    test_acc = calculate_test_accuracy(n, test_set)
    print(f"Test Accuracy: {test_acc}")
    torch.save(n.state_dict(), f"./saved/{dataset_name}_student_final.ckp")

if __name__ == '__main__':
    main()
