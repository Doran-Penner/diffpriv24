import torch
import time
import globals
import numpy as np
from os.path import isfile
from training import train, train_fm


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
        n, acc = train(train_sets[i], valid_sets[i], dat_obj, epochs = 100)
        print("TEACHER",i,"ACC",acc)
        # torch.save(n.state_dict(),f"{globals.SAVE_DIR}/{dat_obj.name}_teacher_{i}_of_{dat_obj.num_teachers-1}.tch")


        print("Model",str(i))
        n.eval()

        ballot = [] # user i's voting record: 2-axis array
        correct = 0
        guessed = 0

        data_loader = torch.utils.data.DataLoader(dat_obj.student_data, shuffle=False, batch_size=256)

        for batch, labels in data_loader:
            batch, labels = batch.to(globals.device), labels.to(globals.device)
            pred_vectors = n(batch)  # 2-axis arr of model's prediction vectors
            preds = torch.argmax(pred_vectors, dim=1)  # gets highest-value indices e.g. [2, 4, 1, 1, 5, ...]
            correct_arr = torch.eq(preds, labels)  # compare to true labels, e.g. [True, False, False, ...]
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

def train_all_fm(dat_obj):
    """
    This function trains all of the teacher models on the specified dataset
    :param dat_obj: datasets._Dataset object representing the dataset being trained on
    :return: Does not return anything, but saves the models instead
    """
    train_sets = dat_obj.teach_train
    valid_sets = dat_obj.teach_valid
    unlab_set = dat_obj.student_data

    file_name = f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_fm_teacher_predictions.npy"
    np.save(file_name, np.arange(0))  # initialize to empty file, overwriting previous work

    for i in range(dat_obj.num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        n, acc = train_fm(train_sets[i], unlab_set, valid_sets[i], dat_obj, lr=0.03, epochs = 100, lmbd=1)
        print(f"TEACHER {i} FINAL VALID ACC: {acc:0.4f}")

        n.eval()
        ballot = []
        correct = 0
        unlab_loader = torch.utils.data.DataLoader(unlab_set, shuffle=False, batch_size=256)
        
        for batch, labels in unlab_loader:
            preds = (
                n(batch.to(globals.device))
                .argmax(dim=1)
                .to(torch.device("cpu"))
            )
            correct += (preds == labels.argmax(dim=1)).sum()
            ballot.append(preds)

        ballot = np.concatenate(ballot)
        votes = np.load(file_name)
        votes = np.append(votes, ballot)  # NOTE this is still 1d array, we reshape it at the end
        np.save(file_name, votes)

        print(f"teacher {i}'s accuracy: {correct / len(unlab_set):0.4f}")

        duration = time.time() - start_time
        print(f"It took {duration // 60} minutes and {duration % 60} seconds to train teacher {i}.")
    # at the end, re-shape the saved data
    all_votes = np.load(file_name, allow_pickle=True)
    correct_shape_votes = all_votes.reshape((dat_obj.num_teachers, len(dat_obj.student_data)))
    np.save(file_name, correct_shape_votes)

def main():
    dat_obj = globals.dataset
    train_all_fm(dat_obj)

if __name__ == '__main__':
    main()
