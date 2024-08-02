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
    for i in range(dat_obj.num_teachers):
        print(f"Training teacher {i} now!")
        start_time = time.time()
        n, acc = train_fm(train_sets[i], unlab_set, valid_sets[i], dat_obj, epochs = 100)
        print("TEACHER",i,"ACC",acc)
        # torch.save(n.state_dict(),f"{globals.SAVE_DIR}/{dat_obj.name}_teacher_{i}_of_{dat_obj.num_teachers-1}.tch")
        file_name = f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_fm_teacher_predictions.npy"

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
        if isfile(file_name):
            votes = np.load(file_name, allow_pickle=True)
            votes = np.append(votes, ballot)
        else:
            votes = ballot
        np.save(file_name, votes)

        print(f"teacher {i}'s accuracy:", correct/guessed)

        duration = time.time()- start_time
        print(f"It took {duration//60} minutes and {duration % 60} seconds to train teacher {i}.")

def main():
    dat_obj = globals.dataset
    train_all(dat_obj)

if __name__ == '__main__':
    main()
