import numpy as np
import torch
from models import CNN
import aggregate
from os.path import isfile
from helper import load_dataset, device


def calculate_prediction_matrix(data, device, dataset='svhn', num_models=250):
    """
    Function for calculating a numpy matrix representing each teacher's vote on each query.
    :param data: DataLoader for all of the data that the student is going to train on (both student training and 
                 student valid)
    :param device: string representing the device that the code is being run on so that pytorch can properly
                   optimize
    :param dataset: string representing the dataset that is being labeled (one of `'svhn'` or `'mnist'`)
    :param num_models: the number of teacher models whose predictions we're getting
    :returns: nothing, but saves a file containing the teachers' predictions
    """
    votes = [] # final voting record
    for i in range(num_models):
        print("Model",str(i))
        state_dict = torch.load(f'./saved/{dataset}_teacher_{i}_of_{num_models-1}.tch',map_location=device)
        model = CNN().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        ballot = [] # user i's voting record: 2-axis array
        correct = 0
        guessed = 0

        for batch, labels in data:
            batch, labels = batch.to(device), labels.to(device)
            pred_vectors = model(batch)  # 2-axis arr of model's prediction vectors
            preds = torch.argmax(pred_vectors, dim=1)  # gets highest-value indices e.g. [2, 4, 1, 1, 5, ...]
            correct_arr = torch.eq(preds, labels)  # compare to true labels, e.g. [True, False, False, ...]
            correct += torch.sum(correct_arr)  # finds number of correct labels
            guessed += len(batch)
            ballot.append(preds.to(torch.device('cpu')))
        
        ballot = np.concatenate(ballot)

        votes.append(ballot)
        print(f"teacher {i}'s accuracy:", correct/guessed)
    np.save(f"./saved/{dataset}_{num_models}_teacher_predictions.npy", np.asarray(votes))
    print("done with the predictions!")


def load_predicted_labels(aggregator, dataset_name="svhn", num_models=250):
    """
    Function for loading and aggregatingthe predicted labels from the matrix created by 
    `calculate_prediction_matrix()`.
    :param aggregator: aggregate.Aggregator subclass used for the private mechanism
    :param dataset_name: string used to represent which dataset is being trained on, one
                         of `'svhn'` or `'mnist'`
    :param num_models: int representing the number of teacher models being used
    :returns: list containing the privately aggregated labels
    """
    votes = np.load(f"./saved/{dataset_name}_{num_models}_teacher_predictions.npy", allow_pickle=True)
    ### BEGIN insert
    votes = votes.transpose()
    agg = aggregate.L1Exp(num_labels=10, epsilon=100, total_num_queries=len(votes))
    return agg.threshold_aggregate(votes)
    ### END insert
    # # NOTE hard-coded noise scale in line below, change sometime
    # agg = lambda x: aggregator.threshold_aggregate(x, 10)  # noqa: E731
    # return np.apply_along_axis(agg, 0, votes)  # yay parallelizing!


def main():
    # PARAMTER LIST (incomplete?)
    # alpha (not optimizing yet)
    # epsilon threshold
    # delta (fixed for calculation, not optimized)
    # sigma {1,2}
    # p (subsample chance)
    # tau (threshold)

    # change these or pass variables in the future
    dataset = 'svhn'
    num_teachers = 250
    agg = aggregate.RepeatGNMax(50, 100, 1, 50)

    train, valid, _test = load_dataset(dataset, 'student', False)
    train = torch.utils.data.ConcatDataset([train, valid])
    loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=256)

    if not isfile(f"./saved/{dataset}_{num_teachers}_teacher_predictions.npy"):
        calculate_prediction_matrix(loader, device, dataset, num_teachers)
    
    labels = load_predicted_labels(agg, dataset, num_teachers)
    print("FINAL tau usages:", agg.tau_tally)

    correct = 0
    guessed = 0
    unlabeled = 0
    for i, label in enumerate(labels):
        guessed += 1
        if label == train[i][1]:
            correct += 1
        if label == -1:
            unlabeled += 1
    print("data points labeled:", guessed-unlabeled)
    print("label accuracy:", correct/guessed)
    if unlabeled != guessed:
        print("label accuracy ON LABELED DATA:", correct/(guessed - unlabeled))

    np.save(f'./saved/{dataset}_{num_teachers}_agg_teacher_predictions.npy', labels)

if __name__ == "__main__":
    main()
