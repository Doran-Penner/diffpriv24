import numpy as np
import torch
from models import CNN
import aggregate
from os.path import isfile
import globals


def calculate_prediction_matrix(data_loader, dat_obj):
    """
    Function for calculating a numpy matrix representing each teacher's vote on each query.
    :param data: DataLoader for all of the data that the student is going to train on (both student training and 
                 student valid)
    :param dat_obj: datasets._Dataset subclass which represents the dataset being labelled
    :returns: nothing, but saves a file containing the teachers' predictions
    """
    votes = [] # final voting record
    for i in range(dat_obj.num_teachers):
        print("Model",str(i))
        state_dict = torch.load(f'./saved/{dat_obj.name}_teacher_{i}_of_{dat_obj.num_teachers-1}.tch',map_location=globals.device)
        model = CNN(dat_obj).to(globals.device)
        model.load_state_dict(state_dict)
        model.eval()

        ballot = [] # user i's voting record: 2-axis array
        correct = 0
        guessed = 0

        for batch, labels in data_loader:
            batch, labels = batch.to(globals.device), labels.to(globals.device)
            pred_vectors = model(batch)  # 2-axis arr of model's prediction vectors
            preds = torch.argmax(pred_vectors, dim=1)  # gets highest-value indices e.g. [2, 4, 1, 1, 5, ...]
            correct_arr = torch.eq(preds, labels)  # compare to true labels, e.g. [True, False, False, ...]
            correct += torch.sum(correct_arr)  # finds number of correct labels
            guessed += len(batch)
            ballot.append(preds.to(torch.device('cpu')))
        
        ballot = np.concatenate(ballot)

        votes.append(ballot)
        print(f"teacher {i}'s accuracy:", correct/guessed)
    np.save(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", np.asarray(votes))
    print("done with the predictions!")


def load_predicted_labels(aggregator, dat_obj, max_epsilon):
    """
    Function for loading and aggregatingthe predicted labels from the matrix created by 
    `calculate_prediction_matrix()`.
    :param aggregator: aggregate.Aggregator subclass used for the private mechanism
    :param dat_obj: datasets._Dataset subclass which represents the dataset being labelled
    :returns: list containing the privately aggregated labels
    """
    votes = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    labels = []
    for vote in votes.T:
        labels.append(aggregator.threshold_aggregate(vote, max_epsilon))
    labels = np.asarray(labels)
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)
    return labels


def main():
    """
    Aggregate the teacher predictions (assumed to be already calculated
    and in a local file) via a given aggregation mechanism.
    Note: the `calculate_prediction_matrix` functionality is
    planned to be moved into `torch_teachers.train_all` soon!
    """

    dat_obj = globals.dataset
    max_epsilon = 10
    agg = aggregate.NoisyVectorAggregator(50, dat_obj, alpha_set=list(range(2,21)))

    student_data = dat_obj.student_data
    loader = torch.utils.data.DataLoader(student_data, shuffle=False, batch_size=256)

    if not isfile(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy"):
        calculate_prediction_matrix(loader, dat_obj)
    
    labels = load_predicted_labels(agg, dat_obj, max_epsilon)
    # safe access of tau_tally without crashing
    if (tau_usages := getattr(agg, "tau_tally", None)) is not None:
        print("FINAL tau usages:", tau_usages)

    if len(labels.shape) == 1:
        # turn all vectors into 1-hot; for "backwards compatibility" with our other aggregation mechanisms
        label_vecs = np.full((len(labels), dat_obj.num_labels), None)
        eye = np.eye(dat_obj.num_labels)
        label_vecs[labels != None] = eye[labels[labels != None].astype(int)]  # noqa: E711
        labels = label_vecs
    
    # this is a bit weird, but seems to work
    full_len = len(labels)
    # get indices of labeled datapoints, for use later
    which_labeled = np.arange(full_len)[np.all(labels != None, axis=1)]  # noqa: E711
    labeled_labels = labels[which_labeled]
    labeled_len = len(labeled_labels)

    correct = 0
    for i, label in zip(which_labeled, labeled_labels):
        if label.argmax() == student_data[i][1].argmax():
            correct += 1

    print(f"data points labeled: {labeled_len} out of {full_len} ({labeled_len / full_len:0.3f})")
    if labeled_len != 0:
        print(f"label accuracy on labeled data: {correct/labeled_len:0.3f}")
    
    # duplicated save but whatever, in case the function changes
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)

if __name__ == "__main__":
    main()
