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
    :param device: string representing the device that the code is being run on so that pytorch can properly
                   optimize
    :param dataset: string representing the dataset that is being labeled (one of `'svhn'` or `'mnist'`)
    :param num_models: the number of teacher models whose predictions we're getting
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


def load_predicted_labels(aggregator, dat_obj):
    """
    Function for loading and aggregatingthe predicted labels from the matrix created by 
    `calculate_prediction_matrix()`.
    :param aggregator: aggregate.Aggregator subclass used for the private mechanism
    :param dataset_name: string used to represent which dataset is being trained on, one
                         of `'svhn'` or `'mnist'`
    :param num_models: int representing the number of teacher models being used
    :returns: list containing the privately aggregated labels
    """
    votes = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    ### BEGIN insert
    # votes = votes.transpose()
    # agg = aggregate.L1Exp(num_labels=10, epsilon=100, total_num_queries=len(votes))
    # return agg.threshold_aggregate(votes)
    ### END insert
    # # NOTE hard-coded epsilon in line below, change sometime
    agg = lambda x: aggregator.threshold_aggregate(x, 10)  # noqa: E731
    labels = np.apply_along_axis(agg, 0, votes)
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)
    return labels


def main():
    # PARAMTER LIST (incomplete?)
    # alpha (not optimizing yet)
    # epsilon threshold
    # delta (fixed for calculation, not optimized)
    # sigma {1,2}
    # p (subsample chance)
    # tau (threshold)

    # change these or pass variables in the future

    dat_obj = globals.dataset
    agg = aggregate.PartRepeatGNMax(
        GNMax_scale=500,
        p=0.8,
        tau=50,
        dat_obj=dat_obj,
        max_num=1000,
        confident=True,
        lap_scale=100,
        GNMax_epsilon=5,
    )

    student_data = dat_obj.student_data
    loader = torch.utils.data.DataLoader(student_data, shuffle=False, batch_size=256)

    if not isfile(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy"):
        calculate_prediction_matrix(loader, dat_obj)
    
    labels = load_predicted_labels(agg, dat_obj)
    print("FINAL tau usages:", agg.tau_tally)

    correct = 0
    guessed = 0
    unlabeled = 0
    for i, label in enumerate(labels):
        guessed += 1
        if label == student_data[i][1]:
            correct += 1
        if label == -1:
            unlabeled += 1
    print("data points labeled:", guessed-unlabeled)
    print("label accuracy:", correct/guessed)
    if unlabeled != guessed:
        print("label accuracy ON LABELED DATA:", correct/(guessed - unlabeled))

if __name__ == "__main__":
    main()
