import numpy as np
import aggregate
import globals


def load_predicted_labels(aggregator, votes, dat_obj, max_epsilon):
    """
    Function for loading and aggregating the predicted labels from the teacher prediction matrix.
    :param aggregator: aggregate.Aggregator subclass used for the private mechanism
    :param votes: (num_teachers, num_datapoints)-shape array of teacher votes
    :param dat_obj: datasets._Dataset subclass which represents the dataset being labeled
    :param max_epsilon: maximum epsilon budget which the aggregation will not exceed
    :returns: list containing the privately aggregated labels
    """
    labels = []
    hist = np.zeros(10)
    for vote in votes.T:
        label = aggregator.threshold_aggregate(vote,max_epsilon)
        labels.append(aggregator.threshold_aggregate(vote, max_epsilon))
        if not np.any(label == None):
            hist += np.float64(label)
    print(hist)
    labels = np.asarray(labels)
    return labels


def main():
    """
    Aggregate the teacher predictions (assumed to be already calculated
    and in a local file) via a given aggregation mechanism.
    """

    dat_obj = globals.dataset
    max_epsilon = 10
    agg = aggregate.ConfidentGNMax(100,35,195,dat_obj)

    student_data = dat_obj.student_data
    
    votes = np.load(f"{globals.SAVE_DIR}/{globals.prefix}_{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    
    labels = load_predicted_labels(agg, votes, dat_obj, max_epsilon)
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
    
    np.save(f'{globals.SAVE_DIR}/{globals.prefix}_{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)

if __name__ == "__main__":
    main()
