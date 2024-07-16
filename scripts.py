import numpy as np
import helper
import pickle
import csv

# NOTE: these are random utility functions for convenience,
# not part of the pipeline and not guaranteed to work!


##### prints average teacher accuracy #####
def avg_teacher_accuracy():
    """
    Function for checking average teacher accuracy on the test database
    :returns: nothing, but prints average teacher accuract on the test database
    """
    teach_test = helper.dataset.teach_test
    teacher_preds = np.load("./saved/svhn_250_teacher_predictions.npy", allow_pickle=False)
    teacher_acc = np.empty((len(teach_test),))\

    for i in range(len(teach_test)):
        _, label = teach_test[i]
        guess = teacher_preds[i]
        teacher_acc[i] = guess == label

    print(np.average(teacher_acc))


##### creates csv from gnmmax_optimizer run, for use in google sheets
def results_csv():
    with open("saved/rep_gnmax_points.pkl", "rb") as f:
        results = pickle.load(f)

    lst = []
    for x in results.items():
        (
            a,
            b,
        ) = x
        lst.append(list(a) + list(b))
    for i, x in enumerate(lst):
        lst[i] = list(map(lambda y: float(y), x))

    with open("saved/optimize.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for x in lst:
            writer.writerow(x)
