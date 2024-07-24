import numpy as np
import globals
import pickle

# NOTE: these are random utility functions for convenience,
# not part of the pipeline and not guaranteed to work!


##### prints average teacher accuracy #####
def avg_teacher_accuracy():
    """
    Function for checking average teacher accuracy on the test database
    :returns: nothing, but prints average teacher accuract on the test database
    """
    teach_test = globals.dataset.teach_test
    teacher_preds = np.load("./saved/svhn_250_teacher_predictions.npy", allow_pickle=False)
    teacher_acc = np.empty((len(teach_test),))

    for i in range(len(teach_test)):
        _, label = teach_test[i]
        guess = teacher_preds[i]
        teacher_acc[i] = guess == label

    print(np.average(teacher_acc))
