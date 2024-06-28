import numpy as np
from helper import load_dataset

##### prints average teacher accuracy #####
def avg_teacher_accuracy():
    _train, _valid, test = load_dataset('svhn', 'teach', False)
    teacher_preds = np.load("./saved/teacher_predictions.npy", allow_pickle=False)
    teacher_acc = np.empty((len(test),))

    for i in range(len(test)):
        _, label = test[i]
        guess = teacher_preds[i]
        teacher_acc[i] = (guess == label)

    print(np.average(teacher_acc))
