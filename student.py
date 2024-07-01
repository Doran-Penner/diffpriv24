import torch
import numpy as np
from torch_teachers import train
from helper import load_dataset, device

train_set, valid_set, test_set = load_dataset('svhn', 'student', False)

batch_size = 64

labels = np.load("./saved/svhn_250_agg_teacher_predictions.npy", allow_pickle=True)

print(len(labels))
print(len(valid_set))

train_labels = labels[:len(train_set)]
valid_labels = labels[len(train_set):len(train_set) + len(valid_set)]

print(np.sum(train_set.dataset.labels[:len(train_set)] == train_labels))

train_set.dataset.labels[:len(train_set)] = train_labels
valid_set.dataset.labels[len(train_set):len(train_set) + len(valid_set)] = valid_labels

(n, acc) = train(train_set, valid_set, 'svhn', device=device)
print("Validation Accuracy:", acc)

test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size)

accs = []
for batch_xs, batch_ys in test_loader:
    batch_xs = batch_xs.to(device)
    batch_ys = batch_ys.to(device)
    preds = n(batch_xs)
    accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
acc = torch.tensor(accs).mean()
# print("TEST ACCURACY:",acc)  # we don't see that :)

torch.save(n.state_dict(), "./saved/svhn_student.stu")
