import torch
import numpy as np
from torch_teachers import train
from helper import load_dataset, device

train_set, valid_set, test_set = load_dataset('svhn', 'student', False)

batch_size = 64

labels = np.load("./saved/svhn_250_agg_teacher_predictions.npy", allow_pickle=True)

labels = list(filter(lambda x: x != -1, labels))
label_len = min(len(labels),len(train_set) + len(valid_set))

train_set.indices = list(filter(lambda i: i < label_len, train_set.indices))
valid_set.indices = list(filter(lambda i: i < label_len, valid_set.indices))

joint_set = torch.utils.data.ConcatDataset([train_set, valid_set])
joint_set.datasets[0].dataset.labels[:label_len] = labels[:label_len]  # NOTE this is very sketchy
train_set, valid_set = torch.utils.data.random_split(joint_set, [0.8, 0.2])

(n, acc) = train(train_set, valid_set, 'svhn', device=device)
print("Validation Accuracy:", acc)

test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size)

accs = []
for batch_xs, batch_ys in test_loader:
    batch_xs = batch_xs.to(device)
    batch_ys = batch_ys.to(device)
    preds = n(batch_xs)
    accs.append((preds.argmax(dim=1) == batch_ys).float())
acc = torch.cat(accs).mean()
print("TEST ACCURACY:",acc)  # we don't see that :)

torch.save(n.state_dict(), "./saved/svhn_student.stu")
