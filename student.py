import torch
import numpy as np
from torch_teachers import train
from helper import load_dataset, device

dataset = load_dataset('svhn', 'student', False)

train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
train_set = torch.utils.data.Subset(dataset, range(train_size))
valid_set = torch.utils.data.Subset(dataset, range(train_size, train_size + valid_size))
test_set = torch.utils.data.Subset(dataset, range(train_size + valid_size, len(dataset)))

batch_size = 64

labels = np.load("./saved/teacher_predictions.npy", allow_pickle=True)

train_labels = labels[:train_size]
valid_labels = labels[train_size:train_size + valid_size]

train_set.targets = train_labels
valid_set.targets = valid_labels

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
print("TEST ACCURACY:",acc)

torch.save(n.state_dict(), "./saved/svhn_student.stu")