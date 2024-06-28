import numpy as np
import torch
from models import CNN
import aggregate
from os.path import isfile
from helper import load_dataset, device


def calculate_prediction_matrix(data, device, dataset='svhn', num_models=250):
    votes = [] # final voting record
    for i in range(num_models):
        print("Model",str(i))
        state_dict = torch.load(f'./saved/{dataset}_teacher_{i}_of_{num_models-1}.tch',map_location=device)
        m = CNN().to(device)
        m.load_state_dict(state_dict)
        m.eval()

        ballot = [] # user i's voting record
        correct = 0
        guessed = 0

        for p, l in data:
            out = m(p.to(device))
            for (j,row) in enumerate(out):
                res = torch.argmax(row)
                if res == l[j]:
                    correct += 1
                guessed += 1
                ballot.append(res.to(torch.device('cpu')))
        votes.append(ballot)
        print(correct/guessed)
    np.save(f"./saved/{dataset}_{num_models}_teacher_predictions.npy", np.asarray(votes))
    print("done with the predictions!")


def load_predicted_labels(aggregator, dataset_name="svhn", num_models=250):
    votes = np.load(f"./saved/{dataset_name}_{num_models}_teacher_predictions.npy", allow_pickle=True)
    labels = []
    for prop in torch.transpose(torch.from_numpy(votes),0,1):
        labels.append(aggregator.threshold_aggregate(prop, 1))
    return labels


def main():
    train, _valid, _test = load_dataset('svhn', 'student', False)
    loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=64)

    if not isfile("./saved/svhn_250_teacher_predictions.npy"):
        calculate_prediction_matrix(loader, device)
    
    agg = aggregate.NoisyMaxAggregator(1)
    labels = load_predicted_labels(agg)

    label_arr = np.asarray(labels)
    np.save('./saved/svhn_250_agg_teacher_predictions.npy', label_arr)


if __name__ == "__main__":
    main()
