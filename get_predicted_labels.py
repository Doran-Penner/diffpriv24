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
        model = CNN().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        ballot = [] # user i's voting record: 2-axis array
        correct = 0
        guessed = 0

        for batch, labels in data:
            batch, labels = batch.to(device), labels.to(device)
            pred_vectors = model(batch)  # 2-axis arr of model's prediction vectors
            preds = torch.argmax(pred_vectors, dim=1)  # gets highest-value indices e.g. [2, 4, 1, 1, 5, ...]
            correct_arr = torch.eq(preds, labels)  # compare to true labels, e.g. [True, False, False, ...]
            correct += torch.sum(correct_arr)  # finds number of correct labels
            guessed += len(batch)
            ballot.append(preds.to(torch.device('cpu')))
        
        ballot = np.concatenate(ballot)

        votes.append(ballot)
        print(f"teacher {i}'s accuracy:", correct/guessed)
    np.save(f"./saved/{dataset}_{num_models}_teacher_predictions.npy", np.asarray(votes))
    print("done with the predictions!")


def load_predicted_labels(aggregator, dataset_name="svhn", num_models=250):
    votes = np.load(f"./saved/{dataset_name}_{num_models}_teacher_predictions.npy", allow_pickle=True)
    # NOTE hard-coded noise scale in line below, change sometime
    agg = lambda x: aggregator.threshold_aggregate(x, 1)  # noqa: E731
    return np.apply_along_axis(agg, 0, votes)  # yay parallelizing!


def main():
    # change these or pass variables in the future
    dataset = 'svhn'
    num_teachers = 250
    agg = aggregate.NoisyMaxAggregator(1.1, noise_fn=np.random.normal)

    train, valid, _test = load_dataset(dataset, 'student', False)
    train = torch.utils.data.ConcatDataset([train, valid])
    loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=256)

    if not isfile(f"./saved/{dataset}_{num_teachers}_teacher_predictions.npy"):
        calculate_prediction_matrix(loader, device, dataset, num_teachers)
    
    labels = load_predicted_labels(agg, dataset, num_teachers)

    correct = 0
    guessed = 0
    for i, label in enumerate(labels):
        guessed += 1
        if label == train[i][1]:
            correct += 1
    print("label accuracy:", correct/guessed)

    np.save(f'./saved/{dataset}_{num_teachers}_agg_teacher_predictions.npy', labels)


if __name__ == "__main__":
    main()
