import numpy as np
import torch
from models import CNN
import torchvision
import torchvision.transforms.v2 as transforms
import aggregate


def getPredictedLabels(data, aggregator, device, dataset='svhn', num_models=250):
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
                ballot.append(res)
        votes.append(ballot)
        print(correct/guessed)
    labels = []
    for prop in torch.transpose(torch.Tensor(votes),0,1):
        labels.append(aggregator.aggregate(prop))
    return labels


def main():
    # set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(device)

    # set up transform
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    public_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(public_dataset, shuffle=False, batch_size=64)
    agg = aggregate.NoisyMaxAggregator(1)
    labels = getPredictedLabels(loader, agg, device)

    label_arr = np.asarray(labels)
    np.save('./saved/teacher_predictions.npy', label_arr)

    # with open('./saved/teacher_predictions.txt','w') as f:
    #     for label in labels:
    #         f.write(f"{label}\n")

if __name__ == "__main__":
    main()
